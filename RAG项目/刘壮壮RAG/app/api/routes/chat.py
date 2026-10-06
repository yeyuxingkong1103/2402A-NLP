import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.config import get_settings
from app.core.llm import build_llm_from_character
from app.core.memory import ShortTermMemory
from app.core.redis_client import get_redis
from app.core.retriever import retrieve_context
from app.models.user import User
from app.schemas.conversation import ChatRequest
from app.services import character_service, chat_service, conversation_service

router = APIRouter()


@router.post("/characters/{character_id}/conversations/{conversation_id}/chat")
async def chat(
    character_id: int,
    conversation_id: int,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    char = await character_service.get_character(db, current_user.id, character_id)
    if char is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    conv = await conversation_service.get_conversation(
        db, current_user.id, character_id, conversation_id
    )
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    user_message = payload.message
    memory = ShortTermMemory(get_redis())
    llm = build_llm_from_character(char)

    async def _safe_get_history():
        try:
            return await memory.get(conversation_id)
        except Exception:
            return []

    async def _safe_append(role: str, content: str) -> None:
        try:
            await memory.append(conversation_id, role, content)
        except Exception:
            pass

    async def event_stream():
        try:
            knowledge, long_memory = await retrieve_context(character_id, user_message)
        except Exception:
            knowledge, long_memory = [], []

        history = await _safe_get_history()
        messages = chat_service.build_messages(
            char.system_prompt, history, knowledge, long_memory, user_message
        )
        await _safe_append("user", user_message)

        full_reply = ""
        try:
            async for piece in llm.chat_stream(messages):
                full_reply += piece
                yield f"data: {json.dumps({'delta': piece}, ensure_ascii=False)}\n\n"
        except Exception:
            yield f"data: {json.dumps({'error': '模型服务不可用'}, ensure_ascii=False)}\n\n"
        finally:
            if full_reply:
                await _safe_append("assistant", full_reply)
            yield "data: [DONE]\n\n"

        # 达到阈值时异步沉淀长期记忆（不阻塞响应）
        try:
            history_msgs = await _safe_get_history()
            user_msgs = len([m for m in history_msgs if m["role"] == "user"])
            if user_msgs >= 10:
                from arq import create_pool
                from arq.connections import RedisSettings

                settings = get_settings()
                redis = await create_pool(
                    RedisSettings(host=settings.redis_host, port=settings.redis_port)
                )
                await redis.enqueue_job(
                    "extract_memory", conversation_id, character_id, current_user.id
                )
                await redis.aclose()
        except Exception:
            pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")
