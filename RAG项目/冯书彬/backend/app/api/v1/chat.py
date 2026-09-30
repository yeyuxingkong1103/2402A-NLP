import asyncio
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.deps import get_current_principal
from backend.app.schemas.chat import ChatRequest, FactCorrection, RegenerateRequest
from backend.app.services.chat_service import ChatService, ConversationNotFoundError, MessageTooLongError, RegenerationLimitError

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
chat_service = ChatService()


def _run(coro):
    # FastAPI 同步端点中执行异步服务，TestClient 环境保持简单稳定。
    return asyncio.run(coro)


def _as_response(result):
    # dataclass 输出转普通 dict，便于 FastAPI JSON 序列化。
    return asdict(result)


def _require_body_user(body_user_id: str, principal: dict) -> str:
    # 所有聊天写操作以认证主体为准，禁止客户端伪造 user_id。
    user_id = str(principal.get("sub") or "")
    if not user_id or body_user_id != user_id:
        raise HTTPException(status_code=403, detail="无权访问该用户聊天数据")
    return user_id


@router.post("/conversations/{conversation_id}/messages")
def send_message(conversation_id: str, body: ChatRequest, principal: dict = Depends(get_current_principal)):
    user_id = _require_body_user(body.user_id, principal)
    try:
        return _as_response(_run(chat_service.send_message(user_id, conversation_id, body.text)))
    except MessageTooLongError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/messages/{message_id}/regenerate")
def regenerate(message_id: str, body: RegenerateRequest, principal: dict = Depends(get_current_principal)):
    user_id = _require_body_user(body.user_id, principal)
    try:
        return _as_response(_run(chat_service.regenerate_answer(user_id, message_id)))
    except RegenerationLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/conversations/{conversation_id}/facts/correct")
def correct_facts(conversation_id: str, body: FactCorrection, principal: dict = Depends(get_current_principal)):
    user_id = _require_body_user(body.user_id, principal)
    try:
        return _as_response(_run(chat_service.correct_facts(user_id, conversation_id, body)))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/conversations/{conversation_id}/restart")
def restart(conversation_id: str, body: RegenerateRequest, principal: dict = Depends(get_current_principal)):
    user_id = _require_body_user(body.user_id, principal)
    chat_service.restart_consultation(user_id, conversation_id)
    return {"restarted": True}
