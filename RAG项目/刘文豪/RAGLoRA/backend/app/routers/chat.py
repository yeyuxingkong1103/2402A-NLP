# -*- coding: utf-8 -*-
"""对话路由：SSE 流式问答（主接口）+ 非流式（供脚本/压测）。

SSE 事件协议：
    event: trace    data: {"rewritten_query":..., "recall":20, "reranked":5, ...}
    event: sources  data: [{"idx":1,"source":"...","score":...,"rerank_score":...}]
    event: delta    data: {"text":"高"}
    event: done     data: {"answer":"...","total_ms":3210}
    event: error    data: {"code":"LLM_FAILED","message":"..."}
"""
import json
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..core.db import SessionLocal, get_db
from ..core.logging import get_logger
from ..deps import get_current_user
from ..services import long_memory
from ..services import memory as memory_svc
from ..services import chain, redis_extra
from .. import models, schemas

router = APIRouter(prefix="/chat", tags=["对话"])
log = get_logger("chat")


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _enforce_rate_limit(user_id: int) -> None:
    """Redis String 计数器限流：每用户每分钟 RAGLORA_RATE_LIMIT 轮（默认 30）。

    放在 _load_context 之前——被限流的请求连数据库都不该碰。
    Redis 不可用时 check_rate_limit 内部 fail-open 放行。
    """
    allowed, retry_after = redis_extra.check_rate_limit(user_id)
    if not allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"提问过于频繁，请 {retry_after} 秒后再试",
            headers={"Retry-After": str(retry_after)},
        )


def _load_context(conv_id: int, user: models.User, db: Session):
    """校验归属并取出会话、角色、短期记忆。"""
    conv = db.get(models.Conversation, conv_id)
    if not conv or conv.user_id != user.id or conv.is_deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "会话不存在")
    character = db.get(models.Character, conv.character_id)
    if not character:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "角色不存在")
    memory = memory_svc.get_context(user.id, conv.id, db)

    # 长期记忆要按用户隔离，而链路函数签名是 (character, question, memory) ——
    # 把 user_id 挂在 character 上是**刻意选择**：改签名会波及 chain 分发器
    # 与两条链路的签名一致性测试（tests/test_chain_dispatch.py）。
    # 该属性不属于 ORM 字段，仅为本次请求携带上下文。
    character._user_id = user.id
    character._conversation_id = conv.id
    return conv, character, memory


def _persist(conv_id: int, user_id: int, question: str, answer: str,
             sources: list, trace: dict, latency_ms: int) -> int | None:
    """流式结束后落库：消息 + 会话计数 + 短期记忆。

    独立开会话，避免复用已随响应结束的请求级会话。
    """
    db = SessionLocal()
    conv = None
    try:
        user_msg = models.Message(conversation_id=conv_id, role="user",
                                  content=question,
                                  rewritten_query=trace.get("rewritten_query"))
        db.add(user_msg)

        asst_msg = models.Message(conversation_id=conv_id, role="assistant",
                                  content=answer, sources_json=sources,
                                  trace_json=trace, latency_ms=latency_ms)
        db.add(asst_msg)

        conv = db.get(models.Conversation, conv_id)
        if conv:
            conv.message_count = (conv.message_count or 0) + 2
            conv.last_message_at = datetime.now()
            # 首条消息后用问题给会话命名
            if not conv.title or conv.title.startswith("与"):
                conv.title = question[:28] + ("…" if len(question) > 28 else "")
        db.commit()
        asst_id = asst_msg.id
    except Exception as e:
        db.rollback()
        log.error("消息落库失败: %s", e)
        asst_id = None
    finally:
        db.close()

    memory_svc.append_turn(user_id, conv_id, question, answer)
    # 长期记忆（跨会话，向量库）。失败不影响主流程，内部已兜底。
    long_memory.remember(user_id, conv_id, question, answer)
    # 热度榜（Redis zSet）：每完成一轮问答给角色记一次热度。
    if conv is not None:
        redis_extra.bump_character_heat(conv.character_id)
    return asst_id


# ---------------------------------------------------------------- 流式
@router.post("/stream", summary="流式对话（SSE）")
def chat_stream(body: schemas.ChatIn, db: Session = Depends(get_db),
                user: models.User = Depends(get_current_user)):
    _enforce_rate_limit(user.id)
    conv, character, memory = _load_context(body.conversation_id, user, db)
    question = body.question.strip()

    # 精排条数可被请求覆盖
    if body.top_k:
        character.rerank_top_k = max(1, min(body.top_k, 20))

    def event_stream():
        t0 = time.time()
        trace: dict = {}
        sources: list = []
        answer = ""

        for event, data in chain.ask_stream(character, question, memory):
            if event == "trace":
                trace = data
            elif event == "sources":
                sources = data
            elif event == "done":
                answer = data.get("answer", "")
            if event != "done":
                yield _sse(event, data)

        msg_id = _persist(conv.id, user.id, question, answer, sources, trace,
                          int((time.time() - t0) * 1000))
        yield _sse("done", {
            "answer": answer,
            "message_id": msg_id,
            "total_ms": int((time.time() - t0) * 1000),
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------- 非流式
@router.post("/completions", summary="非流式对话")
def chat_completions(body: schemas.ChatIn, db: Session = Depends(get_db),
                     user: models.User = Depends(get_current_user)):
    _enforce_rate_limit(user.id)
    conv, character, memory = _load_context(body.conversation_id, user, db)
    question = body.question.strip()
    if body.top_k:
        character.rerank_top_k = max(1, min(body.top_k, 20))

    result = chain.ask(character, question, memory)
    msg_id = _persist(conv.id, user.id, question, result["answer"],
                      result["sources"], result["trace"],
                      result["trace"].get("total_ms", 0))
    result["message_id"] = msg_id
    return result
