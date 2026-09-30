"""问答 SSE HTTP 接口与服务端分帧工具。"""

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app.api.chat_persistence import get_chat_store, get_short_term_memory, persist_turn
from app.api.legal_search import DOCUMENT_TYPE_TO_STORAGE
from app.auth.current_user import CurrentUser
from app.chat.chat_store import ChatSessionStore, SessionAccessDenied
from app.chat.chat_title import derive_title
from app.chat.bootstrap import build_default_chat_service
from app.errors import not_found_error
from app.memory.short_term import ShortTermMemoryStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# 保留句末标点和换行，使所有片段拼接后与原回答逐字一致。
SENTENCE_SEGMENT = re.compile(r"[^。！？!?\n]*(?:[。！？!?]+[ \t]*|\n+|$)")


def encode_sse(event: str, data: dict[str, Any]) -> str:
    """编码单个 SSE 事件，JSON 始终使用 UTF-8 中文。"""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def split_answer_chunks(answer: str, max_chars: int = 80) -> list[str]:
    """按完整句子或换行稳定分帧，禁止为满足长度而切断句子。"""
    if not answer:
        return []
    segments = [match.group(0) for match in SENTENCE_SEGMENT.finditer(answer) if match.group(0)]
    chunks: list[str] = []
    current = ""
    for segment in segments:
        if current and len(current) + len(segment) > max_chars:
            chunks.append(current)
            current = segment
        else:
            current += segment
    if current:
        chunks.append(current)
    return chunks


class ChatOptions(BaseModel):
    top_k: int = Field(default=8, ge=1, le=50)
    enable_query_rewrite: bool = True
    enable_long_term_memory: bool = True
    jurisdiction: str = Field(default="中国大陆", min_length=1)
    as_of_date: date | None = None
    document_types: list[Literal[
        "law",
        "administrative_regulation",
        "judicial_interpretation",
        "case",
    ]] = Field(default_factory=list)


class ChatStreamRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(min_length=1, max_length=128)
    character_id: str = "legal-assistant"
    message: str = Field(min_length=1, max_length=10000)
    options: ChatOptions = Field(default_factory=ChatOptions)
    user_id: str | None = None


@router.post("/stream")
def stream_chat(
    payload: ChatStreamRequest,
    current_user: CurrentUser,
    request: Request,
) -> StreamingResponse:
    """返回服务端分帧 SSE；LLM 客户端当前仍是非流式调用。"""
    request_id = request.state.request_id
    message_id = f"message_{uuid.uuid4().hex}"
    service = _get_chat_service(request)
    # 任务 5.4：流开始前完成会话归属校验 / 首问自动建档（此时错误还是普通 HTTP 响应）
    chat_store = get_chat_store(request)
    try:
        # 已有会话：SQL 层校验归属，他人会话抛 SessionAccessDenied；
        # 新会话：以当前用户身份建档，标题取首条提问（清洗规则见 chat_title）
        chat_store.get_or_create_session(
            user_id=current_user.user_id,
            session_key=payload.session_id,
            character_id=payload.character_id,
            title=derive_title(payload.message),
        )
    except SessionAccessDenied:
        # 归属失败必须在流开始前中断 404：返回统一 JSON 错误，
        # 而不是先发出 message_start 再吐 SSE error（后者会泄露事件流格式差异）
        raise not_found_error("会话不存在") from None
    # 任务 5.4：短期记忆写回实例（失败不影响回答流；测试可注入替身）
    short_term_memory = get_short_term_memory(request)
    return StreamingResponse(
        _event_stream(
            service=service,
            payload=payload,
            user_id=current_user.user_id,
            message_id=message_id,
            request_id=request_id,
            chat_store=chat_store,
            short_term_memory=short_term_memory,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
        },
    )


async def _event_stream(
    *,
    service: Any,
    payload: ChatStreamRequest,
    user_id: str,
    message_id: str,
    request_id: str,
    chat_store: ChatSessionStore | None = None,
    short_term_memory: ShortTermMemoryStore | None = None,
) -> AsyncIterator[str]:
    """生成 SSE 事件流；任务 5.4 起在流完整结束后落库并回写短期记忆。

    chat_store / short_term_memory 允许为 None（直调 _event_stream 的测试
    不经过端点装配，此时跳过持久化，只验证事件协议本身）。
    """
    yield encode_sse(
        "message_start",
        {"message_id": message_id, "request_id": request_id},
    )
    started_at = time.monotonic()
    completed = False
    # 任务 5.4：提问文本统一去首尾空白，检索、落库、记忆写回共用同一份
    question = payload.message.strip()
    try:
        options_kwargs = dict(
            rerank_top_n=payload.options.top_k,
            as_of_date=(
                payload.options.as_of_date.isoformat()
                if payload.options.as_of_date
                else None
            ),
            jurisdiction=payload.options.jurisdiction,
            document_types=[
                DOCUMENT_TYPE_TO_STORAGE[item]
                for item in payload.options.document_types
            ] or None,
            user_id=user_id,
            session_id=(
                payload.session_id
                if payload.options.enable_query_rewrite
                else None
            ),
            request_id=request_id,
        )
        stream_context: dict[str, Any] = {}
        if hasattr(service, "chat_stream"):
            # 真流式：LLM 增量生成，token 帧实时下发；护栏结果经 stream_context 回传。
            # chat_stream 是同步生成器，iterate_in_threadpool 保证阻塞的 next()
            # 跑在线程池里，不卡事件循环（与旧 run_in_threadpool 同一原则）
            gen = service.chat_stream(question, stream_context=stream_context, **options_kwargs)
            async for text in iterate_in_threadpool(gen):
                yield encode_sse("token", {"text": text})
            if "replace" in stream_context:
                # 方案 A：已发出的 token 撤不回，护栏判定不可信时让前端整段替换
                yield encode_sse("replace", {"text": stream_context["replace"]})
            result = stream_context["result"]
        else:
            # 兼容旧式 chat() 服务（测试替身等）：整段返回后分帧
            result = await run_in_threadpool(
                lambda: service.chat(question, **options_kwargs)
            )
            for text in split_answer_chunks(result.answer):
                yield encode_sse("token", {"text": text})
        for citation in result.sources:
            yield encode_sse("citation", citation)
        yield encode_sse(
            "message_end",
            {
                "finish_reason": "stop",
                # 现有 LLM 客户端未暴露 usage；返回 null，禁止伪造 token 数。
                "usage": None,
                # 从 message_start 到回答完成的服务端耗时，便于用户判断本次消耗时间。
                "elapsed_seconds": round(time.monotonic() - started_at, 2),
            },
        )
        # 任务 5.4：回答完整返回后落库 + 回写短期记忆（放在 completed 置位前；
        # 两个动作内部都吞异常，绝不能让持久化问题破坏已完成的回答流）
        persist_turn(
            chat_store=chat_store,
            short_term_memory=short_term_memory,
            user_id=user_id,
            session_key=payload.session_id,
            question=question,
            result=result,
            request_id=request_id,
            # assistant 消息对外 ID：流开始时已发给客户端，落库同值以满足 7.6 一致性契约
            message_id=message_id,
        )
        completed = True
    except GeneratorExit:
        raise
    except Exception as error:
        # error 仅表示检索依赖、Embedding/Reranker/LLM 或事件组装抛出的系统异常；
        # 正常业务拒答会返回 ChatResult(refused=True)，不会抛到这里，也绝不能走前端报错通道。
        logger.error(
            "问答流生成失败：%s",
            type(error).__name__,
            extra={"request_id": request_id},
        )
        yield encode_sse(
            "error",
            {
                "code": 50001,
                "message": "模型服务暂时不可用",
                "retryable": True,
                "request_id": request_id,
            },
        )
    finally:
        logger.info(
            "问答流结束：message_id=%s completed=%s",
            message_id,
            completed,
            extra={"request_id": request_id},
        )


def _get_chat_service(request: Request) -> Any:
    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        service = build_default_chat_service()
        request.app.state.chat_service = service
    return service
