"""多轮对话的 HTTP 路由（specs/010 的 I-12）。

本模块只做 **HTTP ↔ 模型** 的协议转换，MUST NOT 含业务判定 ——
与 `routes.py` 的既有约束一致。脱敏、窗口裁剪、拒答决策全部在
`backend/chat/` 的服务层。

---

## 会话判定发生在**流之前**（contracts/chat-sse.md §4）

`contracts/chat-sse.md` §4 给了两条合规路径：
① `status` 首帧先发、会话判定在流内；② 流前先做一次存在性判定。

**本实现选 ②**。理由是客户端能拿到一个标准的 404 错误体，而不是一个
已经开始的流 —— 后者要求前端认识第五种事件（错误帧），而 chat 目前
没有前端，把复杂度放进服务端不划算。

代价是首帧延迟一次 Redis 往返（本地 < 5 ms）。Redis 不可达时同样在流前失败，
返回 503 而不是"流中断" —— 那反而是更好的处置。

⚠️ 该选择 MUST 回写 `docs/05`（契约文件要求写明实际采用了哪一条）。
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.chat.dialogue import stream_chat_turn
from backend.chat.service import ChatService

from . import (
    CHAT_HISTORY_PATH,
    CHAT_MESSAGE_PATH,
    CHAT_SESSION_ITEM_PATH,
    CHAT_SESSION_PATH,
)
from .capture import capture_question
from .chat_schemas import (
    ChatMessage,
    CreateSessionRequest,
    HistoryResponse,
    RenameSessionRequest,
    SendMessageRequest,
    SessionResponse,
)
from .sse_headers import SSE_HEADERS

logger = logging.getLogger(__name__)

router = APIRouter()

__all__ = ["router", "get_chat_service"]


def get_chat_service(request: Request) -> ChatService:
    """从应用状态取会话服务。

    ⚠️ **未配置时抛错而不是惰性创建。** 惰性创建意味着第一次用户请求
    承担"连 Redis"的成本，且把"Redis 不可达"从启动期推迟到请求期 ——
    而启动期的失败是可见的、可修的，请求期的失败只是用户看到一句拒答。
    与 `backend/retrieve/bundle.get_index()` 的处置同一取向。
    """

    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        raise RuntimeError(
            "会话服务尚未初始化。这是**编程错误**而非数据问题："
            "服务端 MUST 在启动期构造 ChatService 并挂到 app.state.chat_service"
            "（见 backend/serve.py）。"
        )
    return service


@router.post(CHAT_SESSION_PATH)
async def create_session(request: Request, payload: CreateSessionRequest) -> SessionResponse:
    """建会话。`user_id` 可选，缺省自动生成（Q5）。"""

    service = get_chat_service(request)
    session_id, meta = await service.create_session(payload.user_id)
    config = request.app.state.config

    return SessionResponse(
        session_id=session_id,
        user_id=meta.user_id,
        created_at=meta.created_at,
        last_active_at=meta.last_active_at,
        expires_in=config.chat_session_ttl,
    )


@router.patch(CHAT_SESSION_ITEM_PATH)
async def rename_session(
    request: Request, session_id: str, payload: RenameSessionRequest
) -> SessionResponse:
    """改发起者标识。**不刷新 TTL、不更新 `last_active_at`**（Q5 裁决 A）。"""

    service = get_chat_service(request)
    meta = await service.rename_session(session_id, payload.user_id)
    config = request.app.state.config

    return SessionResponse(
        session_id=session_id,
        user_id=meta.user_id,
        created_at=meta.created_at,
        last_active_at=meta.last_active_at,
        expires_in=config.chat_session_ttl,
    )


@router.post(CHAT_MESSAGE_PATH)
async def send_message(request: Request, payload: SendMessageRequest) -> StreamingResponse:
    """发送消息并取回回答（SSE）。

    ⚠️ **校验在返回 `StreamingResponse` 之前完成**（Pydantic 已做）。
    若先返回流再报错，客户端会拿到 status 帧却等不到 done，
    无法区分"服务端拒绝了"与"网络断了" —— 而这两件事该做的处置完全不同。

    ⚠️ **会话存在性判定同样在流之前**（见模块文档）。这是本实现相对
    `contracts/chat-sse.md` §4 的路径 ① 的偏离，已记录在案。
    """

    service = get_chat_service(request)
    config = request.app.state.config

    # 会话不存在 → ChatError(session_missing=True) → 404（errors.py 的处理器）
    await service.ensure_exists(payload.session_id)

    answer_id = str(uuid.uuid4())

    # 与 `/ask` 同一条路径：留存问题 + 顺带算出查询向量（S8）。
    #
    # ⚠️ MUST NOT 在本模块自己编码 —— 那是第二个编码入口（FR-003）。
    # 与 `/ask` 共用 `capture_question` 同时意味着：chat 的问题也被
    # 留存到 data/questions，与单轮提问进同一份记录。
    query_vector = capture_question(payload.content, answer_id)

    logger.info(
        "收到会话消息 answer_id=%s session_id=%s content_len=%d",
        answer_id,
        payload.session_id,
        len(payload.content),
    )

    return StreamingResponse(
        stream_chat_turn(
            session_id=payload.session_id,
            question=payload.content,
            answer_id=answer_id,
            config=config,
            service=service,
            query_vector=query_vector,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get(CHAT_HISTORY_PATH)
async def get_history(
    request: Request, session_id: str, limit: int | None = None
) -> HistoryResponse:
    """读历史。`limit` 取最近 N 条，返回仍为正序；`total` 与 `limit` 无关。"""

    if limit is not None and limit <= 0:
        # ⚠️ 用 `HTTPException(422)` 而不是 Pydantic 校验：`limit` 是查询参数，
        #    而它在**校验通过**之后才需要做范围判断（`None` 是合法的）。
        #
        #    422 会被 `errors.py` 的 `_http_exception` 处理器归到
        #    `CHAT_INVALID_REQUEST`（chat 路径前缀），错误体形状与其它错误一致。
        raise HTTPException(status_code=422, detail="limit 必须 > 0")

    service = get_chat_service(request)
    # ⚠️ 读全量再切片，而不是让 `service.get_history` 带 limit：
    #    响应里的 `total` 是**会话的消息总条数**，与 limit 无关。
    #    带 limit 读会让 total 变成 limit，前端据此判断"还有更多"就永远为假。
    #    存储上限 50 条，读全量在量级上完全可行。
    messages = await service.get_history(session_id)
    selected = messages[-limit:] if limit is not None else messages

    return HistoryResponse(
        session_id=session_id,
        total=len(messages),
        messages=[
            ChatMessage(
                role=m.role,
                content=m.content,
                timestamp=m.timestamp,
                message_id=m.message_id,
            )
            for m in selected
        ],
    )


@router.delete(CHAT_SESSION_ITEM_PATH)
async def delete_session(request: Request, session_id: str) -> dict:
    """删除会话。**幂等** —— 删不存在的会话同样返回 200（contracts §5）。"""

    service = get_chat_service(request)
    await service.delete_session(session_id)
    return {"ok": True}
