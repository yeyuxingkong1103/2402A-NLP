import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..auth import current_user
from .answer import render_solution_pdf
from .understand import AskRequest, SolutionGenerateRequest, SolutionPdfRequest


router = APIRouter(prefix="/api/v1/legal", tags=["rag"])
logger = logging.getLogger("law_rag.rag_api")


class SimpleRateLimiter:
    """基于滑动窗口的内存速率限制器。

    按 key（user_id 或 IP）统计请求数，超过上限返回 429。
    匿名用户限制更严格，登录用户宽松一些。
    """

    def __init__(self, anonymous_limit: int = 5, user_limit: int = 20, window_seconds: int = 60):
        self.anonymous_limit = anonymous_limit
        self.user_limit = user_limit
        self.window = window_seconds
        self._buckets: dict[str, deque] = defaultdict(deque)

    def check(self, key: str, is_anonymous: bool) -> None:
        limit = self.anonymous_limit if is_anonymous else self.user_limit
        now = time.monotonic()
        bucket = self._buckets[key]
        while bucket and bucket[0] <= now - self.window:
            bucket.popleft()
        if len(bucket) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"请求过于频繁，请 {self.window} 秒后重试",
            )
        bucket.append(now)


_rate_limiter = SimpleRateLimiter()


def _client_key(request: Request, user: dict | None) -> str:
    if user:
        return f"user:{user['user_id']}"
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    return f"anon:{ip}"


def optional_user(request: Request) -> dict | None:
    try:
        return current_user(request)
    except HTTPException:
        return None


def _solution_cache_key(user: dict | None, session_id: str | None, payload: SolutionGenerateRequest) -> str:
    user_part = user["user_id"] if user else "anonymous"
    session_part = session_id or "no-session"
    answer_text = payload.answer if isinstance(payload.answer, str) else json.dumps(payload.answer, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    source_fingerprint = json.dumps(payload.sources, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    raw = "|".join([user_part, session_part, payload.question.strip(), answer_text, source_fingerprint])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    # 方案版式升级后切换缓存命名空间，避免继续返回旧版"法律解决方案"结构。
    return f"rag:solution:v2:{digest}"


@router.post("/ask")
def ask(payload: AskRequest, request: Request):
    user = optional_user(request)
    _rate_limiter.check(_client_key(request, user), user is None)
    session_id = payload.session_id if user else None
    return {"success": True, "data": request.app.state.services["rag"].run(payload.text(), user, session_id, payload.include_web, payload.thinking_enabled)}


@router.post("/ask/stream")
def ask_stream(payload: AskRequest, request: Request):
    user = optional_user(request)
    _rate_limiter.check(_client_key(request, user), user is None)
    session_id = payload.session_id if user else None

    def events():
        try:
            for event, data in request.app.state.services["rag"].stream(payload.text(), user, session_id, payload.include_web, payload.thinking_enabled):
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/solution")
def solution(payload: SolutionGenerateRequest, request: Request):
    user = optional_user(request)
    _rate_limiter.check(_client_key(request, user), user is None)
    services = request.app.state.services
    rag = services["rag"]
    redis_store = services["redis"]
    session_id = payload.session_id if user else None

    # ========== 新增：读取 force_refresh 参数 ==========
    force_refresh = getattr(payload, "force_refresh", False)
    # ========== force_refresh 结束 ==========

    cache_key = _solution_cache_key(user, session_id, payload)

    # ========== 修改：如果 force_refresh 为 True，跳过缓存直接生成 ==========
    if not force_refresh:
        cached_solution = redis_store.get_json_value(cache_key)
        if cached_solution:
            return {"success": True, "data": cached_solution}
    # ========== 修改结束 ==========

    history = []
    if user and session_id:
        try:
            memory = services.get("memory")
            if memory:
                context = memory.load_context(user["user_id"], session_id, query=payload.question)
                history = getattr(context, "short_term", []) or []
            else:
                history = redis_store.get_history(user["user_id"], session_id)
        except Exception:
            logger.warning(
                "法律解决方案读取会话历史失败，继续基于本轮回答生成",
                extra={"event": "solution_history_load_failed", "fields": {"session_id": session_id or ""}},
                exc_info=True,
            )

    generated = rag.generator.generate_solution(
        payload.question,
        payload.answer,
        payload.sources,
        history=history,
        thinking_enabled=False
    )
    generated["cache_key"] = cache_key

    # ========== 新增：force_refresh 时也更新缓存（覆盖旧缓存） ==========
    redis_store.set_json_value(
        cache_key,
        generated,
        getattr(services["settings"], "history_ttl", 604800)
    )
    # ========== 更新缓存结束 ==========

    return {"success": True, "data": generated}


@router.post("/solution/pdf")
def solution_pdf(payload: SolutionPdfRequest, request: Request):
    user = current_user(request)
    _rate_limiter.check(_client_key(request, user), False)
    markdown = (payload.markdown or "").strip()
    if not markdown or len(markdown) < 50:
        raise HTTPException(status_code=400, detail="解决方案内容为空或过短，无法生成 PDF")
    pdf = render_solution_pdf(markdown)
    filename = quote("法律解决方案.pdf")
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=legal-solution.pdf; filename*=UTF-8''{filename}",
            "Cache-Control": "no-store",
        },
    )


@router.get("/sessions")
def list_sessions(request: Request, limit: int = 20):
    """Return the authenticated user's recent conversations for frontend restore."""
    user = current_user(request)
    memory = request.app.state.services.get("memory")
    if not memory:
        return {"success": True, "data": []}
    safe_limit = min(max(int(limit or 20), 1), 20)
    summaries = memory.history.latest_messages(user["user_id"], limit=safe_limit)
    return {"success": True, "data": summaries}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, request: Request):
    user = current_user(request)
    user_id = user["user_id"]
    memory = request.app.state.services.get("memory")

    try:
        if memory:
            memory.delete_session(user_id, session_id)
    except Exception as exc:
        logger.warning(
            "会话前端删除标记失败，后端历史和材料仍保留",
            extra={"event": "rag_session_frontend_delete_noop_failed", "fields": {"session_id": session_id, "error": str(exc)}},
            exc_info=True,
        )
        return {"success": False, "data": {"deleted": False, "session_id": session_id, "errors": [f"会话删除标记失败: {exc}"]}}

    return {"success": True, "data": {"deleted": True, "session_id": session_id, "retained": ["history", "short_memory", "long_memory", "workspace_vectors"]}}
