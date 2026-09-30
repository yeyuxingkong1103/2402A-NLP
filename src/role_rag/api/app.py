"""FastAPI 应用：多用户 / 多角色的 RAG_try 问答服务（含 SSE 流式输出）。

启动方式：
    uvicorn role_rag.api.app:app --host 127.0.0.1 --port 8020
或使用项目根目录的 start.ps1 / start.sh。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import Depends, FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_config
from ..errors import (
    AccessDeniedError,
    AuthError,
    DependencyError,
    NotFoundError,
    RateLimitError,
    RoleRagError,
    ValidationError,
)
from ..ingest.pipeline import get_ingest_pipeline
from ..jobs import get_job_manager
from ..logging_conf import get_logger, recent_logs, set_request_id, setup_logging
from ..models.embedder import get_embedder
from ..models.llm import get_llm
from ..rag.pipeline import get_pipeline
from ..retrieval.retriever import get_retriever
from ..roles import RoleRegistry
from ..store.milvus_store import get_milvus
from ..store.redis_store import get_redis
from .auth import AuthContext, get_auth_service
from .schemas import (
    ChatRequest,
    CompareRequest,
    CreateUserRequest,
    IngestRequest,
    LoginRequest,
    MemoryClearRequest,
    RetrieveRequest,
)

logger = get_logger(__name__)
CONFIG = get_config()


def _sse(event: str, data: Any) -> str:
    """SSE 帧：event + data(JSON)。"""

    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("启动 %s v%s", CONFIG.get("app.name"), CONFIG.get("app.version"))
    try:
        created = get_auth_service(CONFIG).seed()
        if created:
            logger.info("已初始化 %d 个内置用户", created)
    except DependencyError as exc:
        logger.error("用户初始化失败（Redis 不可用）：%s", exc)
    try:
        get_milvus(CONFIG).ping()
        logger.info("Milvus 连接正常：%s", CONFIG.get("milvus.uri"))
    except DependencyError as exc:
        logger.error("Milvus 不可用：%s", exc)
    if bool(CONFIG.get("app.debug", False)):
        logger.info("调试模式已开启（角色数=%d）", len(RoleRegistry.from_config(CONFIG)))
    if bool(CONFIG.get("app.warmup", True)):
        threading.Thread(target=_warmup, name="role-rag-warmup", daemon=True).start()
    yield
    logger.info("服务已停止")


def _warmup() -> None:
    """后台预热：加载嵌入/生成模型，并预建各角色的 BM25 索引。"""

    started = time.perf_counter()
    try:
        get_embedder(CONFIG).load()
        get_llm(CONFIG).load()
        retriever = get_retriever(CONFIG)
        for role_id in RoleRegistry.from_config(CONFIG).enabled_ids():
            retriever.bm25.get(role_id)
        logger.info("预热完成，耗时 %.1fs", time.perf_counter() - started)
    except Exception as exc:  # pragma: no cover - 预热失败不影响服务
        logger.warning("预热失败（不影响启动）：%s", exc)


app = FastAPI(
    title="Role RAG_try API",
    description="多用户 / 多角色本地化 RAG_try 问答服务（Milvus + Redis + BGE-M3 + Qwen3）",
    version=str(CONFIG.get("app.version", "1.0.0")),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(item) for item in (CONFIG.get("app.cors_origins") or ["*"])],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    set_request_id(request_id)
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-elapsed-ms"] = f"{elapsed:.1f}"
    if not request.url.path.startswith("/api/logs"):
        logger.info("%s %s → %s（%.1f ms）", request.method, request.url.path,
                    response.status_code, elapsed)
    return response


# ---------------------------------------------------------------------------
# 异常处理：统一返回 {code, message, detail}
# ---------------------------------------------------------------------------
@app.exception_handler(RoleRagError)
async def handle_business_error(request: Request, exc: RoleRagError):
    logger.warning("业务异常 %s：%s", exc.code, exc.message)
    return JSONResponse(status_code=exc.http_status, content=exc.to_dict())


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):  # pragma: no cover
    logger.exception("未处理异常：%s", exc)
    return JSONResponse(
        status_code=500,
        content={"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"},
    )


# ---------------------------------------------------------------------------
# 依赖
# ---------------------------------------------------------------------------
COOKIE_NAME = "role_rag_token"


def _token_from_request(request: Request, authorization: str | None) -> str:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    cookie = request.cookies.get(COOKIE_NAME)
    if cookie:
        return cookie
    raise AuthError("缺少访问令牌，请先登录")


async def current_user(
    request: Request, authorization: str | None = Header(default=None)
) -> AuthContext:
    token = _token_from_request(request, authorization)
    return get_auth_service(CONFIG).resolve(token)


def _ensure_role_access(context: AuthContext, role_id: str) -> None:
    if not context.can_use_role(role_id):
        raise AccessDeniedError(f"当前用户无权使用角色：{role_id}", role_id=role_id)


def _rate_limit(context: AuthContext) -> None:
    allowed, remaining = get_redis(CONFIG).check_rate_limit(context.user_id)
    if not allowed:
        raise RateLimitError("提问过于频繁，请稍后再试", remaining=remaining)


# ---------------------------------------------------------------------------
# 基础信息
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, Any]:
    report: dict[str, Any] = {
        "app": CONFIG.get("app.name"),
        "version": CONFIG.get("app.version"),
        "env": CONFIG.get("app.env"),
        "components": {},
    }
    try:
        report["components"]["redis"] = {"ok": True, **get_redis(CONFIG).ping()}
    except RoleRagError as exc:
        report["components"]["redis"] = {"ok": False, "error": exc.message}
    try:
        milvus = get_milvus(CONFIG)
        report["components"]["milvus"] = {"ok": True, **milvus.ping()}
        report["components"]["milvus"]["kb_rows"] = milvus.count_chunks()
    except RoleRagError as exc:
        report["components"]["milvus"] = {"ok": False, "error": exc.message}
    try:
        registry = RoleRegistry.from_config(CONFIG)
        report["components"]["roles"] = {
            "ok": True,
            "total": len(registry),
            "enabled": registry.enabled_ids(),
        }
    except RoleRagError as exc:
        report["components"]["roles"] = {"ok": False, "error": exc.message}
    report["models"] = {
        "embedder": str(CONFIG.embedder_path()),
        "llm": str(CONFIG.llm_path()),
        "embedder_exists": CONFIG.embedder_path().is_dir(),
        "llm_exists": CONFIG.llm_path().is_dir(),
    }
    report["ok"] = all(
        item.get("ok", False) for item in report["components"].values()
    )
    return report


@app.get("/api/roles")
def list_roles(context: AuthContext = Depends(current_user)) -> dict[str, Any]:
    registry = RoleRegistry.from_config(CONFIG)
    visible = context.visible_roles(registry.enabled_ids())
    roles = [registry.get(role_id).to_public_dict() for role_id in visible]
    return {"roles": roles, "user": context.to_dict()}


# ---------------------------------------------------------------------------
# 认证
# ---------------------------------------------------------------------------
@app.post("/api/auth/login")
def login(payload: LoginRequest) -> JSONResponse:
    token, context = get_auth_service(CONFIG).login(payload.username, payload.password)
    response = JSONResponse(
        {
            "ok": True,
            "token": token,
            "user": context.to_dict(),
            "expires_in": int(CONFIG.get("security.token_ttl", 43200)),
        }
    )
    response.set_cookie(
        COOKIE_NAME, token, httponly=True, samesite="lax",
        max_age=int(CONFIG.get("security.token_ttl", 43200)),
    )
    return response


@app.post("/api/auth/logout")
def logout(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        get_auth_service(CONFIG).logout(_token_from_request(request, authorization))
    except AuthError:
        pass
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE_NAME)
    return response


@app.get("/api/auth/me")
def me(context: AuthContext = Depends(current_user)) -> dict[str, Any]:
    registry = RoleRegistry.from_config(CONFIG)
    return {
        "user": context.to_dict(),
        "roles": context.visible_roles(registry.enabled_ids()),
        "limits": {"rate_limit_per_min": int(CONFIG.get("redis.rate_limit_per_min", 60))},
    }


@app.post("/api/auth/users")
def create_user(
    payload: CreateUserRequest, context: AuthContext = Depends(current_user)
) -> dict[str, Any]:
    if not context.is_admin:
        raise AccessDeniedError("只有管理员可以创建用户")
    record = get_auth_service(CONFIG).create_user(
        payload.username, payload.password, payload.display_name, payload.roles, payload.is_admin
    )
    record.pop("password_hash", None)
    return {"ok": True, "user": record}


# ---------------------------------------------------------------------------
# 问答
# ---------------------------------------------------------------------------
@app.post("/api/chat")
def chat(payload: ChatRequest, context: AuthContext = Depends(current_user)):
    _ensure_role_access(context, payload.role_id)
    _rate_limit(context)
    pipeline = get_pipeline(CONFIG)
    options = {
        "session_id": payload.session_id,
        "top_k": payload.top_k,
        "mode": payload.mode,
        "fusion": payload.fusion,
        "use_cache": payload.use_cache,
        "temperature": payload.temperature,
        "max_new_tokens": payload.max_new_tokens,
    }

    if not payload.stream:
        result = pipeline.answer(context.user_id, payload.role_id, payload.question, **options)
        return {"ok": True, "result": result.to_dict()}

    def event_stream() -> Iterator[str]:
        try:
            for event in pipeline.stream(
                context.user_id, payload.role_id, payload.question, **options
            ):
                yield _sse(str(event.get("type", "message")), event)
        except RoleRagError as exc:
            yield _sse("error", {"code": exc.code, "message": exc.message, "detail": exc.detail})
        except Exception as exc:  # pragma: no cover
            logger.exception("SSE 流异常")
            yield _sse("error", {"code": "internal_error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# 会话与记忆
# ---------------------------------------------------------------------------
@app.get("/api/sessions")
def list_sessions(
    limit: int = Query(default=20, ge=1, le=100), context: AuthContext = Depends(current_user)
) -> dict[str, Any]:
    sessions = get_redis(CONFIG).list_sessions(context.user_id, limit=limit)
    return {"ok": True, "sessions": sessions}


@app.get("/api/sessions/{session_id}")
def session_detail(
    session_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    context: AuthContext = Depends(current_user),
) -> dict[str, Any]:
    store = get_redis(CONFIG)
    meta = store.session_meta(session_id)
    if not meta:
        raise NotFoundError("会话不存在", session_id=session_id)
    if str(meta.get("user_id")) != context.user_id and not context.is_admin:
        raise AccessDeniedError("无权访问该会话")
    return {
        "ok": True,
        "session": meta,
        "messages": store.messages(session_id, limit=limit),
        "summary": store.get_summary(session_id),
        "citations": store.session_citations(session_id),
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, context: AuthContext = Depends(current_user)) -> dict[str, Any]:
    store = get_redis(CONFIG)
    meta = store.session_meta(session_id)
    if not meta:
        raise NotFoundError("会话不存在", session_id=session_id)
    if str(meta.get("user_id")) != context.user_id and not context.is_admin:
        raise AccessDeniedError("无权删除该会话")
    store.delete_session(session_id)
    return {"ok": True, "deleted": session_id}


@app.get("/api/memory")
def memory_overview(context: AuthContext = Depends(current_user)) -> dict[str, Any]:
    registry = RoleRegistry.from_config(CONFIG)
    roles = context.visible_roles(registry.enabled_ids())
    manager = get_pipeline(CONFIG).memory
    return {
        "ok": True,
        "user_id": context.user_id,
        "profile": get_redis(CONFIG).profile(context.user_id),
        "roles": [manager.inspect(context.user_id, role_id) for role_id in roles],
        "storage": get_redis(CONFIG).describe(),
    }


@app.post("/api/memory/clear")
def clear_memory(
    payload: MemoryClearRequest, context: AuthContext = Depends(current_user)
) -> dict[str, Any]:
    if payload.role_id:
        _ensure_role_access(context, payload.role_id)
    report = get_pipeline(CONFIG).memory.clear(context.user_id, payload.role_id)
    return {"ok": True, **report}


# ---------------------------------------------------------------------------
# 检索调试
# ---------------------------------------------------------------------------
@app.post("/api/retrieval/search")
def retrieval_search(
    payload: RetrieveRequest, context: AuthContext = Depends(current_user)
) -> dict[str, Any]:
    _ensure_role_access(context, payload.role_id)
    result = get_retriever(CONFIG).search(
        payload.query, payload.role_id, top_k=payload.top_k, mode=payload.mode,
        fusion=payload.fusion, use_cache=payload.use_cache, debug=True,
    )
    return {"ok": True, "result": result.to_dict(with_text=payload.with_text)}


@app.post("/api/retrieval/compare")
def retrieval_compare(
    payload: CompareRequest, context: AuthContext = Depends(current_user)
) -> dict[str, Any]:
    _ensure_role_access(context, payload.role_id)
    report = get_retriever(CONFIG).compare(payload.query, payload.role_id, top_k=payload.top_k)
    return {"ok": True, "report": report}


# ---------------------------------------------------------------------------
# 知识库
# ---------------------------------------------------------------------------
@app.get("/api/kb/status")
def kb_status(context: AuthContext = Depends(current_user)) -> dict[str, Any]:
    pipeline = get_ingest_pipeline(CONFIG)
    report = pipeline.report()
    report["bm25_indexes"] = get_retriever(CONFIG).bm25.stats()
    return {"ok": True, "report": report}


@app.post("/api/kb/ingest")
def kb_ingest(payload: IngestRequest, context: AuthContext = Depends(current_user)) -> dict[str, Any]:
    if not context.is_admin:
        raise AccessDeniedError("只有管理员可以触发全量入库")
    roles = payload.roles
    if roles:
        for role_id in roles:
            _ensure_role_access(context, role_id)
    manager = get_job_manager()
    pipeline = get_ingest_pipeline(CONFIG)

    def _task(progress) -> dict[str, Any]:
        def _on_progress(stage: str, info: dict[str, Any]) -> None:
            done = info.get("done")
            total = info.get("total")
            ratio = (done / total) if done and total else None
            progress(f"{stage}: {info}", ratio)

        return pipeline.ingest_all(roles=roles, recreate=payload.recreate, progress=_on_progress)

    job = manager.submit("kb_ingest", _task)
    return {"ok": True, "job": job.to_dict()}


@app.post("/api/kb/upload")
async def kb_upload(
    role_id: str = Form(...),
    file: UploadFile = File(...),
    recreate: bool = Form(default=False),
    context: AuthContext = Depends(current_user),
) -> dict[str, Any]:
    if not context.is_admin:
        raise AccessDeniedError("只有管理员可以上传知识文件")
    _ensure_role_access(context, role_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".md", ".markdown", ".txt", ".pdf"}:
        raise ValidationError(f"不支持的文件类型：{suffix}")

    target_dir = CONFIG.kb_dir / role_id / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(file.filename or f"upload{suffix}").name
    content = await file.read()
    if not content:
        raise ValidationError("上传文件为空")
    target.write_bytes(content)
    logger.info("已保存上传文件：%s（%d 字节）", target, len(content))

    pipeline = get_ingest_pipeline(CONFIG)
    manager = get_job_manager()

    def _task(progress) -> dict[str, Any]:
        progress(f"入库 {target.name}", 0.2)
        result = pipeline.ingest_files([target], role=role_id, scope=role_id)
        progress("完成", 1.0)
        return result

    job = manager.submit("kb_upload", _task)
    return {"ok": True, "saved": str(target), "job": job.to_dict()}


@app.post("/api/kb/reset")
def kb_reset(
    roles: list[str] | None = None, context: AuthContext = Depends(current_user)
) -> dict[str, Any]:
    if not context.is_admin:
        raise AccessDeniedError("只有管理员可以重建知识库")
    pipeline = get_ingest_pipeline(CONFIG)
    if not roles:
        pipeline.reset()
        return {"ok": True, "reset": "all"}
    removed = {role_id: pipeline.delete_scope(role_id) for role_id in roles}
    return {"ok": True, "removed": removed}


@app.get("/api/jobs")
def list_jobs(limit: int = Query(default=20, ge=1, le=50),
              context: AuthContext = Depends(current_user)) -> dict[str, Any]:
    jobs = [job.to_dict() for job in get_job_manager().list(limit)]
    return {"ok": True, "jobs": jobs}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str, context: AuthContext = Depends(current_user)) -> dict[str, Any]:
    job = get_job_manager().get(job_id)
    if job is None:
        raise NotFoundError("作业不存在", job_id=job_id)
    return {"ok": True, "job": job.to_dict()}


# ---------------------------------------------------------------------------
# 运行状态
# ---------------------------------------------------------------------------
@app.get("/api/stats")
def stats(context: AuthContext = Depends(current_user)) -> dict[str, Any]:
    report: dict[str, Any] = {"ok": True}
    try:
        report["redis"] = get_redis(CONFIG).stats()
    except RoleRagError as exc:
        report["redis"] = {"error": exc.message}
    try:
        milvus = get_milvus(CONFIG)
        report["milvus"] = milvus.stats()
        report["chunks_by_scope"] = milvus.scope_distribution()
    except RoleRagError as exc:
        report["milvus"] = {"error": exc.message}
    report["bm25"] = get_retriever(CONFIG).bm25.stats()
    report["models"] = {"embedder": get_embedder(CONFIG).stats(), "llm": get_llm(CONFIG).stats()}
    report["jobs"] = [job.to_dict() for job in get_job_manager().list(5)]
    return report


@app.get("/api/logs")
def logs(
    limit: int = Query(default=100, ge=1, le=500),
    level: str | None = Query(default=None),
    context: AuthContext = Depends(current_user),
) -> dict[str, Any]:
    return {"ok": True, "logs": recent_logs(limit=limit, level=level)}


# ---------------------------------------------------------------------------
# 前端静态资源（零依赖，由 FastAPI 同源托管）
# ---------------------------------------------------------------------------
WEB_DIR = CONFIG.web_dir
if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> Any:
        return FileResponse(str(WEB_DIR / "index.html"))

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Any:
        icon = WEB_DIR / "favicon.ico"
        if icon.is_file():
            return FileResponse(str(icon))
        # 204 响应不能带 body：JSONResponse 会把 None 渲染成 b"null"，
        # 而 h11 对 204 强制按 Content-Length: 0 分帧，发送这 4 字节会抛
        # "Too much data for declared Content-Length" 并中断连接。
        return Response(status_code=204)
else:  # pragma: no cover
    logger.warning("前端目录不存在：%s", WEB_DIR)


def main() -> None:  # pragma: no cover - 便于 python -m role_rag.api.app 启动
    import uvicorn

    uvicorn.run(
        "role_rag.api.app:app",
        host=str(CONFIG.get("app.host", "127.0.0.1")),
        port=int(CONFIG.get("app.port", 8020)),
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
