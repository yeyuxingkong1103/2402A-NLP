from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.v1.admin import router as admin_router
from backend.app.api.v1.auth import router as auth_router
from backend.app.api.v1.chat import router as chat_router
from backend.app.api.v1.conversations import router as conversations_router
from backend.app.api.v1.feedback import router as feedback_router
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.knowledge_bases import router as knowledge_base_router
from backend.app.api.v1.users import router as users_router
from backend.app.core.config import settings, validate_production_settings
from backend.app.core.metrics import monotonic_time, registry
from backend.app.core.storage import configure_auth_store


def _configure_logging() -> None:
    # 仅设置应用命名空间级别，保留 Uvicorn 的访问日志和部署方根日志配置。
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    application_logger = logging.getLogger("backend")
    application_logger.setLevel(level)
    if not application_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        application_logger.addHandler(handler)


_configure_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 生产启动先校验配置，避免开发后端或缺失密钥进入对外服务。
    validate_production_settings(settings)
    # 启动时按配置选择认证存储；默认内存，生产可切换 SQL。
    configure_auth_store()
    yield


app = FastAPI(title="Legal RAG Assistant", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def collect_http_metrics(request, call_next):
    started_at = monotonic_time()
    response = await call_next(request)
    if settings.METRICS_ENABLED:
        route = request.scope.get("route")
        path = getattr(route, "path", None) or request.url.path
        registry.observe_request(
            method=request.method,
            path=path,
            status_code=response.status_code,
            duration_seconds=monotonic_time() - started_at,
        )
    return response


if settings.METRICS_ENABLED:

    @app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    def metrics() -> str:
        return registry.render_prometheus()


cors_origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]
if cors_origins:
    # 分端口联调时允许前端开发端口访问 API；生产环境应显式收窄来源。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(knowledge_base_router)
app.include_router(chat_router)
app.include_router(feedback_router)
app.include_router(conversations_router)
app.include_router(users_router)

frontend_dir = Path(settings.FRONTEND_DIR)
if settings.SERVE_FRONTEND and frontend_dir.exists():
    # 本地真实浏览器联调使用同源静态托管，避免额外代理和 CORS 配置。
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")


@app.get("/")
def frontend_home() -> RedirectResponse:
    # 单端口本地联调默认进入前端首页，API 仍保留 /api/v1 前缀。
    return RedirectResponse(url="/frontend/pages/chat.html")


@app.get("/health")
def health_check() -> dict[str, str]:
    # 保留旧健康接口，避免已有调用方在 Task 1 中被破坏。
    return {"status": "ok"}
