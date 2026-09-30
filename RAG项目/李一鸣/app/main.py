import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.database import SessionLocal, init_db
from app.core.logging import configure_logging, new_request_id
from app.rag.service import RAGService
from app.storage.repositories import seed_default_roles

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    # FastAPI 启动阶段统一完成目录、数据库、默认角色和 RAG 服务的初始化。
    # 这样路由处理请求时可以直接从 application.state 取得已经准备好的服务。
    settings = get_settings()
    configure_logging(settings)
    for directory in (
        settings.data_dir,
        settings.upload_dir,
        settings.index_dir,
        settings.log_dir,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)
    await init_db()
    async with SessionLocal() as session:
        # 仅在角色表为空时写入演示角色，避免每次重启都重复创建数据。
        await seed_default_roles(session)
    application.state.settings = settings
    application.state.rag = RAGService(settings)
    logger.info("application started: env=%s", settings.app_env)
    yield
    logger.info("application stopped")


app = FastAPI(
    title="RAG Roleplay System",
    version="1.0.0",
    description="多用户、多角色、支持知识库检索和短期记忆的 RAG 角色扮演系统",
    lifespan=lifespan,
)
# 前端开发页面通常运行在不同端口，因此开发环境先放开跨域；生产环境应改为明确的域名白名单。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    # 为每个请求分配追踪 ID，并记录总耗时，便于从日志定位慢请求或异常请求。
    request_id = new_request_id()
    started = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        logger.info(
            "http request completed",
            extra={
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )


app.include_router(router)


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    # 返回服务入口提示，真正的业务接口位于 /api/v1 下。
    return {"service": "rag-roleplay-system", "docs": "/docs", "health": "/api/v1/health"}
