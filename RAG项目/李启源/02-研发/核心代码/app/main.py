"""FastAPI 应用入口。

阅读顺序：
1. 加载 `.env` 与应用配置；
2. 注册安全、监控和请求大小中间件；
3. 挂载问答、知识库、会话路由；
4. 暴露存活检查、就绪检查和静态页面。
"""

from __future__ import annotations

import logging
import os
import socket
from urllib.parse import urlparse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.v1 import routes
from app.api.v1 import chat_routes
from app.api.v1 import kb_routes
from app.api.v1 import session_routes
from app.api.v1.schemas import HealthResponse
from app.core.auth import security_middleware, validate_production_security
from app.core.config import load_config_from_env
from app.core.metrics import metrics_middleware, metrics_payload
from app.core.request_limits import RequestSizeLimitMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# 配置必须在创建应用前读取；生产环境缺少 API Key 时在启动阶段直接失败，
# 避免服务先监听端口、再以不安全状态对外提供接口。
app_config, _, _, _, _, _ = load_config_from_env()
validate_production_security()

app = FastAPI(
    title="RAG API",
    description="Retrieval Augmented Generation API with document ingestion and query",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 中间件按“安全校验 → 指标记录 → 请求体限制”的职责注册。
# 具体执行顺序由 Starlette 的中间件栈决定，不要在业务路由里重复实现这些横切逻辑。
app.middleware("http")(security_middleware)
app.middleware("http")(metrics_middleware)
app.add_middleware(RequestSizeLimitMiddleware)

if app_config.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 所有业务接口统一使用 API_PREFIX（默认 /api/v1），系统级健康检查保留在根路径。
app.include_router(routes.router, prefix=app_config.api_prefix, tags=["RAG"])
app.include_router(chat_routes.router, prefix=app_config.api_prefix, tags=["Chat"])
app.include_router(kb_routes.router, prefix=app_config.api_prefix, tags=["Knowledge Bases"])
app.include_router(session_routes.router, prefix=app_config.api_prefix, tags=["Sessions"])

# 挂载静态文件
static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    return Response(content=metrics_payload(), media_type="text/plain; version=0.0.4")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Liveness probe: only reports that the API process is running."""
    return HealthResponse(status="healthy")


@app.get("/ready", tags=["Health"])
async def readiness_check() -> dict:
    """Report whether configured runtime dependencies are reachable."""
    from app.api.v1 import deps

    # /health 只证明 API 进程活着；/ready 才判断请求所需的外部依赖。
    # 这样编排平台不会因为某个依赖短暂波动而反复重启仍然健康的 API 进程。
    memory = deps.memory_status()
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "mock").lower()
    vector_provider = os.getenv("VECTOR_STORE_PROVIDER", "milvus").lower()
    milvus_reachable = True
    if embedding_provider != "mock" and vector_provider == "milvus":
        milvus_reachable = _milvus_reachable(os.getenv("MILVUS_URI", "http://localhost:19530"))

    llm_provider = os.getenv("LLM_PROVIDER", "mock").lower()
    llm_ready = llm_provider == "mock" or bool(
        os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    components = {
        **memory,
        "milvus": milvus_reachable,
        "milvus_reachable": milvus_reachable,
        "long_term_memory": bool(memory.get("long_term_memory", False)),
        "llm": llm_ready,
    }
    # 端口可达不等于三个长期记忆集合已初始化并加载。
    ready = milvus_reachable and llm_ready
    return {"status": "ready" if ready else "degraded", "components": components}


def _milvus_reachable(uri: str) -> bool:
    """Check the Milvus endpoint without initializing a client or collection."""
    parsed = urlparse(uri if "://" in uri else f"http://{uri}")
    host = parsed.hostname
    port = parsed.port or 19530
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


@app.get("/", tags=["Root"], include_in_schema=False)
async def root() -> RedirectResponse:
    """Root endpoint - redirect to web UI."""
    return RedirectResponse(url="/static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=app_config.api_host,
        port=app_config.api_port,
        reload=app_config.environment == "development",
        log_level=app_config.log_level.lower(),
    )
