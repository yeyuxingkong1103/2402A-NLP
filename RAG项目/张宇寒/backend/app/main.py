from contextlib import asynccontextmanager
import logging
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .auth import router as auth_router, user_router
from .config import get_settings
from .memory.api import router as memory_router
from .rag.api import router as rag_router
from .system import RequestLoggingMiddleware, create_services, router as system_router, setup_logging, shutdown_logging
from .workspace import router as workspace_router


settings = get_settings()
setup_logging(settings)
logger = logging.getLogger(__name__)
frontend_dir = Path(__file__).resolve().parents[2] / "frontend"


def parse_cors_origins(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = re.split(r"[\s,;]+", str(value))
    origins: list[str] = []
    for item in values:
        origin = str(item or "").strip().rstrip("/")
        if origin and origin not in origins:
            origins.append(origin)
    return origins


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = create_services(settings)
    try:
        yield
    finally:
        shutdown_logging()


expose_docs = bool(getattr(settings, "expose_docs", True))
cors_origins = parse_cors_origins(getattr(settings, "cors_allowed_origins", None))
cors_origin_regex = str(getattr(settings, "cors_allowed_origin_regex", "") or "").strip() or None
allow_credentials = bool(getattr(settings, "cors_allow_credentials", True))
if cors_origins:
    allow_origins = cors_origins
    logger.info(
        "CORS 已配置为显式来源白名单",
        extra={"event": "cors_allowlist_configured", "fields": {"count": len(cors_origins)}},
    )
elif cors_origin_regex:
    allow_origins = []
    logger.info(
        "CORS 已配置为来源正则",
        extra={"event": "cors_origin_regex_configured", "fields": {"pattern": cors_origin_regex}},
    )
else:
    logger.warning(
        "CORS 未配置来源白名单，回退为开放来源；生产环境请设置 CORS_ALLOWED_ORIGINS 或 CORS_ALLOWED_ORIGIN_REGEX",
        extra={"event": "cors_open_fallback"},
    )
    allow_origins = ["*"]
app = FastAPI(
    title="LAW-RAG",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if expose_docs else None,
    redoc_url="/redoc" if expose_docs else None,
    openapi_url="/openapi.json" if expose_docs else None,
)
app.add_middleware(RequestLoggingMiddleware, settings=settings)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=cors_origin_regex,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(system_router)
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(workspace_router)
app.include_router(memory_router)
app.include_router(rag_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": str(exc.detail), "code": f"HTTP_{exc.status_code}", "data": None},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    first_error = exc.errors()[0] if exc.errors() else {}
    message = str(first_error.get("msg", "请求参数不正确")).replace("Value error, ", "")
    return JSONResponse(
        status_code=422,
        content={"success": False, "message": message, "code": "VALIDATION_ERROR", "data": None},
    )

if frontend_dir.exists():
    app.mount("/src", StaticFiles(directory=frontend_dir / "src"), name="src")


@app.get("/")
def index():
    return FileResponse(frontend_dir / "index.html")
