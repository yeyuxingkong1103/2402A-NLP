"""FastAPI 应用入口。"""

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from app.auth.router import router as auth_router
from app.api.chat import router as chat_router
from app.api.legal_search import router as legal_search_router
from app.api.memory import router as memory_router
from app.api.review import router as review_router
from app.api.session_routes import router as session_router
from app.api.users_me import router as users_me_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.errors import BusinessException


class ResponseModel(BaseModel):
    """通用响应模型。"""

    code: int
    message: str
    data: dict[str, str] | None
    request_id: str


class RequestIdMiddleware(BaseHTTPMiddleware):
    """为每个请求添加请求追踪 ID。"""

    async def dispatch(self, request: Request, call_next):
        # 外部请求 ID 原样沿用；未提供时生成公开格式，便于响应和日志关联。
        request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        # 将请求 ID 存储到请求状态中，供后续处理器使用
        request.state.request_id = request_id
        # 调用下一个中间件或路由处理器
        response = await call_next(request)
        # 在响应头中添加请求 ID
        response.headers["X-Request-ID"] = request_id
        # 记录请求日志
        logging.getLogger("app.request").info(
            "%s %s %s",
            request.method,
            request.url.path,
            response.status_code,
            extra={"request_id": request_id},
        )
        return response


# 配置日志系统
configure_logging(settings.log_level)

# 创建 FastAPI 应用实例
app = FastAPI(title=settings.app_name)

# 添加请求 ID 中间件
app.add_middleware(RequestIdMiddleware)

# 注册认证路由
app.include_router(auth_router)
# 注册法律检索路由
app.include_router(legal_search_router)
# 注册问答流式路由
app.include_router(chat_router)
# 注册会话管理路由（创建 / 列表 / 历史，任务 5.4）
app.include_router(session_router)
# 注册管理员审核与发布路由（阶段 6，契约 6.3 / 6.4）
app.include_router(review_router)
# 注册长期记忆路由（批次 14，接口 8.1 / 8.2 / 8.3）
app.include_router(memory_router)
# 注册当前用户信息路由（批次 16-A，接口 3.8）
app.include_router(users_me_router)


# 应用启动事件
@app.on_event("startup")
def startup_event():
    """应用启动时初始化 Redis 客户端。"""
    from app.db.redis_client import create_redis_client

    create_redis_client(settings.redis_url)
    logging.getLogger("app.main").info("Redis 客户端初始化完成")


# 全局请求参数异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """参数错误统一返回公开错误，不回显内部模型与输入细节。"""
    return JSONResponse(
        status_code=422,
        content={
            "code": 40000,
            "message": "请求参数错误",
            "data": None,
            "request_id": getattr(request.state, "request_id", ""),
        },
    )


# 全局业务异常处理器
@app.exception_handler(BusinessException)
async def business_exception_handler(request: Request, exc: BusinessException):
    """处理业务异常，返回统一格式的错误响应。"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code,
            "message": exc.message,
            "data": None,
            "request_id": getattr(request.state, "request_id", ""),
        },
    )


# 全局未知异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """处理未捕获异常；日志仅记录类型，避免连接信息和栈泄露。"""
    request_id = getattr(request.state, "request_id", "")
    logging.getLogger("app.error").error(
        "未捕获的异常：%s",
        type(exc).__name__,
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=500,
        content={
            "code": 50000,
            "message": "系统内部错误",
            "data": None,
            "request_id": request_id,
        },
    )


# 存活健康检查接口
@app.get("/health/live", response_model=ResponseModel)
def live_health(request: Request) -> ResponseModel:
    """存活性探针，表示服务进程正在运行。"""
    return ResponseModel(
        code=0,
        message="success",
        data={"status": "alive"},
        request_id=getattr(request.state, "request_id", ""),
    )


# 就绪健康检查接口
@app.get("/health/ready", response_model=ResponseModel)
def ready_health(request: Request) -> ResponseModel:
    """就绪性探针，表示服务可以处理请求。"""
    return ResponseModel(
        code=0,
        message="success",
        data={"status": "ready"},
        request_id=getattr(request.state, "request_id", ""),
    )

