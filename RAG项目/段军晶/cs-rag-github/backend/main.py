# -*- coding: utf-8 -*-
"""
FastAPI 服务入口

职责：
    1. 初始化日志系统
    2. 挂载各 API 路由
    3. 托管前端静态页面（面向普通终端用户的简洁界面）
    4. 提供健康检查接口，供部署验证与 JMeter 压测探活

启动方式：
    python -m backend.main
    或
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

前端约定：
    前端只有四个功能：加载知识库、提问、看答案、看来源文档与页码。
    **不提供任何检索链路可视化页面，也不对外暴露 Trace 查询接口。**
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.db import milvus_client, mysql, redis_client
from backend.logging_config import get_logger, new_request_id, setup_logging

# 日志必须在其它模块导入前初始化，保证启动阶段的日志也能落盘
setup_logging()
logger = get_logger(__name__)


# ===========================================================================
# 应用生命周期
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动与关闭时的处理"""
    settings.ensure_directories()

    logger.info("=" * 68)
    logger.info("%s 正在启动", settings.app_title)
    logger.info("监听地址：http://%s:%s", settings.app_host, settings.app_port)
    logger.info("知识库源目录：%s", settings.source_docs_path)
    logger.info("Milvus 集合：%s", settings.milvus_collection)
    logger.info("=" * 68)

    # 依赖服务连通性自检（失败不阻断启动，仅告警，便于先起服务再排查）
    checks = {
        "Milvus": milvus_client.health_check,
        "MySQL": mysql.health_check,
        "Redis": redis_client.health_check,
    }
    for name, check in checks.items():
        try:
            ok = check()
            if ok:
                logger.info("依赖服务连通：%s", name)
            else:
                logger.warning("依赖服务不可用：%s（相关功能将受影响）", name)
        except Exception as exc:
            logger.warning("依赖服务自检异常：%s | %s", name, exc)

    # 表结构初始化（幂等）：新增的表与字段随服务启动自动就绪，无需手工执行 SQL
    try:
        mysql.init_schema()
    except Exception as exc:
        logger.warning("表结构初始化失败（用户/历史/收藏相关功能可能不可用）：%s", exc)

    yield

    logger.info("%s 已停止", settings.app_title)


# ===========================================================================
# 应用实例
# ===========================================================================

app = FastAPI(
    title=settings.app_title,
    description=(
        "计算机专业知识库 RAG 问答服务。\n\n"
        "- 答案严格依据知识库原文生成，并附带**来源文档名称与页码**\n"
        "- 检索中间过程仅记录于服务端日志，不对外提供 Trace 查询接口\n"
        "- 面向普通终端用户，仅提供：知识库加载、提问、答案展示、来源页码展示"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# 允许本机与局域网前端页面跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# 全局异常处理
# ===========================================================================

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    兜底异常处理：完整堆栈写入服务端日志，对外只返回可理解的中文提示。
    """
    logger.exception("未捕获异常 | 路径=%s | %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务内部错误，请稍后重试（详细信息见服务端日志）"},
    )


def _error_field(err: Dict[str, Any]) -> str:
    """取出校验错误定位中的字段名（loc 形如 ("body", "question")）"""
    loc = err.get("loc") or ()
    return str(loc[-1]) if loc else ""


def _validation_error_detail(exc: RequestValidationError) -> str:
    """
    把 FastAPI 的结构化校验错误转换成一个中文说明。

    《接口文档》1.3 承诺「所有错误统一返回 {"detail": "错误说明（中文）"}」，
    因此这里**绝不**把 pydantic 的 errors 数组回给调用方——那会把内部实现
    结构泄漏给 curl / Postman 等直接调用方。结构化错误只写服务端日志。
    """
    errors = exc.errors()

    # 1) 自定义校验器抛出的 ValueError：消息本身已是为用户写好的中文，最具体，优先采用
    for err in errors:
        if _error_field(err) != "question":
            continue
        message = str(err.get("msg", "")).removeprefix("Value error, ").strip()
        if err.get("type") == "value_error" and message:
            return message

    # 2) 其余情况按「字段 + 错误类型」映射为中文说明
    for err in errors:
        field = _error_field(err)
        err_type = str(err.get("type", ""))
        if field == "question":
            if err_type == "missing":
                return "缺少必填参数：问题"
            if err_type == "string_too_short":
                return "问题不能为空或全为空白字符"
            if err_type == "string_too_long":
                return "问题过长，请控制在 500 字以内"
            return "问题参数不合法，请检查后重试"
        if field == "top_k":
            return "检索片段数量需为 1-20 之间的整数"
        if field == "session_id":
            return "会话 ID 不合法"

    # 3) 兜底：请求体缺失或非法 JSON、未知字段类型等
    return "请求参数不合法，请检查后重试"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    请求参数校验失败处理。

    FastAPI 默认返回 422 + 结构化 errors 数组，这与《接口文档》1.3 的
    「所有错误统一为 {"detail": "中文串"}」以及 400「请求参数错误」不一致，
    故在此转换为 400 + 一句中文提示。
    """
    logger.warning(
        "请求参数校验失败 | 路径=%s | 错误=%s", request.url.path, exc.errors()
    )
    return JSONResponse(
        status_code=400,
        content={"detail": _validation_error_detail(exc)},
    )


# ===========================================================================
# 中间件：为每次请求分配 ID，串联日志
# ===========================================================================

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """给每个请求分配 request_id，并记录访问日志"""
    import time

    from backend.logging_config import get_request_id

    request_id = new_request_id()
    started = time.time()
    response = await call_next(request)
    elapsed_ms = int((time.time() - started) * 1000)

    # 静态资源访问不记录，避免日志噪音
    if not request.url.path.startswith(("/static", "/favicon")):
        logger.info(
            "请求完成 | %s %s | 状态=%d | 耗时=%dms",
            request.method, request.url.path, response.status_code, elapsed_ms,
        )

    # 回传本次请求的 ID，便于对照服务端日志排查问题
    response.headers["X-Request-ID"] = request_id
    return response


# ===========================================================================
# 路由注册
# ===========================================================================

from backend.api import (  # noqa: E402  (须在 app 定义后导入)
    auth,
    build_kb,
    favorite,
    history,
    query,
    upload,
)

app.include_router(build_kb.router)
app.include_router(upload.router)
app.include_router(query.router)
# 增量路由（用户 / 历史记录 / 收藏）：只新增注册行，原注册行一字未动
app.include_router(auth.router)
app.include_router(history.router)
app.include_router(favorite.router)


# ===========================================================================
# 健康检查
# ===========================================================================

@app.get("/api/health", tags=["系统"], summary="健康检查")
def health() -> Dict[str, Any]:
    """
    服务与依赖组件的健康状态。

    部署验证与 JMeter 压测均可使用本接口做探活。
    """
    return {
        "status": "ok",
        "app": settings.app_title,
        "dependencies": {
            "milvus": milvus_client.health_check(),
            "mysql": mysql.health_check(),
            "redis": redis_client.health_check(),
        },
    }


# ===========================================================================
# 前端静态页面托管
# ===========================================================================

_frontend_dir = settings.frontend_path
_index_file = _frontend_dir / "index.html"

if _frontend_dir.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(_frontend_dir)),
        name="static",
    )

    @app.get("/", include_in_schema=False)
    def index() -> Any:
        """返回面向普通用户的问答页面"""
        if _index_file.exists():
            return FileResponse(str(_index_file))
        return JSONResponse(
            status_code=404,
            content={"detail": "前端页面不存在，请检查 frontend/index.html"},
        )
else:
    logger.warning("前端目录不存在：%s", _frontend_dir)


# 用户自选头像（从相册上传的图片）单独托管在 /avatars
# 注意：这段必须放在前端托管块「之外」，否则会挤进 if/else 的缩进层级里
_avatars_dir = settings.avatars_path
try:
    _avatars_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/avatars",
        StaticFiles(directory=str(_avatars_dir)),
        name="avatars",
    )
except Exception as _exc:  # noqa: BLE001
    logger.warning("头像目录挂载失败（自定义头像将无法显示）：%s", _exc)


# ===========================================================================
# 本地启动
# ===========================================================================

def main() -> int:
    """以脚本方式启动服务：python -m backend.main"""
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_config=None,       # 使用本项目自己的 logging 配置
        access_log=False,      # 访问日志由中间件统一记录
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
