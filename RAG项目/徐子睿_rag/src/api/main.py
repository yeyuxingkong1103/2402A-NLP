"""src/api/main.py —— FastAPI 应用装配入口（新架构的服务入口）。

在链路中的位置（最外层）：
    HTTP 客户端 → 【本文件】 → src/api/routers/*（各业务路由）
                             → src/models/database.py、src/offline/milvus_store.py

启动方式（README）：
    python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8902
    （端口 8902；backend 主线用 8901，两条线分开端口才能同时运行）

本文件只做"装配"，不含业务逻辑：
    启动钩子（建表、同步角色、配日志）、trace_id 中间件、全局异常处理、
    挂载六个业务路由、健康检查与根路径。

与 backend/server.py 的关系：
    两者都是 FastAPI 入口，但服务不同的主线 ——
    backend 那条是"单角色知识库问答 + 原生前端"，
    本文件这条是"多用户多角色 + JWT + 统一错误响应"的完整后端。
    本项目同时保留两者，是为了让两条路都能独立跑通、互不依赖。
"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from configs.settings import get_settings
from src.api.routers import auth, chat, feedback, knowledge, role, session
from src.models.database import init_db
from src.offline.milvus_store import store
from src.utils.logging_config import configure_logging, trace_id_var


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子：启动时做一次性初始化。

    参数：
        app: FastAPI 实例（框架传入，本函数未使用）

    启动时做三件事（顺序有意义）：
        1. configure_logging —— 最先执行，后面两步的日志才能带上正确格式和 trace_id
        2. init_db           —— 建表。必须早于 sync_roles，因为后者要往表里写数据
        3. role.sync_roles   —— 把内置角色同步进库

    用 yield 分隔启动与关闭：
        yield 之前的代码在服务启动时执行，之后的在关闭时执行。
        本项目没有需要在关闭时清理的资源（连接由各模块自己管理），所以 yield 后为空。

    为什么用 lifespan 而不是已废弃的 @app.on_event("startup")：
        on_event 在 FastAPI 新版中已不推荐，lifespan 是官方当前的做法，
        且能把启动与关闭逻辑写在同一处、共享同一个作用域。
    """
    settings = get_settings()
    configure_logging(settings.log_dir)
    init_db()
    role.sync_roles()
    yield


app = FastAPI(title="RAG Roleplay System", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """为每个请求分配并透传 trace_id（全链路追踪的起点）。

    参数：
        request: 请求对象
        call_next: 调用下一个处理器（中间件链的下一环）
    返回：
        原始响应，额外带上 X-Trace-Id 响应头。

    三个动作：
        1. 从请求头取 X-Trace-Id，没有就生成一个（说明是这条链路的起点）
        2. 存进 ContextVar，让本次请求内所有日志自动带上它
        3. finally 里 reset —— 这一步很关键

    为什么 finally 里必须 reset：
        ContextVar 是"上下文局部"的，但同一个工作线程会先后处理多个请求。
        不 reset 的话，上一个请求的 trace_id 会残留下来，
        被下一个尚未设置 trace_id 的请求（或抛出异常后的收尾逻辑）误用，
        日志里就会出现"串台"的 trace_id —— 这比没有 trace_id 更误导人。
        reset 要传 set 返回的 token，才能精确还原到设置前的状态。

    回写响应头 X-Trace-Id：
        让前端/调用方能拿到这次请求的 id 并反馈给运维，
        出问题时凭一个 id 就能捞出这次请求的所有日志。
    """
    trace_id = request.headers.get("X-Trace-Id", uuid.uuid4().hex)
    token = trace_id_var.set(trace_id)
    try:
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response
    finally:
        trace_id_var.reset(token)


@app.exception_handler(Exception)
async def error_handler(request: Request, exc: Exception):
    """全局异常兜底：把任何未捕获异常统一成标准错误响应。

    参数：
        request: 请求对象
        exc: 抛出的异常
    返回：
        JSONResponse，状态码 500，结构为 {"code", "message", "data", "trace_id"}。

    统一响应结构的意义（README 里写明"统一错误响应包含 code/message/data/trace_id"）：
        调用方只需写一套解析逻辑，不用为每个接口猜错误格式；
        trace_id 让"用户报错"到"日志定位"之间的链路闭合。

    注意这里直接把 str(exc) 返回给客户端：
        方便调试，但在生产环境可能泄露内部实现细节（如数据库连接串、文件路径）。
        如果这个服务要对外，应改成返回通用文案 + 把详情只写日志。
        当前定位是本地可跑通的教学/演示项目，能看清错误更有价值。
    """
    trace_id = trace_id_var.get()
    return JSONResponse(status_code=500, content={"code": 500, "message": str(exc), "data": None, "trace_id": trace_id})


# 挂载六个业务路由。每个路由模块自带 prefix（如 /api/v1/auth），这里只做 include
app.include_router(auth.router)
app.include_router(role.router)
app.include_router(session.router)
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(feedback.router)


@app.get("/health")
def health():
    """健康检查：只探 Milvus 这一项关键依赖。

    返回：
        {"code": 0 或 503, "message": "ok" 或 "degraded", "trace_id", "data": {"milvus": {...}}}

    两个刻意的设计：
        1. 用 code 字段表达状态（0 正常 / 503 降级），而不是返回 HTTP 503
           —— 这样监控脚本能拿到结构化信息，不必解析 HTTP 状态码与响应体的组合
        2. 明确区分 ok / degraded：
           Milvus 挂了时检索和知识库管理不可用，但注册登录、会话管理、
           反馈这些功能仍然正常（它们只依赖关系库）。
           报 degraded 能指明"是依赖服务的问题"，而不是让你以为整个应用崩了。
    """
    milvus_ok, milvus_detail = store.health()
    return {
        "code": 0 if milvus_ok else 503,
        "message": "ok" if milvus_ok else "degraded",
        "trace_id": trace_id_var.get(),
        "data": {"milvus": {"ok": milvus_ok, "detail": milvus_detail}},
    }


@app.get("/")
def root():
    """根路径：给出文档和健康检查的地址（服务自描述）。"""
    return {"name": "RAG Roleplay System", "docs": "/docs", "health": "/health"}
