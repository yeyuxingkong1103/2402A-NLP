# -*- coding: utf-8 -*-
"""FastAPI 应用：HTTP JSON 接口。

接口清单
--------
=============================  ==========================================================
方法 + 路径                     说明
=============================  ==========================================================
``GET``     ``/health``          组件健康：聚合状态 + 各后端可达性/延迟 + GPU + 并发 + 活条数
``GET``     ``/metrics``         Prometheus 文本（业务 + 系统层），**轻量**（不触发检索/生成）
``GET``     ``/roles``           角色库
``POST``    ``/sessions``        创建/登记会话
``GET``     ``/sessions``        某用户的会话列表
``POST``    ``/chat``            一次问答（可流式；多轮由 session 维护）
``POST``    ``/ingest``          重建/增量更新知识库
``GET``     ``/documents/upload``     上传能力自描述（允许的扩展名/上限）
``POST``    ``/documents/upload``     上传 PDF/txt/md/json 并**立即入库**（返回 job_id）
``GET``     ``/documents``            已入库文档列表
``GET``     ``/documents/jobs``       最近的上传入库 job
``GET``     ``/documents/jobs/{id}``  单个 job 进度（pending/running/completed/failed）
``DELETE``  ``/documents/{doc_id}``   删除该文档的向量分块 + 磁盘文件
=============================  ==========================================================

并发模型（详见 docs/API.md）
---------------------------
* 端点全部 ``async def``；所有**同步阻塞**调用（pymilvus / BM25 / Ollama）经
  :func:`legal_rag.api.concurrency.run_blocking` 卸载到 anyio 线程池
  （大小 = ``config.concurrency.thread_pool_size``），事件循环不会被堵死；
* 重请求（chat / ingest / upload）先抢全局信号量
  （``config.concurrency.max_concurrent_requests``），排队超时返回 **429**；
* 同一 ``(user_id, role_id, session_id)`` 的「读历史 → 生成 → 追加历史」用**每会话写锁**
  串行，避免并发下历史错乱；
* Milvus / Redis / Ollama 客户端在 ``lifespan`` 里建一次、全程复用。

隔离约定：所有业务请求必须携带 user_id + role_id + session_id；
短期记忆按这三元组做键，跨用户/跨会话从存储层面即不互通。
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, StreamingResponse)

from .. import metrics as M
from ..config import KNOWLEDGE_DIR, RagConfig
from ..generate.llm_base import failure_reason, summarize_exception
from ..generate.llm_base import LLMError
from ..generate.prompt import build_memory_block
from ..logging_setup import get_logger, setup_logging
from ..memory.session import message_id_for, split_window, with_pair_ids
from ..observability import bind_request, current_request_id, new_request_id, reset_request
from ..roles import get_role
from .concurrency import RETRY_AFTER_SECONDS, QueueFull, run_blocking
from .deps import AppState

logger = get_logger("legal_rag.api.app")

# ---------------- 模块级常量（2026-09-29 拆到 api/constants.py）----------------
# WEB_DIR / UI_PATH / AUTH_PAGE_FILES / PAGE_CACHE_CONTROL / REQUEST_ID_HEADER /
# _SNIFF_LIMIT / _REJECT_COUNTER / _LLM_REASON_TEXT 原先散落在本文件顶部与末尾，
# 现集中到 `constants.py` —— 路由模块与服务层都要用它们，集中一处才能避免互相
# import 成环。`PAGE_CACHE_CONTROL` 与 `REQUEST_ID_HEADER` 是行为契约，不是可调参数。
from .constants import (  # noqa: E402 - 保持与既有 import 分组的相对位置
    AUTH_PAGE_FILES, LLM_REASON_TEXT, PAGE_CACHE_CONTROL, REJECT_COUNTER,
    REQUEST_ID_HEADER, SNIFF_LIMIT, UI_PATH, WEB_DIR,
)
# SSE 帧构造与流包装（2026-09-29 拆到 api/services/streaming.py）
from .services.streaming import sse_generator
# 上传的校验与落盘（2026-09-29 拆到 api/services/uploads.py）
# 跨路由复用的 HTTP 小工具（2026-09-29 拆到 api/services/shared.py）。
# 保留 `_` 前缀别名，避免改动大量既有调用点、放大 diff。
from .services.shared import path_label as _path_label
from .services.shared import request_token as _request_token
# 依赖注入函数（2026-09-29 拆到 api/dependencies.py）：把原先的闭包助手变成
# "只依赖 Request"的可注入函数，供 routers/* 与 app.py 共同使用。
from .dependencies import page_user
# 路由模块（2026-09-29 起逐步从 create_app 内搬出）
from .routers import auth as auth_router
from .routers import documents as documents_router
from .routers import ops as ops_router
from .routers import prefs as prefs_router
from .routers import sessions as sessions_router

# 兼容既有内部名（下文多处使用旧的下划线前缀写法，改名会放大 diff）
_SNIFF_LIMIT = SNIFF_LIMIT
_REJECT_COUNTER = REJECT_COUNTER
_LLM_REASON_TEXT = LLM_REASON_TEXT


def iter_api_routes(app: FastAPI) -> list[Any]:
    """枚举 ``app`` 上**全部** ``APIRoute``，**含**通过 ``include_router`` 挂载的。

    为什么需要它（2026-09-30 踩到）：FastAPI 0.141 的 ``include_router`` **不再把子路由
    拍平进 ``app.routes``**，而是往列表里放一个 ``_IncludedRouter`` 容器
    （懒匹配，连 ``.routes`` 属性都没有）。于是"遍历 ``app.routes`` 找 ``APIRoute``"
    这套写法**会静默漏掉**所有经路由模块挂载的端点 ——
    ``scripts/export_postman.py`` 与 ``tests/test_export_postman.py`` 当时都这么写，
    结果 Postman 集合少了 ``/health`` ``/livez`` ``/metrics`` ``/roles`` 四条。

    本函数是**唯一的枚举出口**：新代码要遍历路由一律用它，不要再自己写
    ``for route in app.routes``。容器类型通过 ``original_router`` 展开
    （该属性是 ``_IncludedRouter`` 的公开 API）。
    """
    from fastapi.routing import APIRoute

    out: list[Any] = []
    stack: list[Any] = list(app.routes)
    while stack:
        route = stack.pop(0)
        if isinstance(route, APIRoute):
            out.append(route)
            continue
        # `_IncludedRouter`（FastAPI ≥0.141）等容器：展开它持有的原始 router
        inner = getattr(route, "original_router", None)
        if inner is not None:
            stack.extend(getattr(inner, "routes", []) or [])
    return out


def _page_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """页面响应头的**唯一出口**：四个页面都从这里取头，避免以后新增页面漏加缓存策略。"""
    headers = {"Cache-Control": PAGE_CACHE_CONTROL}
    if extra:
        headers.update(extra)
    return headers


# ---------------- 请求 / 响应模型（2026-09-29 拆到独立模块）----------------
#
# 原先 16 个 Pydantic 模型堆在本文件里（约 155 行），与路由、装配逻辑混在一起。
# 它们没有任何闭包依赖、也不调用业务代码 ⇒ 拆到 `api/schemas.py` 零风险，
# 且能被多个路由模块与服务层共享。字段名/默认值/约束是对外契约，改动前先读那里的注释。
from .schemas import (  # noqa: E402 - 保持与既有 import 分组的相对位置
    ChatRequest, ChatResponse, Citation, IngestRequest,
    SessionCreateRequest, SessionMessage, SessionMessagesResponse,
    SessionRenameRequest, SessionResponse,
)

# ---------------- 应用 ----------------

def create_app(config: RagConfig | None = None, state: AppState | None = None) -> FastAPI:
    setup_logging()
    cfg = config or RagConfig.from_env()
    app_state = state or AppState(config=cfg, sources=cfg_sources(cfg))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # 一次装配、全程复用（Milvus/Redis/Ollama 客户端都不每请求新建）
        app_state.build()
        app_state.blocking_limiter()          # 在事件循环里把线程池上限建好
        app_state.start_sampler()
        app_state.ensure_index()
        logger.info("法律 RAG 服务已就绪")
        try:
            yield
        finally:
            app_state.close()

    app = FastAPI(title="法律 RAG 服务", version="0.2.0", lifespan=lifespan)
    app.state.app_state = app_state

    # ---------- 路由模块（2026-09-29 起从本函数内搬出）----------
    # 搬出的路由**路径/方法/状态码/响应模型一字不变**；依赖通过
    # `api/dependencies.py` 的 Depends 注入（`app_state` 从 app.state 取回）。
    # 每搬一个域就用一次全量回归验证，见 docs/REFACTOR-PLAN.md。
    app.include_router(ops_router.router)
    app.include_router(auth_router.router)
    app.include_router(prefs_router.router)
    app.include_router(sessions_router.router)
    app.include_router(documents_router.router)

    # ---------- 中间件：request_id + HTTP 指标 + 排队/限流语义 ----------
    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        token = bind_request(request_id, path=request.url.path)
        path_label = _path_label(request)
        started = time.perf_counter()
        inflight = M.gauge("inflight_requests")
        inflight.inc(path=path_label)
        try:
            response = await call_next(request)
        except QueueFull as exc:                       # 排队超时 -> 429
            logger.warning("请求被限流：%s %s（%s）", request.method, request.url.path, exc)
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": str(exc), "request_id": request_id},
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            )
        except Exception as exc:                       # 未预期异常也要能定位
            # t82：5xx 响应体**不回填异常文本**。异常类名 / 内网 host:port / collection 名
            # 都是内部信息，泄漏给客户端既无助于用户、又暴露内网结构。
            # 处置：**完整异常与堆栈只进日志**（带 request_id），响应体只给通用文案 + request_id，
            # 客户端报 request_id，服务端一条 grep 就能定位 —— 可追溯性一点没丢。
            logger.exception("请求处理失败：%s %s request_id=%s 异常=%s",
                             request.method, request.url.path, request_id,
                             type(exc).__name__)
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "服务内部错误（internal error），请稍后重试；"
                                   "如需排查请提供 request_id",
                         "request_id": request_id})
        finally:
            elapsed = time.perf_counter() - started
            inflight.dec(path=path_label)
            M.observe("request_seconds", elapsed, path=path_label)
            M.counter("requests_total", "HTTP 请求数（path/status）", "count").inc(
                path=path_label, status=str(response.status_code))
            if elapsed > 1.0:
                logger.info("请求完成：%s %s -> %s 耗时 %.2fs request_id=%s",
                            request.method, request.url.path, response.status_code,
                            elapsed, request_id)
            reset_request(token)

        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    # ---------- 依赖 ----------------
    def require_engine():
        if app_state.engine is None:
            raise HTTPException(status_code=503, detail="引擎尚未就绪")
        return app_state.engine

    def require_documents():
        if app_state.documents is None:
            raise HTTPException(status_code=503, detail="文档服务尚未就绪")
        return app_state.documents

    def require_identity(request: Request, claimed_user_id: str | None = None) -> dict | None:
        """强制鉴权开关打开时的「身份 + 授权」闸门（`AC-AU-29`）。

        * 开关**关闭**（默认）→ 返回 ``None``，端点继续用请求体里的 `user_id`，
          既有行为与既有测试**一行都不变**；
        * 开关打开 + 无令牌/伪造/已登出/过期 → **401** ``unauthenticated``；
        * 开关打开 + 登录态与 ``claimed_user_id`` 不一致 → **403** ``user_mismatch``。

        这就是"带 A 的 token 用 B 的 user_id"的封堵点：`user_id` 只能来自
        服务端令牌行（`auth_tokens.user_id`），请求体里的那个只被用来**比对**，
        永远不被信任。选 403 而不是"静默覆盖"的理由见 `api/auth.py` 模块头 §3。
        """
        if app_state.auth is None:
            raise HTTPException(status_code=503, detail="账号服务尚未就绪")
        decision = app_state.auth.authorize(_request_token(request), claimed_user_id)
        if decision is None:
            return None
        if "error" in decision:
            error = decision["error"]
            raise HTTPException(status_code=error["status_code"],
                                detail={"error": error["error"],
                                        "message": error["message"]})
        return decision["user"]

    def auth_required_only(request: Request) -> dict | None:
        """只要令牌有效即可（不比对 user_id 的端点，如 `/documents`、`/documents/upload`）。"""
        return require_identity(request, None)

    def client_key(request: Request) -> str:
        """限速用的来源标识（纯标准库；不解析 X-Forwarded-For —— 那可以被伪造）。"""
        host = request.client.host if request.client else "unknown"
        return f"{host}|{request.url.path}"

    def set_session_cookie(response: JSONResponse, token: str) -> None:
        if app_state.auth is None:
            return
        response.set_cookie(
            key=app_state.auth.cookie_name, value=token,
            max_age=app_state.auth.token_ttl_seconds,
            **app_state.auth.cookie_attributes())

    def clear_session_cookie(response: JSONResponse) -> None:
        if app_state.auth is None:
            return
        response.delete_cookie(
            key=app_state.auth.cookie_name,
            path=str(app_state.auth.cookie_attributes().get("path") or "/"))

    # ---------- 认证页面（第二批：独立页面 + 页面级登录守卫）----------
    #
    # ⚠️ 页面级判定**与 API 的 `AUTH_REQUIRED` 解耦**：网页端**永远要求登录**
    #    （未登录访问 `/ui` 一律 302 到 `/login`），而业务 API 仍按 `AUTH_REQUIRED`
    #    的语义工作（默认关闭时既有调用方一行都不用改；API 层**不得**用 302 表达未登录）。
    #
    # `page_user` 的实现已搬到 `api/dependencies.py`（从闭包变成依赖注入函数），
    # 这里用同名导入保持调用点不变。

    def auth_page(kind: str) -> Response:
        """返回认证页：优先 `web/<kind>.html`；**文件不存在就回最小兜底页**（不是 404）。

        ⚠️ **两个分支都带 `Cache-Control: no-store`**（见 `PAGE_CACHE_CONTROL`）：
        页面是应用壳，被浏览器复用旧 body 会导致"看到旧壳"的旧行为（旧壳没有跳转兜底/超时）。
        """
        path = WEB_DIR / AUTH_PAGE_FILES[kind]
        if path.is_file():
            logger.debug("认证页 %s 由文件提供（%s）；Cache-Control=%s",
                         kind, path.name, PAGE_CACHE_CONTROL)
            return FileResponse(path, media_type="text/html; charset=utf-8",
                                headers=_page_headers())
        logger.warning("认证页文件缺失，使用最小兜底页：%s（同样带 Cache-Control=%s，"
                       "否则浏览器会缓存这页兜底壳）", path, PAGE_CACHE_CONTROL)
        return HTMLResponse(_fallback_auth_page(kind), status_code=200,
                            headers=_page_headers())

    @app.get("/login", include_in_schema=False)
    async def login_page(request: Request) -> Response:
        """默认首屏 = 登录页（`AC-AU-38`）。已登录访问 → 302 `/ui`（`AC-AU-43`）。"""
        if page_user(request) is not None:
            return RedirectResponse(url="/ui", status_code=status.HTTP_302_FOUND)
        return auth_page("login")

    @app.get("/register", include_in_schema=False)
    async def register_page(request: Request) -> Response:
        if page_user(request) is not None:
            return RedirectResponse(url="/ui", status_code=status.HTTP_302_FOUND)
        return auth_page("register")

    @app.get("/reset", include_in_schema=False)
    async def reset_page(request: Request) -> Response:
        """找回密码页（`AC-RC-1`：未登录可直接访问）。"""
        if page_user(request) is not None:
            return RedirectResponse(url="/ui", status_code=status.HTTP_302_FOUND)
        return auth_page("reset")

    # ---------- 问答 ----------
    @app.post("/chat")
    async def chat(req: ChatRequest, request: Request):
        engine = require_engine()
        assert app_state.sessions is not None

        # 强制鉴权开关打开时：登录态必须与请求体 user_id 一致（403 user_mismatch）。
        # 关掉开关也**不等于**没有防线：下面的会话归属校验用的是会话表里的真实
        # user_id，因此即便有人伪造 user_id，也写不进别人的会话（AC-AU-29 的兜底）。
        identity = require_identity(request, req.user_id)
        # 长期记忆召回用的用户维度：**优先取服务端令牌里的 user_id**（开关打开时它已被
        # 校验过等于 req.user_id）；开关关闭时才退回请求体（与短期记忆的既有信任模型一致）。
        effective_user = str((identity or {}).get("user_id") or req.user_id)

        # 会话归属校验：session_id 已属于别人时直接拒绝，防止串号
        if app_state.business is not None:
            existing = await run_blocking(app_state.business.get_session, req.session_id,
                                          limiter=app_state.blocking_limiter())
            if existing is not None and existing.get("user_id") != req.user_id:
                raise HTTPException(status_code=403, detail="该会话不属于当前用户")

        role = get_role(req.role_id)
        session_key = (app_state.session_locks.key_of(req.user_id, req.role_id,
                                                      req.session_id)
                       if app_state.session_locks is not None else "")

        if req.stream:
            return await _stream_chat(app_state, req, role, session_key, effective_user)
        return await _blocking_chat(app_state, req, role, session_key, effective_user)

    async def _memory_block_for(app_state: AppState, req: ChatRequest, user_id: str,
                                limiter) -> str:
        """召回长期记忆（阻塞活，卸载到线程池）；返回**提示词分区**文本，无记忆时为空串。"""
        try:
            return await run_blocking(_recall_memory_block, app_state, user_id,
                                      req.role_id, req.message, limiter=limiter)
        except Exception as exc:  # noqa: BLE001 - run_blocking 层的失败同样不能静默
            logger.warning("[B-9] 长期记忆召回任务失败（user=%s）：%s: %s —— 本轮按无记忆继续",
                           user_id, type(exc).__name__, exc)
            return ""

    async def _blocking_chat(app_state: AppState, req: ChatRequest, role, session_key: str,
                             effective_user: str = ""):
        """非流式：整条链路都在线程池里跑，事件循环只等结果。"""
        assert app_state.limiter is not None and app_state.session_locks is not None
        assert app_state.sessions is not None
        limiter = app_state.blocking_limiter()
        async with app_state.limiter.slot("chat"):
            # 同一会话串行：读历史 -> 生成 -> 写回，避免并发交错导致历史错乱
            async with app_state.session_locks.hold(session_key):
                history = await run_blocking(app_state.sessions.prompt_history, req.user_id,
                                             req.role_id, req.session_id, limiter=limiter)
                memory_block = await _memory_block_for(app_state, req,
                                                       effective_user or req.user_id, limiter)
                try:
                    answer = await run_blocking(app_state.engine.ask, req.message, role,
                                                req.session_id, req.user_id, history,
                                                req.top_k, None, memory_block,
                                                limiter=limiter)
                except LLMError as exc:
                    # t120 F1：后端全挂时**不**给假答案 —— 显式报错 + 可读原因 + 可重试
                    raise _llm_unavailable(exc) from exc
                await run_blocking(_remember_turn, app_state, req, answer.text,
                                   answer.citations, limiter=limiter)
        return ChatResponse(
            answer=answer.text,
            citations=[Citation(**cite) for cite in answer.citations],
            session_id=req.session_id,
            role_id=answer.role_id,
            provider=answer.provider,
            degraded=answer.degraded,
            degraded_reason=answer.degraded_reason,
        )

    async def _stream_chat(app_state: AppState, req: ChatRequest, role, session_key: str,
                           effective_user: str = ""):
        """流式：生成器里**不做**任何阻塞调用（每次取值都卸载到线程池）。

        并发流式互不串包的关键：每次请求各自持有一份 ``iterator`` 与两个上下文
        （全局槽位 + 会话锁），线程池只负责推进自己那一路。
        """
        assert app_state.limiter is not None and app_state.session_locks is not None
        assert app_state.sessions is not None
        limiter = app_state.blocking_limiter()
        slot = app_state.limiter.slot("chat")
        await slot.__aenter__()
        lock_ctx = app_state.session_locks.hold(session_key)
        await lock_ctx.__aenter__()
        try:
            history = await run_blocking(app_state.sessions.prompt_history, req.user_id,
                                         req.role_id, req.session_id, limiter=limiter)
            memory_block = await _memory_block_for(app_state, req,
                                                   effective_user or req.user_id, limiter)
            try:
                answer, iterator = await run_blocking(
                    app_state.engine.ask_stream, req.message, role, req.session_id,
                    req.user_id, history, req.top_k, None, memory_block,
                    limiter=limiter)
            except LLMError as exc:
                # 流式也一样：router.stream 会先探第一个 chunk，所以这一步就能判定"全挂"，
                # 在**任何字节发出之前**返回 503（不让前端收到半截流 + 假答案）。
                raise _llm_unavailable(exc) from exc
        except Exception:
            await lock_ctx.__aexit__(None, None, None)
            await slot.__aexit__(None, None, None)
            raise

        request_id = current_request_id() or ""

        def next_chunk():
            try:
                return next(iterator)
            except StopIteration:
                return None

        async def generator():
            try:
                while True:
                    piece = await run_blocking(next_chunk, limiter=limiter)
                    if piece is None:
                        break
                    if piece:
                        yield piece
                await run_blocking(_remember_turn, app_state, req, answer.text,
                                   answer.citations, limiter=limiter)
            finally:
                await lock_ctx.__aexit__(None, None, None)
                await slot.__aexit__(None, None, None)

        if req.stream_mode == "sse":
            return StreamingResponse(sse_generator(answer, generator()),
                                     media_type="text/event-stream",
                                     headers={REQUEST_ID_HEADER: request_id,
                                              "Cache-Control": "no-cache",
                                              "X-Accel-Buffering": "no"})
        return StreamingResponse(generator(), media_type="text/plain; charset=utf-8",
                                 headers={REQUEST_ID_HEADER: request_id})

    # ---------- 入库 ----------
    # ---------- 文档：上传 / 列表 / job / 删除 ----------
    @app.get("/ui", include_in_schema=False)
    async def ui(request: Request) -> Response:
        """托管单页前端（无构建、零依赖；用户交互都在这里）。

        **页面级登录守卫**（`AC-AU-40/41`，与 `AUTH_REQUIRED` 解耦）：未登录一律
        302 到 `/login` —— 因此本响应里**不可能**出现登录/注册表单（服务端可判）。

        ⚠️ 页面必须带 `Cache-Control: no-store`（见 `PAGE_CACHE_CONTROL` 的注释）：
        `FileResponse` 默认不发这个头，浏览器会复用旧壳 → 用户"进对话页一直转圈"。
        """
        if page_user(request) is None:
            return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
        if not UI_PATH.is_file():
            logger.error("对话页文件缺失：%s（/ui 只能回 404）", UI_PATH)
            raise HTTPException(status_code=404,
                                detail={"reason": "ui_missing", "path": str(UI_PATH)})
        logger.debug("对话页 /ui 由文件提供（%s）；Cache-Control=%s",
                     UI_PATH.name, PAGE_CACHE_CONTROL)
        return FileResponse(UI_PATH, media_type="text/html; charset=utf-8",
                            headers=_page_headers())

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/ui")

    # ---------- 入库前判定：这份资料是否符合该角色设定 ----------



    # 页面新鲜度策略**在日志里也能看到**（不静默）：一行启动日志 + 每个页面响应的响应头。
    # 正常请求路径只在 debug 级打点（避免每次刷新都刷日志），但这个策略本身必须可审计。
    logger.info("页面缓存策略：/ui 与 /login /register /reset 一律 Cache-Control=%s"
                "（业务 API 端点不加，保持既有缓存语义）", PAGE_CACHE_CONTROL)
    return app


# ---------------- 内部工具（阻塞部分都在线程池里跑）----------------

def _fallback_auth_page(kind: str) -> str:
    """三个认证页的**最小兜底页**（`web/<kind>.html` 还没落地时用）。

    为什么要有它：前端页面由另一个任务产出，后端不能因为"文件还没生成"就 404 ——
    那样整条链路会断在这里（连"页面是否存在"都测不出来）。兜底页刻意满足
    §13.1 的硬要求：`lang=zh-CN`、**恰好一个** `<form>`、唯一 `<h1>`、label 在上、
    主按钮满宽；`/login` 另带「忘记密码」→ `/reset`、「注册账号」→ `/register` 两个链接。

    ⚠️ 注册页特别处理：成功后**不能直接跳 `/ui`** —— 恢复码只在注册响应里出现这一次
    （`AC-AU-49/52`），直接跳走就等于把码丢了。所以兜底页把码**大字显示**，并要求用户输入
    「我已得知这是恢复码及其功能」才放开「我已保存，进入对话」。
    """
    titles = {"login": "登录", "register": "注册账号", "reset": "重置密码"}
    actions = {"login": "/auth/login", "register": "/auth/register", "reset": "/auth/reset"}
    title = titles.get(kind, "登录")
    action = actions.get(kind, "/auth/login")
    if kind == "reset":
        fields = (("username", "用户名", "text"),
                  ("recovery_code", "恢复码", "text"),
                  ("new_password", "新密码", "password"))
    elif kind == "register":
        fields = (("username", "用户名", "text"), ("password", "密码", "password"))
    else:
        fields = (("username", "用户名", "text"), ("password", "密码", "password"))
    inputs = "\n".join(
        f'      <label for="{name}">{label}</label>\n'
        f'      <input id="{name}" name="{name}" type="{itype}" '
        f'autocomplete="{"new-password" if itype == "password" else "username"}" required>'
        for name, label, itype in fields)
    links = ('      <p class="row"><a href="/reset">忘记密码</a> · '
             '<a href="/register">注册账号</a></p>\n') if kind == "login" else ""
    note = ""
    if kind == "register":
        note = ('      <p class="hint">注册成功后会给你一枚<b>一次性恢复码</b>：'
                '仅此一次、立即保存，找不回就只能重新注册。</p>\n')
    elif kind == "reset":
        note = ('      <p class="hint">用注册时保存的<b>一次性恢复码</b>重置密码：'
                '不收集手机号或邮箱，也不发验证码。</p>\n')
    show_code = "true" if kind == "register" else "false"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body{{margin:0;font:14px/1.6 system-ui,"Segoe UI",sans-serif;background:#0F1115;color:#E6E9EF}}
    main{{max-width:280px;margin:8vh auto;padding:24px;background:#161A21;border-radius:10px}}
    h1{{font-size:20px;margin:0 0 16px}}
    h2{{font-size:16px;margin:0 0 8px}}
    form{{display:flex;flex-direction:column;gap:8px}}
    label{{display:block;font-size:13px;margin-top:8px}}
    input{{width:100%;box-sizing:border-box;padding:8px 10px;min-height:40px;
           background:#1D222B;color:inherit;border:1px solid #6B7484;border-radius:8px}}
    button{{width:100%;min-height:40px;margin-top:16px;border:0;border-radius:8px;
            background:#4C8DFF;color:#0F1115;font-size:14px;cursor:pointer}}
    button[disabled]{{background:#39414F;color:#98A2B3;cursor:not-allowed}}
    .hint{{font-size:12.5px;color:#98A2B3}}
    .row{{font-size:12.5px;margin:12px 0 0}}
    a{{color:#4C8DFF}}
    .recovery{{margin-top:16px;padding-top:12px;border-top:1px solid #39414F}}
    .recovery .code{{margin:8px 0;padding:10px;background:#1D222B;border:1px dashed #4C8DFF;
                     border-radius:8px;font:16px/1.5 ui-monospace,Consolas,monospace;
                     word-break:break-all;user-select:all}}
  </style>
</head>
<body>
  <main>
    <h1>{title}</h1>
    <form id="{kind}-form" method="post" action="javascript:void(0)">
{inputs}
      <button type="submit">{title}</button>
    </form>
{note}{links}  </main>
  <script>
    var PHRASE = "我已得知这是恢复码及其功能";

    function showRecovery(code) {{
      document.getElementById("{kind}-form").style.display = "none";
      var panel = document.createElement("div");
      panel.className = "recovery";
      panel.innerHTML = '<h2>你的恢复码（只显示这一次）</h2>' +
        '<p class="code" id="recovery-code"></p>' +
        '<p class="hint">它是找回密码的唯一凭据：服务端只存它的哈希，' +
        '拿不回、丢了就只能重新注册。请立刻抄下来或存进密码管理器。</p>' +
        '<label for="recovery-ack">请输入「' + PHRASE + '」以确认你已得知</label>' +
        '<input id="recovery-ack" autocomplete="off">' +
        '<button id="recovery-go" disabled>我已保存，进入对话</button>';
      document.querySelector("main").appendChild(panel);
      document.getElementById("recovery-code").textContent = code;
      var ack = document.getElementById("recovery-ack");
      ack.addEventListener("input", function () {{
        document.getElementById("recovery-go").disabled = ack.value.trim() !== PHRASE;
      }});
      document.getElementById("recovery-go").addEventListener("click", function () {{
        location.href = "/ui";
      }});
      ack.focus();
    }}

    document.getElementById("{kind}-form").addEventListener("submit", function (event) {{
      event.preventDefault();
      var body = {{}};
      this.querySelectorAll("input").forEach(function (el) {{ body[el.name] = el.value; }});
      fetch("{action}", {{method: "POST", headers: {{"Content-Type": "application/json"}},
                        body: JSON.stringify(body), credentials: "same-origin"}})
        .then(function (r) {{ return r.json().then(function (d) {{ return [r.status, d]; }}); }})
        .then(function (pair) {{
          if (pair[0] >= 200 && pair[0] < 300) {{
            var data = pair[1] || {{}};
            if ({show_code} && data.recovery_code) {{ showRecovery(data.recovery_code); return; }}
            location.href = "/ui"; return;
          }}
          var d = pair[1] || {{}};
          alert((d.message || (d.detail && d.detail.message)) || "提交失败，请重试");
        }})
        .catch(function () {{ alert("网络异常，请重试"); }});
    }});
  </script>
</body>
</html>
"""




def _recall_memory_block(app_state: AppState, user_id: str, role_id: str,
                         question: str) -> str:
    """B-9 **长期记忆召回**（读路径）：把该用户语义最相近的旧对话渲染成提示词分区。

    为什么单独一个函数：`/chat` 里有两条分支（阻塞 / 流式）都要用同一份口径，
    复制两份必然走偏。

    红线（与写入侧同源）：

    * **用户维度只能来自服务端**：``user_id`` 由调用方从登录态（或会话归属校验后的
      请求体）传入，**永远不接受**任何客户端 filter；``recall()`` 内部还会用
      ``doc_id`` 做存储层过滤、再用 ``source`` 前缀做应用层复核（两道防线）。
    * **失败必须看得见**：召回异常一律 WARNING + 计数器，**本轮降级为"没有记忆"继续答**
      （记忆缺失不该让用户拿不到法律答案），但绝不允许静默。
    * **不参与引用**：召回片段只进提示词，**绝不**交给 ``build_citations()`` ——
      用户自己的旧话不是法条。
    """
    config = getattr(app_state, "config", None)
    if not bool(getattr(config, "longterm_recall_enabled", False)):
        M.counter("longterm_recall_total").inc(status="disabled")
        return ""
    memory = getattr(app_state, "longterm", None)
    if memory is None or not getattr(memory, "enabled", False):
        M.counter("longterm_recall_total").inc(status="disabled")
        logger.info("[B-9] 召回开关已开，但长期记忆未启用（LONGTERM_ENABLED=false），本轮不召回")
        return ""
    top_k = int(getattr(config, "longterm_recall_top_k", 3) or 3)
    min_chars = int(getattr(config, "longterm_recall_min_chars", 8) or 8)
    try:
        hits = memory.recall(question, user_id=user_id, role_id=role_id, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - 召回失败不能拖垮问答，但必须可见
        M.counter("longterm_recall_total").inc(status="failed")
        logger.warning("[B-9] 长期记忆召回失败（user=%s role=%s request_id=%s）：%s: %s"
                       " —— 本轮按「没有记忆」继续，用户仍能拿到法律答案",
                       user_id, role_id, current_request_id() or "-",
                       type(exc).__name__, exc)
        return ""
    texts = []
    for hit in hits or []:
        text = str(getattr(getattr(hit, "chunk", None), "text", "") or "").strip()
        if len(text) >= min_chars:
            texts.append(text)
    block = build_memory_block(texts)
    if not block:
        M.counter("longterm_recall_total").inc(status="empty")
        logger.info("[B-9] 长期记忆召回为空（user=%s role=%s top_k=%d），本轮不注入记忆",
                    user_id, role_id, top_k)
        return ""
    M.counter("longterm_recall_total").inc(status="ok")
    M.counter("longterm_recall_hits_total").inc(len(texts))
    logger.info("[B-9] 长期记忆召回成功：user=%s role=%s 命中 %d 条、注入 %d 字",
                user_id, role_id, len(texts), len(block))
    return block


def _turn_rows(app_state: AppState, req: ChatRequest, answer_text: str,
               citations: list[dict] | None, *, seq: int) -> list[dict]:
    """把一轮问答的两条消息整理成关系库行（`B-9` 的权威全量历史）。

    ⚠️ 两条消息的 ``message_id`` 按**位置**取（``seq`` / ``seq+1``），不是按内容 ——
    否则"同一会话重问同一个问题"会让用户消息撞 ID 被跳过、助手消息照写，
    在会话历史里留下**没有提问的孤儿回答**（详见 `memory/session.py::message_id_for`）。
    """
    user_id = str(req.user_id)
    session_id = str(req.session_id)
    pairs = (("user", req.message, []),
             ("assistant", answer_text, list(citations or [])))
    now = time.time()
    rows: list[dict] = []
    for offset, (role, content, cites) in enumerate(pairs):
        rows.append({
            "message_id": message_id_for(user_id, session_id, role, content,
                                         seq=seq + offset),
            "session_id": session_id, "user_id": user_id, "role": role,
            "content": content, "citations": cites,
            "turn_index": max((seq - 1) // 2, 0), "seq": seq + offset,
            "created_at": now,
        })
    return rows






def _remember_turn(app_state: AppState, req: ChatRequest, answer_text: str,
                   citations: list[dict] | None = None) -> int:
    """一次问答的收尾写回（**B-9 三层顺序**，r16 §20）。

    顺序（不可交换，`AC-ST-7`/`AC-ST-9`）：

    1. **关系库**：这一轮两条消息写进 `messages` 表（权威全量历史，message_id 幂等）；
    2. **算窗口**：把「当前 Redis 缓冲 + 本轮」按"组"切开，得到**滚出窗口**的旧轮次；
    3. **Milvus 长期记忆**：只写滚出的那些消息（幂等 upsert；失败 ⇒ WARNING + 指标，
       且**本轮不裁剪 Redis**，下一轮自动补写）；
    4. **Redis 窗口**：append 本轮；写长期记忆成功**才**裁剪（日志同 `request_id`
       可证"写入成功早于裁剪完成"）。

    返回值仍是"会话行是否被本用户更新"（0 = 该会话属于别人，仅记录不报错）。
    """
    assert app_state.sessions is not None
    sessions = app_state.sessions
    business = app_state.business
    request_id = current_request_id() or ""
    user_id, role_id, session_id = str(req.user_id), str(req.role_id), str(req.session_id)

    # ① 关系库（历史的权威来源；不依赖 Redis 存活）
    rows: list[dict] = []
    seq: int | None = None
    if business is not None:
        business.upsert_user(user_id)
        seq = business.next_seq(session_id, user_id)
        rows = _turn_rows(app_state, req, answer_text, citations, seq=seq)
        business.append_messages(rows)

    # ② 这一轮之后哪些消息滚出窗口（按"组"对齐；窗口模式才需要）
    turns = int(sessions.turns or 0)
    rolled: list[dict] = []
    new_items = [{"message_id": row["message_id"], "role": row["role"],
                  "content": row["content"], "created_at": float(row["created_at"]),
                  "citations": list(row["citations"]), "turn_index": int(row["turn_index"])}
                 for row in rows]
    if turns:
        if not rows:   # 没有关系库时的兜底：按内容现算 ID，长期记忆仍然可用
            new_items = [{"message_id": message_id_for(user_id, session_id, "user", req.message),
                          "role": "user", "content": req.message, "created_at": time.time(),
                          "citations": [], "turn_index": 0},
                         {"message_id": message_id_for(user_id, session_id, "assistant", answer_text),
                          "role": "assistant", "content": answer_text, "created_at": time.time(),
                          "citations": list(citations or []), "turn_index": 0}]
        # 配对必须做在**整段缓冲**上（老消息来自 Redis、新消息来自本轮），
        # 且 `pair_id` 用**问题内容键**（与消息 ID 口径解耦）：
        # 助手记忆记录的主键取自它 ⇒ "同一个问题重问"只 upsert、不多写一条
        # （见 memory/session.py::with_pair_ids / question_key）。
        _keep, rolled = split_window(
            with_pair_ids(sessions.raw_messages(user_id, role_id, session_id) + new_items,
                          user_id=user_id, session_id=session_id),
            turns)

    # ③ 长期记忆：只写滚出窗口的旧轮次；失败可见且**不裁剪**
    longterm_ok = True
    if rolled and app_state.longterm is not None:
        result = app_state.longterm.remember_messages(user_id, role_id, session_id, rolled)
        longterm_ok = bool(result.get("ok", True))

    # ④ Redis 窗口（append 后按开关决定是否裁剪）
    #    传 `seq`：让窗口条目的 message_id 与关系库**逐字一致**（位置寻址）——
    #    这是 `api/services/session_turns.py::sync_window_into_db`
    #    能对上账、不重复回填的前提。
    sessions.append_turn(user_id, role_id, session_id, req.message, answer_text, citations,
                         seq=seq)
    if turns:
        if rolled and not longterm_ok:
            M.counter("session_trim_skipped_total").inc(reason="longterm_failed",
                                                        collection=app_state.longterm.collection
                                                        if app_state.longterm else "")
            logger.warning("[B-9] request_id=%s 因长期记忆写入失败而**暂不裁剪** Redis："
                           "session=%s（该轮仍留在窗口里，下轮自动补写）", request_id, session_id)
        else:
            remaining = sessions.trim_window(user_id, role_id, session_id)
            M.counter("session_trim_total").inc(status="trimmed")
            logger.info("[B-9] request_id=%s Redis 裁剪完成：session=%s 保留 %d 条"
                        "（窗口 = 最近 %d 组问答）", request_id, session_id, remaining, turns)

    # ⑤ 会话行（标题/归属）—— 与原实现一致：用户改过名的会话不被自动标题覆盖
    updated = 0
    if business is not None:
        updated = business.upsert_session(session_id, user_id, role_id,
                                          title=req.message[:30], user_set_title=False)
    return int(updated or 0)



def _llm_unavailable(exc: BaseException) -> HTTPException:
    """后端全挂：**显式报错**（t120 F1 方案①）——可读原因 + 可重试入口，不给假答案。

    * 响应体只说人话（原因分类 + 这次没有生成回答 + 接下来能做什么），不泄露内部细节；
    * 技术细节（provider/model/异常摘要）进日志与指标 —— 见 ``[LLM-FAIL]`` 行；
    * ``Retry-After`` 与 ``X-LLM-Failure-Reason`` 给前端做重试按钮与埋点用。
    """
    reason = failure_reason(exc)
    logger.warning("[LLM-UNAVAILABLE] reason=%s summary=%s", reason, summarize_exception(exc))
    text = _LLM_REASON_TEXT.get(reason, _LLM_REASON_TEXT["unknown"])
    return HTTPException(
        status_code=503,
        detail=(f"{text}（{reason}）。这次没有生成回答，也没有用演示模型顶替；"
                f"请稍后重试，或让运维检查大模型配置（LLM_PROVIDER / "
                f"OPENAI_COMPAT_BASE_URL / OPENAI_API_KEY）。"),
        headers={"Retry-After": "10", "X-LLM-Failure-Reason": reason},
    )



def cfg_sources(config: RagConfig) -> list[str]:
    return [str(config.knowledge_dir or KNOWLEDGE_DIR)]


app = None  # 供 `uvicorn legal_rag.api.app:create_app --factory` 使用
