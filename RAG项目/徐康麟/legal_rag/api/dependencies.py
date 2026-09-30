# -*- coding: utf-8 -*-
"""FastAPI 依赖：把 ``AppState`` 与"身份/就绪"闸门从闭包变成可注入的依赖。

为什么需要这个模块
------------------
``api/app.py`` 的 34 条路由原先**全部定义在 `create_app()` 内部**，通过闭包访问
``app_state`` 以及 ``require_engine`` / ``require_identity`` / ``page_user`` 等助手。
要把路由搬到 ``routers/*`` 模块，这些闭包必须变成**显式依赖**。

做法：``create_app`` 里已经有 ``app.state.app_state = app_state``（lifespan 之前），
因此可以在请求期从 ``request.app.state`` 取回它 —— 这正是 FastAPI 依赖注入的标准形态。
``Depends(get_app_state)`` 在**请求期**解析，所以路由定义顺序与装配顺序无关。

⚠️ 这些函数都**只依赖 ``Request``**，不持有模块级全局 —— 因此多个 `create_app`
实例（测试里常见）互不串味。
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request

from ..logging_setup import get_logger
from .auth import ERROR_USER_MISMATCH
from .deps import AppState
from .services.shared import request_token

_LOGGER = get_logger("legal_rag.api.dependencies")

__all__ = [
    "get_app_state", "require_engine", "require_documents",
    "require_identity", "auth_required_only", "page_user", "account_user",
    "require_session_owner",
    "client_key", "set_session_cookie", "clear_session_cookie",
]

#: ``request.app.state`` 上挂 AppState 的属性名。
#: ⚠️ 必须与 ``create_app`` 里的赋值（``app.state.app_state = app_state``）一致；
#: 两处都写死字符串是刻意的 —— 这样类型检查与 grep 都能直接找到。
APP_STATE_ATTR = "app_state"


def get_app_state(request: Request) -> AppState:
    """从 ``request.app.state`` 取回装配好的 :class:`AppState`。

    取不到说明 ``create_app`` 的装配被绕过（理论上不该发生）⇒ 500 而不是静默继续，
    否则后续会以 ``None.engine`` 之类的形式炸在更难定位的地方。
    """
    state = getattr(getattr(request.app, "state", None), APP_STATE_ATTR, None)
    if state is None:
        raise HTTPException(status_code=500, detail="应用状态尚未装配")
    return state


def require_engine(request: Request) -> Any:
    """引擎未就绪 ⇒ 503（而不是 500：这是"还没准备好"，不是"服务出错"）。"""
    state = get_app_state(request)
    if state.engine is None:
        raise HTTPException(status_code=503, detail="引擎尚未就绪")
    return state.engine


def require_documents(request: Request) -> Any:
    """文档服务未就绪 ⇒ 503。"""
    state = get_app_state(request)
    if state.documents is None:
        raise HTTPException(status_code=503, detail="文档服务尚未就绪")
    return state.documents


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
    state = get_app_state(request)
    if state.auth is None:
        raise HTTPException(status_code=503, detail="账号服务尚未就绪")
    decision = state.auth.authorize(request_token(request), claimed_user_id)
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


def page_user(request: Request) -> dict | None:
    """页面层登录态（有效 Cookie/Bearer 才算登录；无效一律当作未登录）。

    ⚠️ 与 API 层的 :func:`require_identity` **语义不同**：这里**永不抛异常**
    （页面守卫不该把整个页面搞成 500），拿不到就按未登录处理。
    """
    state = get_app_state(request)
    if state.auth is None:
        return None
    try:
        return state.auth.resolve(request_token(request))
    except Exception as exc:  # noqa: BLE001 - 页面守卫不该把整个页面搞成 500
        _LOGGER.warning("页面登录态判定失败（按未登录处理）：%s: %s",
                        type(exc).__name__, exc)
        return None


def account_user(request: Request) -> dict | None:
    """偏好接口的身份：**只认令牌**（Cookie 或 Bearer），不看请求体/查询串。

    与 :func:`require_identity` 的差别：账号偏好是**账号级**数据，未登录时没有任何
    合法对象 ⇒ ``AUTH_REQUIRED=false``（默认）下也**必须** 401（`AC-PR-6③`）。
    因此调用方拿到 ``None`` 就该回 401，而不是"继续用请求体里的 user_id"。

    与 :func:`page_user` 的差别：那个**永不抛异常**（页面守卫不该 500），
    这个让异常照常冒泡（API 层要如实报错）。
    """
    state = get_app_state(request)
    if state.auth is None:
        return None
    return state.auth.resolve(request_token(request))


def client_key(request: Request) -> str:
    """限速用的来源标识（纯标准库；不解析 X-Forwarded-For —— 那可以被伪造）。"""
    host = request.client.host if request.client else "unknown"
    return f"{host}|{request.url.path}"


def require_session_owner(row: dict | None, session_id: str, user_id: str) -> dict:
    """会话归属闸门：行不存在或不属于 ``user_id`` → 403 ``user_mismatch``。

    与 `/chat` 的会话归属校验**同源**（都用会话表里的真实 `user_id`，
    即使 `AUTH_REQUIRED` 关闭也拦得住伪造的 user_id），区别只是把结论写成
    错误码 `user_mismatch` 便于前端分支处理 —— **绝不静默覆盖/静默忽略**
    （docs/API.md §4.6）。

    ⚠️ 这是**纯函数**（不需要 ``Request``）：调用方自己先取出会话行再传进来，
    因此它既能当普通助手用，也能被多个路由复用而不产生额外 I/O。
    """
    if row is None or str(row.get("user_id") or "") != user_id:
        _LOGGER.warning("会话归属校验拒绝：session_id=%s 请求 user_id=%s 实际=%s",
                        session_id, user_id,
                        (row or {}).get("user_id") or "(不存在)")
        raise HTTPException(
            status_code=403,
            detail={"error": ERROR_USER_MISMATCH,
                    "message": "该会话不属于当前用户"})
    return row


def set_session_cookie(request: Request, response: Any, token: str) -> None:
    """把会话令牌写进 HttpOnly Cookie（属性取自 ``auth.cookie_attributes()``）。"""
    state = get_app_state(request)
    if state.auth is None:
        return
    response.set_cookie(
        key=state.auth.cookie_name, value=token,
        max_age=state.auth.token_ttl_seconds,
        **state.auth.cookie_attributes())


def clear_session_cookie(request: Request, response: Any) -> None:
    """登出：清掉会话 Cookie（路径必须与写入时一致，否则清不掉）。"""
    state = get_app_state(request)
    if state.auth is None:
        return
    response.delete_cookie(
        key=state.auth.cookie_name,
        path=str(state.auth.cookie_attributes().get("path") or "/"))
