# -*- coding: utf-8 -*-
"""账号端点：注册 / 登录 / 当前用户 / 登出 / 找回密码 / 用户名查重 —— 从 ``api/app.py`` 拆出。

闭包依赖只有 4 个，且都在 ``api/dependencies.py`` 里：
``app_state``（Depends 注入）、``set_session_cookie`` / ``clear_session_cookie``（显式传 request）、
``client_key``（显式传 request）。

⚠️ 本模块的三条**安全契约**（改动前务必读）：
1. **注册响应只含 user_id / username / recovery_code**，不含哈希/盐/令牌（`AC-AU-31`）；
   恢复码**只在这一个响应里出现一次**，此后任何接口不再回显（`AC-AU-49/52`）。
2. **登录失败不可区分**：用户名不存在与密码错误回**逐字相同**的 401
   （状态码/body/文案一致，且"不存在"分支也跑一次 scrypt 抹平耗时差）；
   仅"用户名格式非法"回 400（只取决于本次请求，不泄露库里有谁）。
3. **找回密码的三种失败也逐字相同**（`用户名、恢复码或新密码不正确`）且**不消耗**恢复码；
   成功则旧令牌全部失效 + 恢复码轮换 + 清 Cookie。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse

from ...logging_setup import get_logger
from ..auth import (ERROR_INVALID_USERNAME, ERROR_RESET_FIELDS, ERROR_UNAUTHENTICATED,
                    ERROR_WEAK_PASSWORD)
from ..concurrency import run_blocking
from ..deps import AppState
from ..dependencies import clear_session_cookie, client_key, get_app_state, set_session_cookie
from ..schemas import LoginRequest, RegisterRequest, ResetRequest
from ..services.shared import request_token

router = APIRouter()

logger = get_logger("legal_rag.api.routers.auth")


def _require_auth_service(app_state: AppState):
    """账号服务未就绪 ⇒ 503（与既有实现一致）。"""
    if app_state.auth is None:
        raise HTTPException(status_code=503, detail="账号服务尚未就绪")
    return app_state.auth


@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def auth_register(req: RegisterRequest, request: Request,
                        app_state: AppState = Depends(get_app_state)) -> JSONResponse:
    """注册（**只有 username + password**，没有手机号/邮箱/验证码）。

    成功 **201** + ``Set-Cookie: lr_session=...; HttpOnly; SameSite=Lax; Path=/``；
    用户名重复 **409** ``username_taken``（唯一性由数据库 UNIQUE 约束兜底，
    **不是**"先查再插" —— 并发注册下后者会漏）；
    格式/长度不合规 **400** ``invalid_username`` / ``weak_password``（带 `field`）。

    响应体**只**含 ``user_id``/``username``：不含哈希、不含盐、不含令牌（`AC-AU-31`）。
    """
    auth = _require_auth_service(app_state)
    outcome = await run_blocking(auth.register, req.username, req.password,
                                 limiter=app_state.blocking_limiter())
    if not outcome.ok:
        error = outcome.error
        assert error is not None
        code = (status.HTTP_409_CONFLICT
                if error.error == "username_taken"
                else status.HTTP_400_BAD_REQUEST)
        raise HTTPException(status_code=code, detail=error.to_dict())
    user = outcome.user
    assert user is not None
    # ⚠️ 必须直接构造带 201 的响应对象再挂 Cookie：先在临时响应上 set_cookie 再丢掉，
    #    浏览器会「注册成功但没登录」（Cookie 随临时对象一起被丢掉了）。
    response = JSONResponse(status_code=status.HTTP_201_CREATED, content={
        "user_id": user["user_id"], "username": user["username"],
        "auth_required": auth.require_auth,
        # 一次性恢复码：**只在这一个响应里出现**（`AC-AU-49/52`）。
        # 此后的 GET/POST 认证接口一律不再回显它（明文不落库、不进日志）。
        "recovery_code": outcome.recovery_code or "",
    })
    # 注册成功即登录（少一步手动登录，主流产品的做法），令牌同样只走 Cookie
    if outcome.token:
        set_session_cookie(request, response, outcome.token)
    return response


@router.post("/auth/login")
async def auth_login(req: LoginRequest, request: Request,
                     app_state: AppState = Depends(get_app_state)) -> JSONResponse:
    """登录。

    成功 **200** + Cookie；用户名不存在与密码错误**完全一致**地回
    **401** ``{"error":"invalid_credentials","message":"用户名或密码错误"}``
    （状态码、body、文案逐字相同，且"不存在"分支也跑一次 scrypt 抹平耗时差，
    见 `api/auth.py` 模块头 §5）；用户名**格式非法**回 400（与凭据错误区分开，
    这只取决于本次请求本身，不泄露"库里有谁"）。
    """
    auth = _require_auth_service(app_state)
    outcome = await run_blocking(auth.login, req.username, req.password,
                                 limiter=app_state.blocking_limiter())
    if not outcome.ok:
        error = outcome.error
        assert error is not None
        code = (status.HTTP_400_BAD_REQUEST
                if error.error == "invalid_username" else status.HTTP_401_UNAUTHORIZED)
        raise HTTPException(status_code=code, detail=error.to_dict())
    user = outcome.user
    assert user is not None and outcome.token
    response = JSONResponse(content={
        "user_id": user["user_id"], "username": user["username"],
        "expires_at": outcome.expires_at,
        "auth_required": auth.require_auth,
    })
    set_session_cookie(request, response, outcome.token)
    return response


@router.get("/auth/me")
async def auth_me(request: Request,
                  app_state: AppState = Depends(get_app_state)) -> JSONResponse:
    """当前登录用户（凭 HttpOnly Cookie；脚本可用 ``Authorization: Bearer``）。

    无令牌 / 伪造 / 已登出 / 已过期 → **401** ``unauthenticated``（不区分原因）。
    响应体**不含**密码哈希、不不含令牌明文。
    """
    auth = _require_auth_service(app_state)

    user = await run_blocking(auth.current_user, request_token(request),
                              limiter=app_state.blocking_limiter())
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail={"error": ERROR_UNAUTHENTICATED,
                                    "message": "未登录或登录已过期"})
    return JSONResponse(content={
        "user_id": user["user_id"], "username": user["username"],
        "expires_at": user["expires_at"],
        "auth_required": auth.require_auth,
    })


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
@router.get("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def auth_logout(request: Request,
                      app_state: AppState = Depends(get_app_state)) -> Response:
    """登出：服务端置 ``revoked_at``（**该令牌立即失效**）并清 Cookie，返回 **204**。

    同时接受 POST 与 GET：规范里只写了 POST，但"点了退出却没退出"是最容易被
    踩的反人类坑，多挂一个 GET 让任何调用姿势都不会静默失败（响应都是 204）。
    """
    auth = _require_auth_service(app_state)

    await run_blocking(auth.revoke, request_token(request),
                       limiter=app_state.blocking_limiter())
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    clear_session_cookie(request, response)
    return response


@router.post("/auth/reset")
async def auth_reset(request: Request, req: ResetRequest | None = None,
                     app_state: AppState = Depends(get_app_state)) -> JSONResponse:
    """找回密码（一次性恢复码，`AC-AU-53..56` / `AC-RC-1..6`）。

    * 请求体只认 ``username`` / ``recovery_code`` / ``new_password``（`AC-AU-56`）；
    * 缺任意一个 → **400** 且无副作用（`AC-RC-3`）；用户名格式非法 / 新密码不合规 → 400；
      连**整个 body 都没有**也是 400（`req` 声明成可选并补一个空模型，否则 FastAPI 会对
      「无 body」先回 422 —— 同一个"缺字段"在规范里只有一个答案：400）；
    * 「用户名不存在 / 恢复码错 / 不匹配」→ **401** 且三者的 `error` 与文案**逐字相同**
      （`用户名、恢复码或新密码不正确`），**不泄露存在性**，且**不消耗**恢复码；
    * 成功 → 旧令牌**全部失效** + 恢复码**轮换**（新码随本次响应只回一次）+
      清掉浏览器里的旧 Cookie（页面回 `/login` 让用户重新登录，`AC-RC-5`）。
    """
    auth = _require_auth_service(app_state)
    if req is None:
        req = ResetRequest()
    outcome = await run_blocking(auth.reset_password, req.username,
                                 req.recovery_code, req.new_password,
                                 limiter=app_state.blocking_limiter())
    if not outcome.ok:
        error = outcome.error
        assert error is not None
        code = (status.HTTP_400_BAD_REQUEST
                if error.error in (ERROR_INVALID_USERNAME, ERROR_WEAK_PASSWORD,
                                   ERROR_RESET_FIELDS)
                else status.HTTP_401_UNAUTHORIZED)
        raise HTTPException(status_code=code, detail=error.to_dict())
    user = outcome.user
    assert user is not None
    response = JSONResponse(status_code=status.HTTP_200_OK, content={
        "user_id": user["user_id"], "username": user["username"],
        # 轮换后的**新**恢复码：同样只在这里出现一次（`AC-RC-6`）
        "recovery_code": outcome.recovery_code or "",
        "message": "密码已重置，请重新登录",
    })
    # 旧令牌已全部失效：顺手清掉浏览器里的 Cookie，避免"带着废令牌进 /ui 又被弹回"
    clear_session_cookie(request, response)
    return response


@router.get("/auth/username-available")
@router.get("/auth/check-username")
async def auth_username_available(request: Request, username: str = Query(...),
                                  app_state: AppState = Depends(get_app_state)) -> JSONResponse:
    """用户名查重（前端预检；**服务端注册路径仍会再兜一次**，`AC-AU-15`）。

    两个路径都指向本处理器：``/auth/username-available``（UX 规范 §3.9 的名字，
    前端按它调用）与 ``/auth/check-username``（任务书里的名字），**都可用**。

    * ``200 {"username": <规范化后>, "available": true|false}``
      —— 已占用也回 200（`available` 才携带信息）；
    * ``400 {"error":"invalid_username","message":...}`` 格式非法；
    * ``429 {"error":"rate_limited",...}`` 超限（默认 30 次/60 秒/IP，`AC-AU-36`）。

    只回"能不能用"，不泄露其它任何信息（不暴露账号创建时间、不暴露他人的
    `user_id` 之外的字段）—— 但要如实说明：**"已占用"本身就等于"这个用户名存在"**，
    查重接口必然泄露存在性，这是与登录"不可区分"的固有张力，处置是限速而不是假装。
    """
    auth = _require_auth_service(app_state)
    result = await run_blocking(auth.check_username, username,
                                client_key=client_key(request),
                                limiter=app_state.blocking_limiter())
    headers = {}
    if result["status"] == status.HTTP_429_TOO_MANY_REQUESTS:
        headers["Retry-After"] = str(result.get("retry_after") or 1)
    return JSONResponse(status_code=int(result["status"]), content=result["body"],
                        headers=headers)
