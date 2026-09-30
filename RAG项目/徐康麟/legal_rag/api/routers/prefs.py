# -*- coding: utf-8 -*-
"""账号偏好端点：``GET/PUT/POST/DELETE /prefs`` —— 从 ``api/app.py`` 拆出。

契约要点（`AC-PR-*` / r17 §21）：
* 只有两个业务字段：``theme`` 与 ``role_id``（`AC-PR-1` 的字段白名单）；
* 取值一律来自**登录态**，请求体里的 ``user_id`` 一律被忽略（`AC-PR-5④`）；
* ``theme=null`` = **显式跟随系统**，与"不传 theme"**不同**（用 ``model_fields_set`` 区分）；
* ``role_id`` 必须是角色库成员，未知 → **400 且不改动已有偏好**（校验在写库之前）；
* 未登录 → **401**（即使 ``AUTH_REQUIRED=false``，账号级数据没有合法对象，`AC-PR-6③`）。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ...roles import ROLE_LIBRARY
from ..auth import ERROR_UNAUTHENTICATED
from ..concurrency import run_blocking
from ..deps import AppState
from ..schemas import PrefsResponse, PrefsUpdate
from ..dependencies import account_user, get_app_state
from ..services.prefs import prefs_payload
from ...logging_setup import get_logger

router = APIRouter()

logger = get_logger("legal_rag.api.routers.prefs")

#: 未登录时的统一响应（文案与既有实现逐字一致）。
#:
#: ⚠️ 写成**函数**而不是模块级常量：异常对象被 raise 时会带上 traceback/上下文状态，
#: 跨请求复用同一个实例是隐患（并发下可能串味）。既有实现是每处 `raise HTTPException(...)`
#: 新建，这里保持同一语义。
def _unauthenticated() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": ERROR_UNAUTHENTICATED,
                "message": "未登录：账号偏好只对登录账号开放"})


def _require_business(app_state: AppState):
    """业务数据层未就绪 ⇒ 503（与既有实现一致，不是 500）。"""
    if app_state.business is None:
        raise HTTPException(status_code=503, detail="业务数据层尚未就绪")
    return app_state.business


@router.get("/prefs", response_model=PrefsResponse)
async def get_prefs(request: Request,
                    app_state: AppState = Depends(get_app_state)) -> PrefsResponse:
    """读**当前登录账号**的主题与角色偏好（r17 §21）。

    * 取值一律来自**登录态**（`lr_session` / Bearer）；请求里带 `user_id` **不生效**；
    * 未登录 → **401** ``unauthenticated``（不返回任何用户数据，`AC-PR-6③`）；
    * 未设置过 → ``theme=null``（跟随系统）、``role_id=lawyer``（默认角色，`AC-PR-3②`）；
    * 字段集合只有 `theme` + `role_id`（+ `updated_at`/`role_id_is_default` 元数据），
      不含行为历史 / 设备指纹 / 凭据（`AC-PR-1`）。
    """
    business = _require_business(app_state)
    user = await run_blocking(account_user, request,
                              limiter=app_state.blocking_limiter())
    if user is None:
        raise _unauthenticated()
    user_id = str(user.get("user_id") or "")
    row = await run_blocking(business.get_prefs, user_id,
                             limiter=app_state.blocking_limiter())
    return PrefsResponse(**prefs_payload(user_id, row))


@router.put("/prefs", response_model=PrefsResponse)
@router.post("/prefs", response_model=PrefsResponse)
async def write_prefs(req: PrefsUpdate, request: Request,
                      app_state: AppState = Depends(get_app_state)) -> PrefsResponse:
    """写账号偏好（`PUT` 与 `POST` 同一处理器；只接受 `theme` / `role_id`）。

    * `theme ∈ {dark, light, null}`；**`null` = 显式跟随系统**，与不传该字段不同；
    * `role_id` 必须是角色库里已有的角色（未知 → **400**，`AC-PR-2/3`）；
    * 未登录 → **401**；请求体里的 `user_id` 一律**被登录态覆盖**（`AC-PR-5④`）；
    * 非法值 → **4xx 且不改动已有偏好**（校验在写库之前完成）。
    """
    business = _require_business(app_state)
    user = await run_blocking(account_user, request,
                              limiter=app_state.blocking_limiter())
    if user is None:
        raise _unauthenticated()
    provided = set(getattr(req, "model_fields_set", set()) or set())
    changes: dict = {}
    if "theme" in provided:
        theme = req.theme
        if theme is not None and str(theme) not in ("dark", "light"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "invalid_theme",
                        "message": "theme 只能是 dark、light 或 null（null = 跟随系统）"})
        changes["theme"] = None if theme is None else str(theme)
    if "role_id" in provided:
        role_id = "" if req.role_id is None else str(req.role_id).strip()
        # ⚠️ 用 `ROLE_LIBRARY` 判成员，**不能**用 `get_role()`：后者对未知 id 会
        # **合成**一个兜底角色（既有语义，供上游容错），拿它做校验等于永不报错。
        if role_id and role_id not in ROLE_LIBRARY:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "invalid_role_id",
                        "message": f"未知角色：{role_id}（可用角色见 GET /roles）"})
        changes["role_id"] = role_id
    user_id = str(user.get("user_id") or "")
    row = await run_blocking(business.upsert_prefs, user_id, **changes,
                             limiter=app_state.blocking_limiter())
    logger.info("账号偏好已更新：user=%s 变更字段=%s", user_id, sorted(changes))
    return PrefsResponse(**prefs_payload(user_id, row))


@router.delete("/prefs", response_model=PrefsResponse)
async def reset_prefs(request: Request,
                      app_state: AppState = Depends(get_app_state)) -> PrefsResponse:
    """清空账号偏好（回到"未设置"：`theme=null` 跟随系统、`role_id=lawyer`）。

    对应 `AC-PR-3③` 的"显式清空/重置偏好后回落默认角色"。
    """
    business = _require_business(app_state)
    user = await run_blocking(account_user, request,
                              limiter=app_state.blocking_limiter())
    if user is None:
        raise _unauthenticated()
    user_id = str(user.get("user_id") or "")
    await run_blocking(business.delete_prefs, user_id,
                       limiter=app_state.blocking_limiter())
    logger.info("账号偏好已清空：user=%s", user_id)
    return PrefsResponse(**prefs_payload(user_id, None))
