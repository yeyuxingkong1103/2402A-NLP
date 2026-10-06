# -*- coding: utf-8 -*-
"""会话端点：建 / 列 / 读历史 / 重命名 / 删除 —— 从 ``api/app.py`` 拆出。

三条**不可交换**的语义（改动前务必读）：
1. **归属闸门**：``session_id`` 是全局主键，「存在吗」与「是不是你的」都必须判，
   且一律回 **403 user_mismatch**（不区分两种情况 —— 顺带不让它变成探测口）；
   这条与 `/chat` 的会话归属校验同源，见 ``dependencies.require_session_owner``。
2. **全量历史来自关系库**（B-9）：Redis 只是"近 5 组问答"的上下文窗口，
   ``GET /sessions/{id}/messages`` 读的是 `messages` 表，条数不因 Redis 裁剪而减少；
   读之前会把 Redis 窗口里关系库还没有的消息**懒回填**一次（幂等，`AC-ST-12`）。
3. **role_id 一律由服务端从会话行读**：短期记忆的隔离键是
   ``user_id + role_id + session_id``，让客户端传 role_id 等于把隔离键交给对方拼。
"""
from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from ...logging_setup import get_logger
from ...memory.session import message_id_for
from ..auth import ERROR_USER_MISMATCH
from ..concurrency import run_blocking
from ..deps import AppState
from ..dependencies import get_app_state, require_identity, require_session_owner
from ..schemas import (SessionCreateRequest, SessionMessage, SessionMessagesResponse,
                       SessionRenameRequest, SessionResponse)
from ..services.session_turns import register_session, sync_window_into_db
from ..services.shared import load_citations

router = APIRouter()

logger = get_logger("legal_rag.api.routers.sessions")


def _require_business(app_state: AppState):
    """业务数据层未就绪 ⇒ 503（与既有实现一致）。"""
    if app_state.business is None:
        raise HTTPException(status_code=503, detail="业务数据层尚未就绪")
    return app_state.business


def _not_owned() -> HTTPException:
    """竞态下的归属失败（校验通过后这行被别人删了/改了归属）—— 如实拒绝，不假装成功。

    ⚠️ 每次新建实例：异常对象被 raise 时会带 traceback/上下文状态，
    跨请求复用同一实例是隐患。
    """
    return HTTPException(status_code=403,
                         detail={"error": ERROR_USER_MISMATCH,
                                 "message": "该会话不属于当前用户"})


@router.post("/sessions", response_model=SessionResponse)
async def create_session(req: SessionCreateRequest, request: Request,
                         app_state: AppState = Depends(get_app_state)) -> SessionResponse:
    business = _require_business(app_state)
    # 开关打开时：user_id 必须与登录态一致（403 user_mismatch），否则 401
    require_identity(request, req.user_id)
    # t101（修 t96 观察①）：自动 id 原来用**秒级时间戳**拼接 ⇒ 同一秒内两次新建会撞成
    # 同一个 session_id，两个会话被合并（第二次直接登记到第一次那行上）。
    # 现在：毫秒时间戳 + **uuid4 前缀**（16 个十六进制字符）—— 同秒/同毫秒都不会碰撞；
    # 显式传入的 session_id 仍然**优先原样使用**（既有调用方与历史 id 一概不受影响，
    # 无需任何数据迁移）。
    session_id = req.session_id or (
        f"{req.user_id}-{req.role_id}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:16]}")
    limiter = app_state.blocking_limiter()
    # **归属闸门**（F-G）：知道别人的 session_id 也不能借"建会话"覆盖对方的数据
    # （原实现只做身份闸门 → B 能改掉 A 的标题，且响应体谎报 user_id）。
    existing = await run_blocking(business.get_session, session_id, limiter=limiter)
    if existing is not None and str(existing.get("user_id") or "") != req.user_id:
        logger.warning("拒绝登记他人会话：session_id=%s 属于 %s，请求方 %s",
                       session_id, existing.get("user_id"), req.user_id)
        raise _not_owned()
    # 用户手填标题 = 「用户设定」（自动标题此后不得覆盖它，见 store.upsert_session）
    await run_blocking(register_session, app_state, req.user_id, session_id,
                       req.role_id, req.title, bool(req.title.strip()),
                       limiter=limiter)
    # 到这里 session_id 一定属于 req.user_id（新建的，或已确认同主人），响应不再谎报
    return SessionResponse(session_id=session_id, user_id=req.user_id,
                           role_id=req.role_id, title=req.title)


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(request: Request, user_id: str = Query(..., min_length=1),
                        app_state: AppState = Depends(get_app_state)
                        ) -> list[SessionResponse]:
    business = _require_business(app_state)
    # 会话列表也是**他人数据**：开关打开时不允许查别人的 user_id
    require_identity(request, user_id)
    rows = await run_blocking(business.list_sessions, user_id,
                              limiter=app_state.blocking_limiter())
    return [
        SessionResponse(
            session_id=row["session_id"], user_id=row["user_id"],
            role_id=row["role_id"], title=row.get("title") or "",
        )
        for row in rows
    ]


# ---------- 会话：单会话的读取 / 重命名 / 删除 ----------
#
# 归属闸门（三个端点共用）：`session_id` 是**全局主键**（sessions 表），
# 因此"这个 id 存在吗"和"是不是你的"都必须判；不区分这两种情况，一律
# **403 user_mismatch**（顺带不把 session_id 变成探测口）。
# 实现在 `api/dependencies.py::require_session_owner`（纯函数，三处共用）。


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def session_messages(session_id: str, request: Request,
                           user_id: str = Query(..., min_length=1),
                           app_state: AppState = Depends(get_app_state)
                           ) -> SessionMessagesResponse:
    """读取某会话的历史消息（**只读**，供网页左栏点开会话时渲染）。

    * **`role_id` 一律由服务端从 `sessions` 表读**：短期记忆的 key 是
      ``user_id + role_id + session_id``，让客户端传 role_id 等于把隔离键交给
      对方拼（拼错会读到另一个角色的历史，或者读成空）；
    * **全量历史来自关系库**（B-9 / r16 §20.3 改判）：Redis 只是"近 5 组问答"的
      上下文窗口，**不得**再用它当历史来源 —— 因此这里读 `messages` 表，
      条数不因 Redis 裁剪而减少（`AC-ST-4`）；
    * **存量兼容**：读之前把 Redis 窗口里、关系库还没有的消息**懒回填**一次
      （幂等），改造前只写在 Redis 的老会话因此不丢（`AC-ST-12`）；
    * **无历史 → 空列表**（200 + ``messages: []``）：会话存在但还没聊过是正常状态，
      不是 404；
    * 未知会话 / 非本人会话 → **403** ``{"error":"user_mismatch", ...}``；
    * `AUTH_REQUIRED=true` 时无有效令牌 → **401** ``unauthenticated``。
    """
    business = _require_business(app_state)
    require_identity(request, user_id)
    limiter = app_state.blocking_limiter()
    row = require_session_owner(
        await run_blocking(business.get_session, session_id, limiter=limiter),
        session_id, user_id)
    assert app_state.sessions is not None
    await run_blocking(sync_window_into_db, app_state, user_id, row["role_id"],
                       session_id, limiter=limiter)
    stored = await run_blocking(business.list_messages, session_id, user_id,
                                limiter=limiter)
    if stored:
        messages = [SessionMessage(role=str(item.get("role") or ""),
                                   content=str(item.get("content") or ""),
                                   created_at=float(item.get("created_at") or 0.0),
                                   citations=load_citations(item.get("citations")),
                                   message_id=str(item.get("message_id") or ""))
                    for item in stored]
    else:
        # 兜底（正常路径不该走到）：关系库还没有该会话的消息时，仍按窗口返回，
        # 免得把"改造前写入、尚未回填"的历史显示成空。
        history = await run_blocking(app_state.sessions.history, user_id,
                                     row["role_id"], session_id, limiter=limiter)
        messages = [SessionMessage(role=m.role, content=m.content,
                                   created_at=float(m.created_at or 0.0),
                                   citations=list(getattr(m, "citations", None) or []),
                                   message_id=message_id_for(str(user_id), str(session_id),
                                                             m.role, m.content))
                    for m in history]
    return SessionMessagesResponse(
        session_id=session_id, user_id=str(row.get("user_id") or user_id),
        role_id=str(row["role_id"]), title=str(row.get("title") or ""),
        messages=messages,
    )


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
async def rename_session(session_id: str, req: SessionRenameRequest, request: Request,
                         app_state: AppState = Depends(get_app_state)) -> SessionResponse:
    """重命名会话（把 ``title`` **持久化**到 `sessions` 表）。

    * 归属闸门同 `GET /sessions/{id}/messages`：未知/非本人 → 403 ``user_mismatch``；
    * 落库用带 ``user_id`` 条件的 UPDATE（校验漏判也改不到别人的行）；
    * 响应回**落库后的真实值**（紧接着 `GET /sessions` 会读到同一个 title，
      刷新/换进程依然生效 —— 因为它在数据库里，不在进程内存里）。
    """
    business = _require_business(app_state)
    require_identity(request, req.user_id)
    limiter = app_state.blocking_limiter()
    row = require_session_owner(
        await run_blocking(business.get_session, session_id, limiter=limiter),
        session_id, req.user_id)
    updated = await run_blocking(business.rename_session, session_id,
                                 req.title, req.user_id, limiter=limiter)
    if int(updated or 0) == 0:
        # 竞态：校验通过后这行被别人删了/改了归属 —— 如实拒绝，不假装成功
        raise _not_owned()
    fresh = await run_blocking(business.get_session, session_id, limiter=limiter)
    row = fresh or row
    return SessionResponse(session_id=session_id,
                           user_id=str(row.get("user_id") or req.user_id),
                           role_id=str(row["role_id"]),
                           title=str(row.get("title") or ""))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request,
                         user_id: str = Query(..., min_length=1),
                         app_state: AppState = Depends(get_app_state)) -> dict:
    """删除会话：删 `sessions` 行 **并且** 清掉该会话的短期记忆。

    * 历史清理走 ``SessionStore.clear(user_id, role_id, session_id)`` ——
      `role_id` 同样从会话行读（隔离键不由客户端拼）；
    * **不动长期记忆**（Milvus ``legal_rag_memory``）：那是跨会话的用户画像，
      删一个会话不该抹掉它；本端点只做会话级删除；
    * 响应 ``{"deleted": true, "session_id": ..., "role_id": ..., "messages_cleared": N}``；
    * 未知/非本人会话 → 403 ``user_mismatch``（与上面两个端点一致）。
    """
    business = _require_business(app_state)
    require_identity(request, user_id)
    limiter = app_state.blocking_limiter()
    row = require_session_owner(
        await run_blocking(business.get_session, session_id, limiter=limiter),
        session_id, user_id)
    assert app_state.sessions is not None
    # 先清历史再删行：万一删行失败，留下的是"空会话"（自洽、可再删），
    # 而不是"有历史但列表里没有它"的孤儿历史。
    cleared = await run_blocking(app_state.sessions.clear, user_id,
                                 row["role_id"], session_id, limiter=limiter)
    # B-9：关系库里的消息同样按 user_id 删（会话行删掉后不该留下历史孤儿）
    removed_messages = await run_blocking(business.delete_messages, session_id,
                                          user_id, limiter=limiter)
    removed = await run_blocking(business.delete_session, session_id,
                                 user_id, limiter=limiter)
    if int(removed or 0) == 0:
        raise _not_owned()
    logger.info("删除会话 %s（user=%s role=%s）：清历史 %d 条、删消息 %d 条、删行 %d 行",
                session_id, user_id, row["role_id"], int(cleared or 0),
                int(removed_messages or 0), int(removed or 0))
    return {"deleted": True, "session_id": session_id,
            "role_id": str(row["role_id"]),
            "messages_cleared": int(cleared or 0)}
