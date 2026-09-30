"""会话管理接口。

`docs/API_DOCUMENTATION.md` 第 6 节规定了这套接口，但代码里一直是空白——
`app/memory/` 下的 SessionManager / RedisMemory 早就写完并有单测，
只是从来没接到 HTTP 层。这个文件补的就是那一层。

所有接口在 Redis 或 MySQL 不可用时返回 503，而不是 500：
这是「依赖没起」而不是「代码崩了」，两者对排查的意义完全不同。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, status

from app.api.v1 import access, deps
from app.api.v1.session_schemas import (
    HistoryMessage,
    RoleSwitchRequest,
    RoleSwitchResponse,
    SessionClearResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionDetailResponse,
    SessionHistoryResponse,
    SessionArchiveResponse,
    SessionRoleInfo,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_UNAVAILABLE = (
    "会话功能不可用：需要 Redis 和 MySQL 同时就绪。"
    "请检查 REDIS_URL / DATABASE_URL 配置以及服务是否启动。"
)


def _require_session_manager():
    manager = deps.get_session_manager()
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE
        )
    return manager


def _require_redis_memory():
    memory = deps.get_redis_memory()
    if memory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="短期记忆不可用：Redis 未就绪。",
        )
    return memory


@router.post(
    "/sessions", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED
)
async def create_session(request: SessionCreateRequest) -> SessionCreateResponse:
    """创建会话。"""
    manager = _require_session_manager()

    role_id = request.role_id
    if role_id is None:
        role_id = deps.resolve_role_id(request.role_key)
        if role_id is None:
            # sessions.role_id 有外键指向 roles(id)，塞个不存在的 id 会在
            # INSERT 时报 1452，错误信息对调用方毫无意义。这里提前挡掉并说清原因。
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"角色不存在或未启用: {request.role_key}",
            )

    user_id = access.user_id(request.user_id)
    tenant_id = access.tenant_id(request.tenant_id)

    try:
        session_id = manager.create_session(
            user_id=user_id,
            tenant_id=tenant_id,
            role_id=role_id,
            channel=request.channel,
            metadata=request.metadata,
        )
    except Exception as exc:
        logger.exception("创建会话失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="创建会话失败，请稍后重试",
        ) from exc

    return SessionCreateResponse(
        session_id=session_id,
        user_id=user_id,
        tenant_id=tenant_id,
        role_id=role_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str) -> SessionDetailResponse:
    """查询会话详情。"""
    manager = _require_session_manager()

    meta = access.owned_session(manager, session_id)

    role_info = SessionRoleInfo(role_id=_as_int(meta.get("role_id")))
    client = deps.get_mysql_client()
    if client is not None and role_info.role_id is not None:
        row = client.fetchone(
            "SELECT role_key, persona_name FROM roles WHERE id = %s", (role_info.role_id,)
        )
        if row:
            role_info.role_key = row.get("role_key")
            role_info.persona_name = row.get("persona_name")

    return SessionDetailResponse(
        session_id=session_id,
        user_id=_as_int(meta.get("user_id")),
        tenant_id=_as_int(meta.get("tenant_id")),
        role=role_info,
        conversation_count=_as_int(meta.get("message_count")) or 0,
        created_at=meta.get("created_at") or meta.get("started_at"),
        last_active_at=meta.get("updated_at") or meta.get("last_active_at"),
        status=str(meta.get("status", "active")),
    )


@router.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="返回的消息条数"),
) -> SessionHistoryResponse:
    """查询会话历史。

    ``limit`` 的单位是**消息条数**，不是轮数。RedisMemory.get_messages 收的是
    轮数（1 轮 = user + assistant 两条），所以这里要除以 2 再向上取整，
    否则 limit=10 会拿回 20 条。
    """
    manager = _require_session_manager()
    access.owned_session(manager, session_id)
    memory = _require_redis_memory()

    max_turns = max(1, (limit + 1) // 2)
    messages = memory.get_messages(session_id, max_turns=max_turns)

    return SessionHistoryResponse(
        session_id=session_id,
        history=[
            HistoryMessage(
                role=str(m.get("role", "")),
                content=str(m.get("content", "")),
                timestamp=m.get("timestamp"),
            )
            for m in messages[-limit:]
        ],
    )


@router.post("/sessions/{session_id}/clear", response_model=SessionClearResponse)
async def clear_session(session_id: str) -> SessionClearResponse:
    """清空会话历史（保留会话本身）。"""
    manager = _require_session_manager()
    access.owned_session(manager, session_id)
    memory = _require_redis_memory()

    # 只删消息列表，不动 meta —— 「清空历史」和「删除会话」是两件事，
    # RedisMemory.delete_session 会把 meta 一起删掉，那样 session_id 就失效了。
    memory.client.delete(f"session:{session_id}:messages")

    manager = deps.get_session_manager()
    if manager is not None:
        try:
            manager.update_session(session_id, {"message_count": 0})
        except Exception as exc:
            logger.warning("清空会话后重置计数失败: %s", exc)

    return SessionClearResponse(message="会话历史已清空", session_id=session_id)


@router.post(
    "/sessions/{session_id}/end",
    response_model=SessionArchiveResponse,
)
async def end_session(session_id: str) -> SessionArchiveResponse:
    """结束会话并把跨会话摘要写入长期记忆。"""
    manager = _require_session_manager()
    meta = access.owned_session(manager, session_id)

    archived = False
    memory_manager = deps.get_memory_manager()
    user_id = _as_int(meta.get("user_id"))
    tenant_id = _as_int(meta.get("tenant_id"))
    if memory_manager is not None and user_id is not None and tenant_id is not None:
        try:
            archived = bool(memory_manager.archive_session(session_id, user_id, tenant_id))
        except Exception as exc:
            logger.warning("会话长期记忆归档失败: %s", exc)

    manager.end_session(session_id)
    return SessionArchiveResponse(
        session_id=session_id,
        status="archived",
        archived=archived,
        message=("会话已结束，长期摘要已保存" if archived else "会话已结束"),
    )


@router.put("/sessions/{session_id}/role", response_model=RoleSwitchResponse)
async def switch_role(session_id: str, request: RoleSwitchRequest) -> RoleSwitchResponse:
    """切换会话角色，并记入 role_switch_logs。"""
    manager = _require_session_manager()
    client = deps.get_mysql_client()

    meta = access.owned_session(manager, session_id)

    new_role_id = deps.resolve_role_id(request.role)
    if new_role_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"角色不存在或未启用: {request.role}",
        )

    previous_role_id = _as_int(meta.get("role_id"))
    previous_role_key = None
    if client is not None and previous_role_id is not None:
        row = client.fetchone("SELECT role_key FROM roles WHERE id = %s", (previous_role_id,))
        if row:
            previous_role_key = row.get("role_key")

    if not manager.update_session(session_id, {"role_id": new_role_id}):
        # update_session 里 Redis 先写、MySQL 后写，整体被 try 包住，失败只 log 并
        # 返回 False。不接这个返回值的话：MySQL 没落库、Redis 已经改了、接口还回
        # 200「切换成功」，等 Redis meta 过期回落 MySQL，角色会自己弹回旧值。
        # 宁可报错让调用方重试，也不能谎报成功。
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="角色切换失败：会话状态未能持久化，请重试。",
        )

    if client is not None:
        try:
            client.execute(
                """INSERT INTO role_switch_logs
                   (session_id, user_id, from_role_id, to_role_id, reason, switch_type)
                   VALUES (%s, %s, %s, %s, %s, 'manual')""",
                (
                    session_id,
                    _as_int(meta.get("user_id")) or deps.default_user_id(),
                    previous_role_id,
                    new_role_id,
                    request.reason,
                ),
            )
        except Exception as exc:
            # 日志写失败不该让切换本身失败——切换已经生效了。
            logger.warning("写入角色切换日志失败: %s", exc)

    return RoleSwitchResponse(
        session_id=session_id,
        previous_role=previous_role_key,
        current_role=request.role,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def _as_int(value) -> int | None:
    """Redis 里所有值都是字符串，MySQL 回来的是 int，统一成 int。"""
    if value is None or value == "" or value == "None":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
