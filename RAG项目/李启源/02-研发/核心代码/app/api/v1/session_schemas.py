"""会话管理接口的 schema。

字段按 `docs/API_DOCUMENTATION.md` 第 6 节的约定来，尽量不自创命名，
免得文档和实现再次分家。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    user_id: int | None = Field(default=None, ge=1, description="用户ID，必须与认证身份一致")
    tenant_id: int | None = Field(default=None, ge=1, description="租户ID，必须与认证身份一致")
    role_id: int | None = Field(default=None, ge=1, description="角色ID，与 role_key 二选一")
    role_key: str = Field(
        default="customer_service", min_length=1, max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$", description="角色标识，role_id 未传时按它解析"
    )
    channel: str = Field(
        default="web", pattern=r"^(web|mobile|api|wechat)$", description="会话渠道"
    )
    metadata: dict[str, Any] | None = Field(default=None, description="附加元数据")


class SessionCreateResponse(BaseModel):
    session_id: str
    user_id: int
    tenant_id: int
    role_id: int
    created_at: str
    status: str = "active"


class SessionRoleInfo(BaseModel):
    role_id: int | None = None
    role_key: str | None = None
    persona_name: str | None = None


class SessionDetailResponse(BaseModel):
    session_id: str
    user_id: int | None = None
    tenant_id: int | None = None
    role: SessionRoleInfo = SessionRoleInfo()
    conversation_count: int = 0
    created_at: str | None = None
    last_active_at: str | None = None
    status: str = "active"


class HistoryMessage(BaseModel):
    role: str
    content: str
    timestamp: str | None = None


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: list[HistoryMessage] = []


class SessionClearResponse(BaseModel):
    message: str
    session_id: str


class SessionArchiveResponse(BaseModel):
    session_id: str
    status: str
    archived: bool = False
    message: str


class RoleSwitchRequest(BaseModel):
    role: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$", description="目标角色标识")
    reason: str | None = Field(default=None, max_length=500, description="切换原因")


class RoleSwitchResponse(BaseModel):
    session_id: str
    previous_role: str | None = None
    current_role: str
    updated_at: str
