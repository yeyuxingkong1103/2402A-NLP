from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.models.chat import AIRole, User
from app.schemas.roles import (
    AIRoleResponse,
    RoleCreateRequest,
    RoleListResponse,
    RoleUpdateRequest,
)
from app.security.auth import get_current_user

router = APIRouter(prefix="/api/v1/roles", tags=["roles"])

DEFAULT_ROLES = (
    {
        "role_id": "mental-health",
        "name": "心理健康助手",
        "description": "通用心理健康科普与陪伴",
        "system_instruction": "你是温和、稳妥的通用心理健康科普与陪伴助手。",
    },
    {
        "role_id": "stress-management",
        "name": "压力管理助手",
        "description": "帮助识别压力并练习可执行的调节方法",
        "system_instruction": "你专注于压力识别、情绪调节和可执行的日常减压方法。",
    },
    {
        "role_id": "youth-companion",
        "name": "青少年陪伴助手",
        "description": "面向青少年的安全、友善心理支持",
        "system_instruction": "你面向青少年提供安全、尊重、易懂的心理健康陪伴。",
    },
)


def ensure_default_roles(db: Session) -> None:
    changed = False
    for item in DEFAULT_ROLES:
        role = db.scalar(select(AIRole).where(AIRole.role_id == item["role_id"]))
        if role:
            continue
        now = datetime.utcnow()
        db.add(AIRole(**item, is_active=True, created_at=now, updated_at=now))
        changed = True
    if changed:
        db.commit()


def serialize_role(role: AIRole) -> AIRoleResponse:
    return AIRoleResponse(
        role_id=role.role_id,
        name=role.name,
        description=role.description,
        is_active=role.is_active,
    )


def require_admin(user: User) -> None:
    if user.account_role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")


@router.get("", response_model=RoleListResponse)
def list_roles(
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> RoleListResponse:
    ensure_default_roles(db)
    roles = db.scalars(
        select(AIRole).where(AIRole.is_active.is_(True)).order_by(AIRole.id.asc())
    ).all()
    return RoleListResponse(roles=[serialize_role(role) for role in roles])


@router.post("", response_model=AIRoleResponse, status_code=status.HTTP_201_CREATED)
def create_role(
    request: RoleCreateRequest,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> AIRoleResponse:
    require_admin(current_user)
    if db.scalar(select(AIRole).where(AIRole.role_id == request.role_id)):
        raise HTTPException(status_code=409, detail="角色标识已存在")
    now = datetime.utcnow()
    role = AIRole(**request.model_dump(), is_active=True, created_at=now, updated_at=now)
    db.add(role)
    db.commit()
    db.refresh(role)
    return serialize_role(role)


@router.patch("/{role_id}", response_model=AIRoleResponse)
def update_role(
    role_id: str,
    request: RoleUpdateRequest,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> AIRoleResponse:
    require_admin(current_user)
    role = db.scalar(select(AIRole).where(AIRole.role_id == role_id))
    if not role:
        raise HTTPException(status_code=404, detail="角色不存在")
    for key, value in request.model_dump(exclude_unset=True).items():
        setattr(role, key, value)
    role.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(role)
    return serialize_role(role)
