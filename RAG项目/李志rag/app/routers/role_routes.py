from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import admin_user, current_user
from app.database import get_db
from app.models import Role, User
from app.schemas import RoleCreate, RoleResponse

router = APIRouter(prefix="/roles", tags=["角色"])


@router.get("", response_model=list[RoleResponse])
def list_roles(
    _: User = Depends(current_user), database: Session = Depends(get_db)
) -> list[Role]:
    return list(database.scalars(select(Role).where(Role.is_public.is_(True)).order_by(Role.id)))


@router.post("", response_model=RoleResponse, status_code=201)
def create_role(
    payload: RoleCreate,
    _: User = Depends(admin_user),
    database: Session = Depends(get_db),
) -> Role:
    if database.scalar(select(Role).where(Role.name == payload.name)):
        raise HTTPException(status_code=409, detail="角色名称已存在")
    role = Role(**payload.model_dump())
    database.add(role)
    database.commit()
    database.refresh(role)
    return role

