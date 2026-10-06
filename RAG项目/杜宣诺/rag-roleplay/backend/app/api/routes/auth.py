from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.security import create_access_token, hash_password, verify_password
from ...db.base import get_db
from ...db.models import User
from ..deps import get_current_user
from ..schemas import LoginIn, RegisterIn, ok

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
users_router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.post("/register")
async def register(body: RegisterIn, db: AsyncSession = Depends(get_db)):
    exists = await db.scalar(select(User).where(User.username == body.username))
    if exists:
        raise HTTPException(status_code=400, detail="用户名已存在")
    user = User(username=body.username, password_hash=hash_password(body.password))
    db.add(user)
    await db.commit()
    return ok({"id": user.id, "username": user.username})


@router.post("/login")
async def login(body: LoginIn, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(select(User).where(User.username == body.username))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(str(user.id))
    return ok({"token": token, "token_type": "bearer"})


@users_router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return ok({"id": user.id, "username": user.username})
