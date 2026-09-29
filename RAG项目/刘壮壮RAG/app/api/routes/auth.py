from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.schemas.auth import LoginRequest, Token
from app.schemas.user import UserCreate, UserRead
from app.services import user_service

router = APIRouter()


@router.post("/auth/register", response_model=UserRead, status_code=201)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    try:
        return await user_service.create_user(db, payload.username, payload.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/auth/login", response_model=Token)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await user_service.authenticate_user(
        db, payload.username, payload.password
    )
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return Token(access_token=user_service.issue_token(user.id))


@router.get("/auth/me", response_model=UserRead)
async def me(current_user=Depends(get_current_user)):
    return current_user
