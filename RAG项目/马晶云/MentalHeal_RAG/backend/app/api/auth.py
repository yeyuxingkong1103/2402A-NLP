from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.models.chat import ChatMessage, ChatSession, User
from app.schemas.auth import AuthResponse, LoginRequest, RegisterRequest, UserResponse
from app.security.auth import create_access_token, get_current_user, hash_password, verify_password

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def serialize_user(user: User) -> UserResponse:
    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        display_name=user.display_name,
        account_role=user.account_role,
    )


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, db: Session = Depends(get_db_session)) -> AuthResponse:
    username = request.username.strip()
    if db.scalar(select(User).where(User.username == username)):
        raise HTTPException(status_code=409, detail="用户名已存在")
    had_users = db.scalar(select(User.id).limit(1)) is not None
    now = datetime.utcnow()
    user = User(
        user_id=str(uuid4()),
        username=username,
        password_hash=hash_password(request.password),
        display_name=request.display_name.strip(),
        account_role="admin" if not had_users else "user",
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    db.add(user)
    try:
        db.flush()
        if not had_users:
            db.execute(
                update(ChatSession)
                .where(ChatSession.user_id == "anonymous")
                .values(user_id=user.user_id)
            )
            db.execute(
                update(ChatMessage)
                .where(ChatMessage.user_id == "anonymous")
                .values(user_id=user.user_id)
            )
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="用户名已存在") from None
    return AuthResponse(access_token=create_access_token(user), user=serialize_user(user))


@router.post("/login", response_model=AuthResponse)
def login(request: LoginRequest, db: Session = Depends(get_db_session)) -> AuthResponse:
    user = db.scalar(select(User).where(User.username == request.username.strip()))
    if not user or not user.is_active or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    user.updated_at = datetime.utcnow()
    db.commit()
    return AuthResponse(access_token=create_access_token(user), user=serialize_user(user))


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return serialize_user(current_user)


@router.post("/logout")
def logout(current_user: User = Depends(get_current_user)) -> dict[str, str]:
    return {"message": "已退出登录"}
