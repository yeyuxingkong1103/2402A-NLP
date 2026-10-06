from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.security import create_access_token, hash_password, verify_password
from backend.app.models.entities import User
from backend.app.models.schemas import TokenOut, UserCreate, UserLogin


router = APIRouter(prefix="/auth", tags=["认证"])


@router.post("/register", response_model=TokenOut)
def register(payload: UserCreate, db: Session = Depends(get_db)) -> TokenOut:
    exists = db.query(User).filter(User.username == payload.username).first()
    if exists:
        raise HTTPException(status_code=400, detail="用户名已存在")
    user = User(username=payload.username, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user)
    return TokenOut(access_token=token, user_id=user.id, username=user.username)


@router.post("/login", response_model=TokenOut)
def login(payload: UserLogin, db: Session = Depends(get_db)) -> TokenOut:
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(user)
    return TokenOut(access_token=token, user_id=user.id, username=user.username)
