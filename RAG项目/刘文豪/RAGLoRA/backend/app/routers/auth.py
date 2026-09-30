# -*- coding: utf-8 -*-
"""鉴权路由：注册 / 登录 / 当前用户。"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..core.db import get_db
from ..core.logging import get_logger
from ..core.security import create_token, hash_password, verify_password
from ..deps import get_current_user
from .. import models, schemas

router = APIRouter(prefix="/auth", tags=["鉴权"])
log = get_logger("auth")


@router.post("/register", response_model=schemas.TokenOut, summary="注册")
def register(body: schemas.RegisterIn, db: Session = Depends(get_db)):
    if db.query(models.User).filter_by(username=body.username).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "用户名已存在")

    pwd_hash, salt = hash_password(body.password)
    user = models.User(
        username=body.username,
        password_hash=pwd_hash,
        salt=salt,
        display_name=body.display_name or body.username,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    log.info("新用户注册: %s (id=%s)", user.username, user.id)

    return schemas.TokenOut(
        access_token=create_token(user.id, user.username),
        user=schemas.UserOut.model_validate(user),
    )


@router.post("/login", response_model=schemas.TokenOut, summary="登录")
def login(body: schemas.LoginIn, db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(username=body.username).first()
    # 统一错误文案，避免暴露「用户是否存在」
    if not user or not verify_password(body.password, user.password_hash, user.salt):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "账号已停用")

    return schemas.TokenOut(
        access_token=create_token(user.id, user.username),
        user=schemas.UserOut.model_validate(user),
    )


@router.get("/me", response_model=schemas.UserOut, summary="当前用户")
def me(user: models.User = Depends(get_current_user)):
    return user
