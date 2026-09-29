# -*- coding: utf-8 -*-
"""FastAPI 依赖：当前登录用户。"""
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from .core.db import get_db
from .core.security import decode_token
from . import models


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> models.User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "缺少 Bearer Token")

    payload = decode_token(authorization.split(" ", 1)[1].strip())
    if not payload:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token 无效或已过期")

    user = db.get(models.User, int(payload["sub"]))
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户不存在或已停用")
    return user
