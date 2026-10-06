# -*- coding: utf-8 -*-
"""【认证鉴权 · auth.py】注册登录安全：密码 PBKDF2 加盐哈希存储 + HMAC 无状态 token 签发/校验。"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional

from sqlalchemy.orm import Session

from config import SECRET_KEY
from database import User


def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 加盐哈希，存储格式：salt$digest。"""
    salt = secrets.token_hex(16)  # 每个用户独立随机盐，防彩虹表
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    """用同样算法重算摘要并做恒定时间比较（防时序攻击）。"""
    try:
        salt, digest = stored.split("$", 1)
    except ValueError:
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000).hex()
    return hmac.compare_digest(check, digest)


def create_user(db: Session, username: str, password: str) -> User:
    """新建用户：密码只存哈希，不存明文。"""
    user = User(username=username.strip(), password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate(db: Session, username: str, password: str) -> Optional[User]:
    """登录校验：查用户 → 验密码，成功返回用户对象，失败返回 None。"""
    user = db.query(User).filter(User.username == username.strip()).first()
    if user and verify_password(password, user.password_hash):
        return user
    return None


def make_token(user_id: int) -> str:
    """签发无状态 token：user_id:随机数:HMAC签名（服务端无需存会话表）。"""
    raw = f"{user_id}:{secrets.token_hex(16)}"  # 随机数保证同一用户每次 token 不同
    sig = hmac.new(SECRET_KEY.encode(), raw.encode(), hashlib.sha256).hexdigest()
    return f"{raw}:{sig}"


def parse_token(token: str) -> Optional[int]:
    """校验 token 签名，通过返回 user_id，否则返回 None。"""
    try:
        user_id, nonce, sig = token.split(":")
        raw = f"{user_id}:{nonce}"
        expect = hmac.new(SECRET_KEY.encode(), raw.encode(), hashlib.sha256).hexdigest()
        if hmac.compare_digest(expect, sig):  # 恒定时间比较防篡改
            return int(user_id)
    except Exception:
        return None
    return None
