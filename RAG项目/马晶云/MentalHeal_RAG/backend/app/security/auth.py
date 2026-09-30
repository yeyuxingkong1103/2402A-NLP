import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.models.chat import User


PBKDF2_ITERATIONS = 310_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        _encode(salt),
        _encode(digest),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            _decode(salt),
            int(iterations),
        )
        return hmac.compare_digest(_encode(digest), expected)
    except (TypeError, ValueError):
        return False


def create_access_token(user: User, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    payload = {
        "sub": user.user_id,
        "role": user.account_role,
        "exp": int(time.time()) + settings.access_token_expire_minutes * 60,
    }
    encoded_payload = _encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _sign(encoded_payload, settings.secret_key)
    return f"mh1.{encoded_payload}.{signature}"


def decode_access_token(token: str, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    try:
        version, encoded_payload, signature = token.split(".", 2)
        if version != "mh1" or not hmac.compare_digest(
            signature, _sign(encoded_payload, settings.secret_key)
        ):
            raise ValueError
        payload = json.loads(_decode(encoded_payload))
        if int(payload["exp"]) <= int(time.time()):
            raise ValueError
        return payload
    except (TypeError, ValueError, KeyError, json.JSONDecodeError, binascii.Error):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已失效，请重新登录",
        ) from None


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db_session),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请先登录",
        )
    payload = decode_access_token(authorization.split(" ", 1)[1].strip())
    user = db.scalar(select(User).where(User.user_id == str(payload["sub"])))
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在或已停用",
        )
    return user


def _encode(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _sign(value: str, secret: str) -> str:
    return _encode(hmac.new(secret.encode("utf-8"), value.encode("ascii"), hashlib.sha256).digest())
