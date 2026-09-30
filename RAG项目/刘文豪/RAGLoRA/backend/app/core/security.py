# -*- coding: utf-8 -*-
"""鉴权原语：密码哈希 + JWT 签发/校验。

全部用标准库实现（hashlib/hmac/base64/json），不引入 PyJWT、passlib 等新依赖。
"""
import base64
import hashlib
import hmac
import json
import secrets
import time

from . import config


# ---------------------------------------------------------------- 密码
def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """返回 (password_hash, salt)。"""
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             salt.encode("utf-8"), config.PBKDF2_ROUNDS)
    return dk.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             salt.encode("utf-8"), config.PBKDF2_ROUNDS)
    return hmac.compare_digest(dk.hex(), password_hash)


# ---------------------------------------------------------------- JWT
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(seg: str) -> bytes:
    return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))


def _sign(signing_input: str) -> str:
    sig = hmac.new(config.JWT_SECRET.encode("utf-8"),
                   signing_input.encode("ascii"), hashlib.sha256).digest()
    return _b64e(sig)


def create_token(user_id: int, username: str,
                 expire_hours: int | None = None) -> str:
    now = int(time.time())
    header = {"alg": config.JWT_ALG, "typ": "JWT"}
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + (expire_hours or config.JWT_EXPIRE_HOURS) * 3600,
    }
    seg = (
        _b64e(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        + "."
        + _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    )
    return f"{seg}.{_sign(seg)}"


def decode_token(token: str) -> dict | None:
    """校验并解出 payload；任何异常一律返回 None（调用方转 401）。"""
    try:
        header_seg, payload_seg, sig_seg = token.split(".")
    except ValueError:
        return None
    if not hmac.compare_digest(_sign(f"{header_seg}.{payload_seg}"), sig_seg):
        return None
    try:
        payload = json.loads(_b64d(payload_seg))
    except Exception:
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return payload
