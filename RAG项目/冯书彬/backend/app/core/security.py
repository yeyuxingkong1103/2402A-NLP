import os
import secrets
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from backend.app.core.config import settings
from backend.app.core.crypto import hmac_digest
from backend.app.models.role import is_known_role

_ALGORITHM = "HS256"
_REFRESH_PURPOSE = "refresh-token"


def _jwt_secret() -> str:
    # 优先读取环境变量，确保测试 monkeypatch 和部署密钥即时生效。
    secret = os.getenv("APP_HMAC_KEY") or settings.APP_HMAC_KEY
    if not secret:
        raise ValueError("APP_HMAC_KEY is required")
    return secret


def _normalize_roles(roles: list[str] | None) -> list[str]:
    # 访问令牌角色只接受预定义后台角色，避免无效角色进入鉴权声明。
    normalized_roles: list[str] = []
    for role in roles or []:
        normalized_role = role.strip().lower()
        if not is_known_role(normalized_role):
            raise ValueError(f"unknown role: {role}")
        if normalized_role not in normalized_roles:
            normalized_roles.append(normalized_role)
    return normalized_roles


def utc_now() -> datetime:
    # 统一使用 UTC 时间，避免本地时区影响过期判断。
    return datetime.now(timezone.utc)


def create_access_token(user_id: str, session_id: str, roles: list[str] | None = None) -> str:
    # 访问令牌只保存用户、会话和后台角色，不包含手机号等敏感数据。
    expires_at = utc_now() + timedelta(minutes=settings.ACCESS_TOKEN_MINUTES)
    payload = {"sub": user_id, "sid": session_id, "type": "access", "exp": expires_at, "roles": _normalize_roles(roles)}
    # JWT 密钥复用 HMAC 配置密钥，避免新增未声明的敏感配置。
    return jwt.encode(payload, _jwt_secret(), algorithm=_ALGORITHM)


def create_refresh_token() -> tuple[str, datetime]:
    # Refresh Token 使用高熵随机值，服务端只存 HMAC 摘要。
    token = secrets.token_urlsafe(48)
    expires_at = utc_now() + timedelta(days=settings.REFRESH_TOKEN_DAYS)
    return token, expires_at


def decode_access_token(token: str) -> dict[str, str]:
    # 解码失败统一抛 ValueError，避免向调用方泄露验证细节。
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[_ALGORITHM])
    except JWTError as exc:
        raise ValueError("invalid access token") from exc
    # 只接受访问令牌类型，防止 refresh token 被误用于鉴权。
    if payload.get("type") != "access":
        raise ValueError("invalid access token")
    return payload


def hash_refresh_token(refresh_token: str) -> str:
    # Refresh Token 摘要使用专用 purpose，严禁保存明文 token。
    return hmac_digest(refresh_token, purpose=_REFRESH_PURPOSE)


def hash_otp_code(phone: str, client_id: str, code: str) -> str:
    # 验证码摘要绑定手机号和设备，避免跨设备复用。
    return hmac_digest(f"{phone}:{client_id}:{code}", purpose="otp-code")
