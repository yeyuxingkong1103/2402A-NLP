"""认证与鉴权：PBKDF2 口令散列 + HMAC 签名令牌 + 角色授权。

不引入额外依赖（只用标准库 hash/hmac/base64/json），令牌无状态：
``base64url(payload_json).base64url(hmac_sha256)``，payload 内含 exp 过期时间。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import Config, get_config
from ..errors import AuthError, ValidationError
from ..logging_conf import get_logger
from ..store.redis_store import RedisStore, get_redis

logger = get_logger(__name__)

ALGORITHM = "pbkdf2_sha256"


# --------------------------------------------------------------------- 口令
def hash_password(password: str, iterations: int = 120000) -> str:
    if not password:
        raise ValidationError("密码不能为空")
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    return "$".join(
        [ALGORITHM, str(int(iterations)), _b64(salt), _b64(digest)]
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt_b64, digest_b64 = stored.split("$")
        if algorithm != ALGORITHM:
            return False
        salt = _unb64(salt_b64)
        expected = _unb64(digest_b64)
    except (ValueError, TypeError):
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    return hmac.compare_digest(candidate, expected)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


# --------------------------------------------------------------------- 令牌
class TokenCodec:
    """HMAC-SHA256 签名的无状态令牌。"""

    def __init__(self, secret: str, ttl: int = 43200) -> None:
        if not secret:
            raise AuthError("security.secret 未配置")
        self.secret = secret.encode("utf-8")
        self.ttl = int(ttl)

    def encode(self, payload: dict[str, Any], ttl: int | None = None) -> str:
        body = dict(payload)
        now = int(time.time())
        body["iat"] = now
        body["exp"] = now + int(ttl or self.ttl)
        raw = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return f"{encoded}.{self._sign(encoded)}"

    def decode(self, token: str) -> dict[str, Any]:
        if not token or token.count(".") != 1:
            raise AuthError("令牌格式不正确")
        encoded, signature = token.split(".", 1)
        if not hmac.compare_digest(self._sign(encoded), signature):
            raise AuthError("令牌签名校验失败")
        try:
            payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise AuthError("令牌内容无法解析") from exc
        if int(payload.get("exp", 0)) < int(time.time()):
            raise AuthError("令牌已过期，请重新登录")
        return payload

    def _sign(self, encoded: str) -> str:
        digest = hmac.new(self.secret, encoded.encode("ascii"), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


# --------------------------------------------------------------------- 主体
@dataclass(slots=True)
class AuthContext:
    """已认证用户。"""

    username: str
    display_name: str = ""
    roles: list[str] = field(default_factory=list)
    is_admin: bool = False

    @property
    def user_id(self) -> str:
        return self.username

    def can_use_role(self, role_id: str) -> bool:
        if self.is_admin or "*" in self.roles:
            return True
        return role_id in self.roles

    def visible_roles(self, enabled_ids: Sequence[str]) -> list[str]:
        if self.is_admin or "*" in self.roles:
            return list(enabled_ids)
        return [role for role in enabled_ids if role in self.roles]

    def to_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "display_name": self.display_name,
            "roles": self.roles,
            "is_admin": self.is_admin,
        }


class AuthService:
    """用户认证服务（用户数据存 Redis Hash）。"""

    def __init__(self, config: Config | None = None, store: RedisStore | None = None) -> None:
        self.config = config or get_config()
        self.store = store or get_redis(self.config)
        security = self.config.security
        self.iterations = int(security.get("password_iterations", 120000))
        self.codec = TokenCodec(str(security.get("secret", "")), int(security.get("token_ttl", 43200)))
        self._seeded = False

    # ------------------------------------------------------------------ 初始化
    def seed(self, force: bool = False) -> int:
        if self._seeded and not force:
            return 0
        created = self.store.seed_users(
            self.config.get("security.users_seed", []) or [],
            password_hasher=lambda password: hash_password(password, self.iterations),
        )
        self._seeded = True
        return created

    def create_user(
        self,
        username: str,
        password: str,
        display_name: str = "",
        roles: Sequence[str] = ("*",),
        is_admin: bool = False,
    ) -> dict[str, Any]:
        username = (username or "").strip()
        if not username or len(username) > 32:
            raise ValidationError("用户名长度需在 1-32 之间")
        if len(password or "") < 6:
            raise ValidationError("密码至少 6 位")
        if self.store.get_user(username):
            raise ValidationError(f"用户已存在：{username}")
        return self.store.create_user(
            username=username,
            password=password,
            display_name=display_name or username,
            roles=list(roles),
            is_admin=is_admin,
            password_hasher=lambda value: hash_password(value, self.iterations),
        )

    # ------------------------------------------------------------------ 登录
    def login(self, username: str, password: str) -> tuple[str, AuthContext]:
        user = self.store.get_user((username or "").strip())
        if not user or not verify_password(password or "", str(user.get("password_hash", ""))):
            logger.warning("登录失败：%s", username)
            raise AuthError("用户名或密码错误")
        context = self._to_context(user)
        token = self.codec.encode(context.to_dict())
        self.store.mark_online(context.user_id)
        self.store.incr_stat("logins")
        logger.info("登录成功：%s", context.username)
        return token, context

    def resolve(self, token: str) -> AuthContext:
        payload = self.codec.decode(token)
        context = AuthContext(
            username=str(payload.get("username", "")),
            display_name=str(payload.get("display_name", "")),
            roles=[str(item) for item in payload.get("roles", [])],
            is_admin=bool(payload.get("is_admin", False)),
        )
        if not context.username:
            raise AuthError("令牌缺少用户信息")
        self.store.mark_online(context.user_id)
        return context

    def logout(self, token: str) -> None:
        try:
            payload = self.codec.decode(token)
            self.store.client.srem(self.store.key("online"), str(payload.get("username", "")))
        except AuthError:
            return

    @staticmethod
    def _to_context(user: dict[str, Any]) -> AuthContext:
        return AuthContext(
            username=str(user.get("username", "")),
            display_name=str(user.get("display_name", "")),
            roles=[str(item) for item in (user.get("roles") or [])],
            is_admin=bool(user.get("is_admin", False)),
        )


_service: AuthService | None = None


def get_auth_service(config: Config | None = None) -> AuthService:
    global _service
    if _service is None:
        _service = AuthService(config)
    return _service
