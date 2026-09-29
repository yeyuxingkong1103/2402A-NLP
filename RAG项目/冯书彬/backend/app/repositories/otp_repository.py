from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Protocol

from backend.app.services.auth_store import OtpState


class OtpBackend(Protocol):
    # OTP 后端只暴露验证码状态操作，不让认证服务依赖具体 Redis 客户端。
    def get(self, key: str) -> OtpState | None: ...
    def save(self, key: str, state: OtpState) -> None: ...
    def delete(self, key: str) -> None: ...


def _as_utc(value: datetime) -> datetime:
    # 统一时区，避免不同 Redis 序列化来源导致过期比较异常。
    if value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _serialize_state(state: OtpState) -> str:
    # Redis 只保存验证码摘要和限流元数据，绝不保存验证码明文。
    return json.dumps(
        {
            "code_hash": state.code_hash,
            "expires_at": _as_utc(state.expires_at).isoformat(),
            "attempts": state.attempts,
            "locked_until": _as_utc(state.locked_until).isoformat() if state.locked_until else None,
        },
        separators=(",", ":"),
    )


def _deserialize_state(raw: str | bytes | None) -> OtpState | None:
    # 缺失、损坏或字段不完整的缓存值都按不存在处理，避免认证接口抛内部异常。
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        return OtpState(
            code_hash=str(data["code_hash"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            attempts=int(data["attempts"]),
            locked_until=datetime.fromisoformat(data["locked_until"]) if data.get("locked_until") else None,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


class RedisOtpStore:
    # Redis 负责跨进程共享 OTP，并通过 key TTL 自动清理短期认证状态。
    def __init__(self, client, now_provider=None) -> None:
        self.client = client
        self.now_provider = now_provider or (lambda: datetime.now(timezone.utc))

    def get(self, key: str) -> OtpState | None:
        return _deserialize_state(self.client.get(key))

    def save(self, key: str, state: OtpState) -> None:
        remaining = int((_as_utc(state.expires_at) - _as_utc(self.now_provider())).total_seconds())
        if remaining <= 0:
            self.delete(key)
            return
        self.client.setex(key, remaining, _serialize_state(state))

    def delete(self, key: str) -> None:
        self.client.delete(key)
