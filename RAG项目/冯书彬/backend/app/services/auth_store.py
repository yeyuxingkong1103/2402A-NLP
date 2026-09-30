from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from backend.app.models.user import DeviceSession, User


@dataclass
class OtpState:
    # 只保存验证码摘要，不保存明文验证码。
    code_hash: str
    expires_at: datetime
    attempts: int = 0
    locked_until: datetime | None = None


class OtpBackend(Protocol):
    # OTP 后端可使用内存或 Redis，认证服务只依赖这组最小操作。
    def get(self, key: str) -> OtpState | None: ...
    def save(self, key: str, state: OtpState) -> None: ...
    def delete(self, key: str) -> None: ...


class AuthStore(Protocol):
    # 存储边界可替换为数据库 repository，默认内存实现只用于单进程测试/MVP。
    def set_otp_backend(self, backend: OtpBackend | None) -> None: ...
    def get_otp(self, key: str) -> OtpState | None: ...
    def save_otp(self, key: str, state: OtpState) -> None: ...
    def delete_otp(self, key: str) -> None: ...
    def get_user_by_phone_hash(self, phone_hash: str) -> User | None: ...
    def get_user_by_id(self, user_id: str) -> User | None: ...
    def is_user_deleted(self, user_id: str) -> bool: ...
    def save_user(self, user: User) -> None: ...
    def get_session_by_hash(self, token_hash: str) -> DeviceSession | None: ...
    def save_session(self, session: DeviceSession) -> None: ...
    def delete_session_hash(self, token_hash: str) -> None: ...
    def get_user_session_hashes(self, user_id: str) -> list[str]: ...
    def replace_user_session_hash(self, user_id: str, old_hash: str, new_hash: str) -> None: ...
    def delete_user(self, user_id: str) -> None: ...


class InMemoryAuthStore:
    # 内存 store 便于测试；生产多进程必须替换为持久化 repository。
    def __init__(self) -> None:
        self.otp_store: dict[str, OtpState] = {}
        self.otp_backend: OtpBackend | None = None
        self.users_by_phone: dict[str, User] = {}
        self.sessions_by_hash: dict[str, DeviceSession] = {}
        self.sessions_by_user: dict[str, list[str]] = {}
        self.deleted_user_ids: set[str] = set()

    def set_otp_backend(self, backend: OtpBackend | None) -> None:
        # 显式注入 Redis 等跨进程 OTP 后端；None 表示使用本地内存。
        self.otp_backend = backend

    def get_otp(self, key: str) -> OtpState | None:
        return self.otp_backend.get(key) if self.otp_backend else self.otp_store.get(key)

    def save_otp(self, key: str, state: OtpState) -> None:
        if self.otp_backend:
            self.otp_backend.save(key, state)
            return
        self.otp_store[key] = state

    def delete_otp(self, key: str) -> None:
        if self.otp_backend:
            self.otp_backend.delete(key)
            return
        self.otp_store.pop(key, None)

    def get_user_by_phone_hash(self, phone_hash: str) -> User | None:
        user = self.users_by_phone.get(phone_hash)
        if user is None or user.id in self.deleted_user_ids:
            return None
        return user

    def get_user_by_id(self, user_id: str) -> User | None:
        if user_id in self.deleted_user_ids:
            return None
        for user in self.users_by_phone.values():
            if user.id == user_id:
                return user
        return None

    def is_user_deleted(self, user_id: str) -> bool:
        # 注销标记独立于用户主记录，避免旧 token 通过角色字段绕过检查。
        return user_id in self.deleted_user_ids

    def save_user(self, user: User) -> None:
        self.deleted_user_ids.discard(user.id)
        self.users_by_phone[user.phone_hmac] = user

    def get_session_by_hash(self, token_hash: str) -> DeviceSession | None:
        return self.sessions_by_hash.get(token_hash)

    def save_session(self, session: DeviceSession) -> None:
        self.sessions_by_hash[session.refresh_token_hash] = session
        hashes = self.sessions_by_user.setdefault(session.user_id, [])
        if session.refresh_token_hash not in hashes:
            hashes.append(session.refresh_token_hash)

    def delete_session_hash(self, token_hash: str) -> None:
        self.sessions_by_hash.pop(token_hash, None)

    def get_user_session_hashes(self, user_id: str) -> list[str]:
        return list(self.sessions_by_user.get(user_id, []))

    def replace_user_session_hash(self, user_id: str, old_hash: str, new_hash: str) -> None:
        hashes = self.sessions_by_user.get(user_id, [])
        self.sessions_by_user[user_id] = [new_hash if item == old_hash else item for item in hashes]

    def delete_user(self, user_id: str) -> None:
        # 注销时删除用户主记录和该用户所有设备会话摘要，并记录注销状态使旧 access token 失效。
        self.deleted_user_ids.add(user_id)
        for phone_hash, user in list(self.users_by_phone.items()):
            if user.id == user_id:
                self.users_by_phone.pop(phone_hash, None)
        for token_hash in self.sessions_by_user.pop(user_id, []):
            self.sessions_by_hash.pop(token_hash, None)
