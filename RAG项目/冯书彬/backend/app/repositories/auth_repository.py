from __future__ import annotations

from datetime import timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, MetaData, String, Table, Text, create_engine, delete, insert, select, update
from sqlalchemy.engine import Engine

from backend.app.models.user import DeviceSession, User
from backend.app.services.auth_store import AuthStore, OtpState


auth_metadata = MetaData()

users_table = Table(
    "users",
    auth_metadata,
    Column("id", String(36), primary_key=True),
    Column("encrypted_phone", Text, nullable=False),
    Column("phone_nonce", String(128), nullable=False),
    Column("phone_encrypted_data_key", String(128), nullable=False),
    Column("phone_key_version", String(32), nullable=False),
    Column("phone_hmac", String(64), nullable=False, unique=True, index=True),
    Column("agreement_version", String(32), nullable=False),
    Column("privacy_policy_version", String(32), nullable=False),
    Column("adult_confirmed", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

device_sessions_table = Table(
    "device_sessions",
    auth_metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), ForeignKey("users.id"), nullable=False, index=True),
    Column("refresh_token_hash", String(64), nullable=False, unique=True, index=True),
    Column("device_identifier", String(128), nullable=False),
    Column("user_agent", String(512), nullable=False),
    Column("login_at", DateTime(timezone=True), nullable=False),
    Column("last_activity_at", DateTime(timezone=True), nullable=False),
    Column("province_location", String(64), nullable=True),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("revoked", Boolean, nullable=False),
    Column("revoked_at", DateTime(timezone=True), nullable=True),
)


def create_auth_tables(engine: Engine) -> None:
    # 测试和本地验证可显式创建认证相关表；生产环境仍应使用 Alembic 迁移。
    auth_metadata.create_all(engine, tables=[users_table, device_sessions_table], checkfirst=True)


def create_engine_from_url(database_url: str, **kwargs) -> Engine:
    # 只集中封装引擎创建，避免业务层散落数据库 URL 处理逻辑。
    return create_engine(database_url, future=True, **kwargs)


def _as_utc(value):
    # SQLite 等测试数据库可能丢失 tzinfo，统一补回 UTC 以便过期判断稳定。
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _row_to_user(row) -> User | None:
    if row is None:
        return None
    data = row._mapping
    return User(
        id=data["id"],
        encrypted_phone=data["encrypted_phone"],
        phone_nonce=data["phone_nonce"],
        phone_encrypted_data_key=data["phone_encrypted_data_key"],
        phone_key_version=data["phone_key_version"],
        phone_hmac=data["phone_hmac"],
        agreement_version=data["agreement_version"],
        privacy_policy_version=data["privacy_policy_version"],
        adult_confirmed=bool(data["adult_confirmed"]),
        created_at=_as_utc(data["created_at"]),
        updated_at=_as_utc(data["updated_at"]),
    )


def _row_to_session(row) -> DeviceSession | None:
    if row is None:
        return None
    data = row._mapping
    return DeviceSession(
        id=data["id"],
        user_id=data["user_id"],
        refresh_token_hash=data["refresh_token_hash"],
        device_identifier=data["device_identifier"],
        user_agent=data["user_agent"],
        login_at=_as_utc(data["login_at"]),
        last_activity_at=_as_utc(data["last_activity_at"]),
        province_location=data["province_location"],
        expires_at=_as_utc(data["expires_at"]),
        revoked=bool(data["revoked"]),
        revoked_at=_as_utc(data["revoked_at"]),
    )


def _user_values(user: User) -> dict:
    return {
        "id": user.id,
        "encrypted_phone": user.encrypted_phone,
        "phone_nonce": user.phone_nonce,
        "phone_encrypted_data_key": user.phone_encrypted_data_key,
        "phone_key_version": user.phone_key_version,
        "phone_hmac": user.phone_hmac,
        "agreement_version": user.agreement_version,
        "privacy_policy_version": user.privacy_policy_version,
        "adult_confirmed": user.adult_confirmed,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }


def _session_values(session: DeviceSession) -> dict:
    return {
        "id": session.id,
        "user_id": session.user_id,
        "refresh_token_hash": session.refresh_token_hash,
        "device_identifier": session.device_identifier,
        "user_agent": session.user_agent,
        "login_at": session.login_at,
        "last_activity_at": session.last_activity_at,
        "province_location": session.province_location,
        "expires_at": session.expires_at,
        "revoked": session.revoked,
        "revoked_at": session.revoked_at,
    }


class SQLAlchemyAuthStore(AuthStore):
    # SQL 持久化认证存储先覆盖用户和设备会话；OTP 后续可替换为 Redis。
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.otp_store: dict[str, OtpState] = {}
        self.otp_backend = None

    def set_otp_backend(self, backend) -> None:
        # SQL 用户/会话与 Redis OTP 解耦，便于分别扩展和故障排查。
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
        with self.engine.begin() as connection:
            row = connection.execute(select(users_table).where(users_table.c.phone_hmac == phone_hash)).first()
        return _row_to_user(row)

    def get_user_by_id(self, user_id: str) -> User | None:
        with self.engine.begin() as connection:
            row = connection.execute(select(users_table).where(users_table.c.id == user_id)).first()
        return _row_to_user(row)

    def is_user_deleted(self, user_id: str) -> bool:
        return self.get_user_by_id(user_id) is None

    def save_user(self, user: User) -> None:
        values = _user_values(user)
        with self.engine.begin() as connection:
            exists = connection.execute(select(users_table.c.id).where(users_table.c.id == user.id)).first()
            if exists:
                connection.execute(update(users_table).where(users_table.c.id == user.id).values(**values))
                return
            connection.execute(insert(users_table).values(**values))

    def get_session_by_hash(self, token_hash: str) -> DeviceSession | None:
        with self.engine.begin() as connection:
            row = connection.execute(select(device_sessions_table).where(device_sessions_table.c.refresh_token_hash == token_hash)).first()
        return _row_to_session(row)

    def save_session(self, session: DeviceSession) -> None:
        values = _session_values(session)
        with self.engine.begin() as connection:
            exists = connection.execute(select(device_sessions_table.c.id).where(device_sessions_table.c.id == session.id)).first()
            if exists:
                connection.execute(update(device_sessions_table).where(device_sessions_table.c.id == session.id).values(**values))
                return
            connection.execute(insert(device_sessions_table).values(**values))

    def delete_session_hash(self, token_hash: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(device_sessions_table).where(device_sessions_table.c.refresh_token_hash == token_hash))

    def get_user_session_hashes(self, user_id: str) -> list[str]:
        with self.engine.begin() as connection:
            rows = connection.execute(select(device_sessions_table.c.refresh_token_hash).where(device_sessions_table.c.user_id == user_id)).all()
        return [row._mapping["refresh_token_hash"] for row in rows]

    def replace_user_session_hash(self, user_id: str, old_hash: str, new_hash: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(device_sessions_table)
                .where(device_sessions_table.c.user_id == user_id)
                .where(device_sessions_table.c.refresh_token_hash == old_hash)
                .values(refresh_token_hash=new_hash)
            )

    def delete_user(self, user_id: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(device_sessions_table).where(device_sessions_table.c.user_id == user_id))
            connection.execute(delete(users_table).where(users_table.c.id == user_id))
