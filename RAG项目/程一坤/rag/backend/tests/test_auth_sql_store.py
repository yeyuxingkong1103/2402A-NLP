"""认证落库测试（批次5）：SqlAuthStore 行为。

核心场景：注册后"重启"（新建 SqlAuthStore 实例指向同一数据库）仍能登录。
用 SQLite 内存库（StaticPool 共享连接）模拟 MySQL，不依赖真实数据库；
会话令牌用 FakeRedis 替身（令牌机制本身在 test_session_store 已覆盖）。
"""
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.service import AuthService
from app.auth.session_store import SessionStore
from app.auth.sql_store import SqlAuthStore
from app.db.base import Base
from tests.conftest import FakeRedis


@pytest.fixture()
def shared_engine():
    """共享连接的 SQLite 内存库：多个 SqlAuthStore 实例（模拟重启）看到同一份数据。"""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


def make_auth_service(engine) -> AuthService:
    """构造一套完整认证服务（等价于一次后端进程启动）。"""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return AuthService(
        store=SqlAuthStore(factory),
        session_store=SessionStore(FakeRedis(), 3600),
    )


def register_user(service: AuthService, email: str, password: str) -> str:
    """走完整注册流程（验证码 → 注册），返回访问令牌。"""
    code = service.send_code(email, "register")
    return service.register(email, password, code)


# ==================== 核心验收：重启后仍能登录 ====================


def test_registered_user_can_login_after_restart(shared_engine) -> None:
    """注册 → 新建 store 实例（模拟进程重启）→ 同账号登录成功。"""
    # 第一次"启动"：注册用户
    first_service = make_auth_service(shared_engine)
    register_user(first_service, "persist@qq.com", "password123")

    # 第二次"启动"：全新 AuthService 实例（内存全空，只有数据库是同一份）
    second_service = make_auth_service(shared_engine)

    # 重启后用同一账号密码登录必须成功
    token = second_service.login("persist@qq.com", "password123")
    assert token

    # 令牌绑定的 user_id 必须是库里的 user_key（32 位十六进制）
    session_user = second_service.session_store.read_session(token)
    assert session_user is not None
    assert len(session_user.user_id) == 32
    int(session_user.user_id, 16)  # 必须是合法十六进制

    # user_key 与第一次启动注册时生成的值一致（数据库持久，而非临时生成）
    first_user = first_service.store.get_user("persist@qq.com")
    assert session_user.user_id == first_user.user_key


def test_wrong_password_rejected_after_restart(shared_engine) -> None:
    """重启后密码错误必须被拒绝。"""
    service = make_auth_service(shared_engine)
    register_user(service, "wrongpwd@qq.com", "password123")

    second_service = make_auth_service(shared_engine)
    with pytest.raises(PermissionError):
        second_service.login("wrongpwd@qq.com", "wrong-password")


def test_unknown_user_rejected_after_restart(shared_engine) -> None:
    """重启后不存在的用户登录必须被拒绝（不允许内存兜底）。"""
    second_service = make_auth_service(shared_engine)
    with pytest.raises(PermissionError):
        second_service.login("ghost@qq.com", "password123")


# ==================== 注册约束 ====================


def test_duplicate_email_registration_rejected(shared_engine) -> None:
    """重复 email 注册必须被拒绝，错误信息明确指出邮箱已注册。"""
    service = make_auth_service(shared_engine)
    register_user(service, "dup@qq.com", "password123")

    # 再走一次完整注册流程（新验证码）
    code = service.send_code("dup@qq.com", "register")
    with pytest.raises(ValueError) as exc_info:
        service.register("dup@qq.com", "password456", code)
    assert "已注册" in str(exc_info.value)


def test_user_key_is_unique_32_hex(shared_engine) -> None:
    """user_key 必须是 32 位十六进制，且每个用户唯一。"""
    service = make_auth_service(shared_engine)
    register_user(service, "keya@qq.com", "password123")
    register_user(service, "keyb@qq.com", "password123")

    user_a = service.store.get_user("keya@qq.com")
    user_b = service.store.get_user("keyb@qq.com")
    assert user_a.user_key != user_b.user_key
    assert len(user_a.user_key) == 32
    int(user_a.user_key, 16)


def test_disabled_user_cannot_login(shared_engine) -> None:
    """is_active=False 的用户登录必须被拒绝（对外不暴露禁用状态）。"""
    service = make_auth_service(shared_engine)
    register_user(service, "disabled@qq.com", "password123")

    # 管理侧直接置为禁用
    user = service.store.get_user("disabled@qq.com")
    user.is_active = False
    service.store.save_user(user)

    second_service = make_auth_service(shared_engine)
    with pytest.raises(PermissionError):
        second_service.login("disabled@qq.com", "password123")


# ==================== 密码哈希 ====================


def test_password_hash_is_not_plaintext(shared_engine) -> None:
    """库里的 password_hash 不能是明文，且能用现有 verify_password 校验。"""
    from app.auth.service import verify_password

    from app.db.sql_models import User as UserORM

    service = make_auth_service(shared_engine)
    register_user(service, "hash@qq.com", "s3cret-password")

    factory = sessionmaker(bind=shared_engine, expire_on_commit=False)
    with factory() as session:
        record = session.scalar(
            select(UserORM).where(UserORM.email == "hash@qq.com")
        )
        assert record is not None
        # 不是明文
        assert record.password_hash != "s3cret-password"
        assert "s3cret" not in record.password_hash
        # 长度合理（盐 + 摘要的 base64，远长于普通密码）
        assert len(record.password_hash) >= 40
        # 现有校验逻辑可用
        assert verify_password("s3cret-password", record.password_hash)
        assert not verify_password("wrong", record.password_hash)


# ==================== 改密 ====================


def test_reset_password_survives_restart_and_revokes_old_sessions(shared_engine) -> None:
    """改密后：旧令牌立即失效；重启后新密码可登录、旧密码被拒。"""
    service = make_auth_service(shared_engine)
    old_token = register_user(service, "reset@qq.com", "old-password-1")

    # 改密
    code = service.send_code("reset@qq.com", "reset")
    service.reset_password("reset@qq.com", "new-password-2", code)

    # 旧令牌立即失效
    assert service.session_store.read_session(old_token) is None

    # 重启后：新密码可登录，旧密码被拒
    second_service = make_auth_service(shared_engine)
    new_token = second_service.login("reset@qq.com", "new-password-2")
    assert new_token
    with pytest.raises(PermissionError):
        second_service.login("reset@qq.com", "old-password-1")
