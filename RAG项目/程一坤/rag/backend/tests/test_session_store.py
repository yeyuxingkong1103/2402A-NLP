"""测试会话存储。"""

from __future__ import annotations

import pytest

from app.auth.session_store import SessionStore, SessionUser
from app.core.config import settings


def test_redis_url_points_to_test_db() -> None:
    """验证测试运行时 Redis 指向 9 号库（环境变量覆盖生效）。"""
    assert settings.redis_url == "redis://127.0.0.1:6379/9", (
        f"测试隔离失效：期望 Redis 9 号库，实际 {settings.redis_url}"
    )


class FakeRedis:
    """轻量假 Redis：内存字典 + 记录调用，不依赖真实 Redis 服务。"""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}  # 键值存储
        self.sets: dict[str, set[str]] = {}  # 集合存储
        self.ttls: dict[str, int] = {}  # TTL 记录
        self.should_raise: bool = False  # 是否模拟异常

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        """设置键值对，可选 TTL。"""
        if self.should_raise:
            raise ConnectionError("模拟 Redis 连接失败")
        self.data[key] = value
        if ex is not None:
            self.ttls[key] = ex

    def get(self, key: str) -> str | None:
        """获取键值。"""
        if self.should_raise:
            raise ConnectionError("模拟 Redis 连接失败")
        return self.data.get(key)

    def delete(self, *keys: str) -> int:
        """删除键。"""
        if self.should_raise:
            raise ConnectionError("模拟 Redis 连接失败")
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                count += 1
            if key in self.sets:
                del self.sets[key]
                count += 1
        return count

    def sadd(self, key: str, *members: str) -> int:
        """添加集合成员。"""
        if self.should_raise:
            raise ConnectionError("模拟 Redis 连接失败")
        if key not in self.sets:
            self.sets[key] = set()
        before_count = len(self.sets[key])
        self.sets[key].update(members)
        return len(self.sets[key]) - before_count

    def smembers(self, key: str) -> set[str]:
        """获取集合所有成员。"""
        if self.should_raise:
            raise ConnectionError("模拟 Redis 连接失败")
        return self.sets.get(key, set())

    def expire(self, key: str, seconds: int) -> bool:
        """设置 TTL。"""
        if self.should_raise:
            raise ConnectionError("模拟 Redis 连接失败")
        self.ttls[key] = seconds
        return True

    def ttl(self, key: str) -> int:
        """获取 TTL，-1 表示永不过期，-2 表示不存在。"""
        if self.should_raise:
            raise ConnectionError("模拟 Redis 连接失败")
        if key not in self.data and key not in self.sets:
            return -2
        return self.ttls.get(key, -1)

    def srem(self, key: str, *members: str) -> int:
        """移除集合成员。"""
        if self.should_raise:
            raise ConnectionError("模拟 Redis 连接失败")
        if key not in self.sets:
            return 0
        count = 0
        for member in members:
            if member in self.sets[key]:
                self.sets[key].remove(member)
                count += 1
        return count


def test_create_and_read_session() -> None:
    """测试创建会话后能读回相同 user_id 与 is_admin。"""
    redis = FakeRedis()
    store = SessionStore(redis, session_ttl_seconds=3600)

    token = store.create_session_token(user_id="user123", is_admin=False)
    session = store.read_session(token)

    assert session is not None
    assert session.user_id == "user123"
    assert session.is_admin is False


def test_revoke_session() -> None:
    """测试撤销后同一令牌读不回来。"""
    redis = FakeRedis()
    store = SessionStore(redis, session_ttl_seconds=3600)

    token = store.create_session_token(user_id="user123", is_admin=False)
    store.revoke_session(token)
    session = store.read_session(token)

    assert session is None


def test_revoke_all_sessions() -> None:
    """测试 revoke_all_sessions 能一次撤销该用户的所有令牌。"""
    redis = FakeRedis()
    store = SessionStore(redis, session_ttl_seconds=3600)

    token1 = store.create_session_token(user_id="user123", is_admin=False)
    token2 = store.create_session_token(user_id="user123", is_admin=False)
    token3 = store.create_session_token(user_id="user456", is_admin=True)

    # 撤销 user123 的所有会话
    store.revoke_all_sessions("user123")

    # user123 的两个令牌都失效
    assert store.read_session(token1) is None
    assert store.read_session(token2) is None
    # user456 的令牌仍然有效
    assert store.read_session(token3) is not None


def test_forged_token_returns_none() -> None:
    """测试伪造的令牌读不回来。"""
    redis = FakeRedis()
    store = SessionStore(redis, session_ttl_seconds=3600)

    session = store.read_session("fake-random-token-12345678")

    assert session is None


def test_redis_exception_propagates() -> None:
    """测试 Redis 抛异常时方法向上抛错（不得吞掉异常）。"""
    redis = FakeRedis()
    redis.should_raise = True
    store = SessionStore(redis, session_ttl_seconds=3600)

    # create_session_token 应该抛出异常
    with pytest.raises(ConnectionError, match="模拟 Redis 连接失败"):
        store.create_session_token(user_id="user123", is_admin=False)

    # read_session 应该抛出异常
    with pytest.raises(ConnectionError, match="模拟 Redis 连接失败"):
        store.read_session("some-token")

    # revoke_session 应该抛出异常
    with pytest.raises(ConnectionError, match="模拟 Redis 连接失败"):
        store.revoke_session("some-token")


def test_token_format() -> None:
    """测试令牌形态：长度足够、不含用户信息。"""
    redis = FakeRedis()
    store = SessionStore(redis, session_ttl_seconds=3600)

    token = store.create_session_token(user_id="user123", is_admin=False)

    # 令牌长度应该足够（secrets.token_urlsafe(32) 生成约 43 字符）
    assert len(token) >= 32

    # 令牌不应该包含用户信息明文
    assert "user123" not in token
    assert "user" not in token.lower()


def test_user_sessions_key_has_ttl() -> None:
    """测试 user_sessions:<user_id> 集合设置了 TTL，不会永久存在。"""
    redis = FakeRedis()
    store = SessionStore(redis, session_ttl_seconds=7200)

    token = store.create_session_token(user_id="test-user-id", is_admin=False)

    # 验证 user_sessions 键存在
    user_sessions_key = "user_sessions:test-user-id"
    assert user_sessions_key in redis.sets
    assert token in redis.sets[user_sessions_key]

    # 验证 TTL 已设置（不为 -1）
    ttl = redis.ttl(user_sessions_key)
    assert ttl > 0, f"user_sessions 集合 TTL 应大于 0，实际为 {ttl}"
    assert ttl == 7200


def test_session_binds_real_user_id() -> None:
    """测试会话绑定真实 user_id，不得出现 'pending' 占位值。"""
    redis = FakeRedis()
    store = SessionStore(redis, session_ttl_seconds=3600)

    user_id = "real-user-abc123"
    token = store.create_session_token(user_id=user_id, is_admin=False)

    # 验证会话中的 user_id 与传入的一致
    session = store.read_session(token)
    assert session is not None
    assert session.user_id == user_id

    # 验证 user_sessions 键使用真实 user_id
    user_sessions_key = f"user_sessions:{user_id}"
    assert user_sessions_key in redis.sets
    assert token in redis.sets[user_sessions_key]

    # 验证 Redis 中不存在 "pending" 占位键
    assert "user_sessions:pending" not in redis.sets
    assert "pending" not in session.user_id
