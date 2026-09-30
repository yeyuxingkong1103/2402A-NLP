"""测试 Redis 短期记忆存储。"""

from __future__ import annotations

import json

import pytest

from app.memory.short_term import ShortTermMemoryStore


class FakeRedis:
    """支持短期记忆所需命令的最小 Redis 替身。"""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.should_raise = False

    def rpush(self, key: str, value: str) -> int:
        self._check()
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    def ltrim(self, key: str, start: int, end: int) -> bool:
        self._check()
        values = self.lists.get(key, [])
        if end == -1:
            self.lists[key] = values[start:]
        else:
            self.lists[key] = values[start : end + 1]
        return True

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        self._check()
        values = self.lists.get(key, [])
        if end == -1:
            return values[start:]
        return values[start : end + 1]

    def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self._check()
        self.values[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True

    def get(self, key: str) -> str | None:
        self._check()
        return self.values.get(key)

    def expire(self, key: str, seconds: int) -> bool:
        self._check()
        self.ttls[key] = seconds
        return True

    def delete(self, *keys: str) -> int:
        self._check()
        deleted = 0
        for key in keys:
            deleted += int(key in self.lists) + int(key in self.values)
            self.lists.pop(key, None)
            self.values.pop(key, None)
            self.ttls.pop(key, None)
        return deleted

    def ttl(self, key: str) -> int:
        self._check()
        return self.ttls.get(key, -2)

    def _check(self) -> None:
        if self.should_raise:
            raise ConnectionError("模拟 Redis 连接失败")


def test_append_and_read_recent_messages_with_bounded_window() -> None:
    redis = FakeRedis()
    store = ShortTermMemoryStore(redis, ttl_seconds=3600, max_messages=3)

    for index in range(4):
        store.append_message("user-1", "session-1", {"role": "user", "content": str(index)})

    assert store.read_messages("user-1", "session-1") == [
        {"role": "user", "content": "1"},
        {"role": "user", "content": "2"},
        {"role": "user", "content": "3"},
    ]


def test_message_list_has_ttl() -> None:
    redis = FakeRedis()
    store = ShortTermMemoryStore(redis, ttl_seconds=7200, max_messages=4)

    store.append_message("user-1", "session-1", {"role": "assistant", "content": "答复"})

    assert redis.ttl("short_memory:user-1:session-1:messages") == 7200


def test_summary_is_read_and_has_ttl() -> None:
    redis = FakeRedis()
    store = ShortTermMemoryStore(redis, ttl_seconds=1800, max_messages=4)

    store.write_summary("user-1", "session-1", "用户在咨询劳动合同解除")

    assert store.read_summary("user-1", "session-1") == "用户在咨询劳动合同解除"
    assert redis.ttl("short_memory:user-1:session-1:summary") == 1800


def test_user_and_session_are_isolated() -> None:
    redis = FakeRedis()
    store = ShortTermMemoryStore(redis, ttl_seconds=3600, max_messages=4)

    store.append_message("user-a", "same-session", {"role": "user", "content": "机密"})
    store.append_message("user-b", "same-session", {"role": "user", "content": "其他"})
    store.append_message("user-a", "other-session", {"role": "user", "content": "另一会话"})

    assert store.read_messages("user-a", "same-session") == [
        {"role": "user", "content": "机密"}
    ]
    assert store.read_messages("user-b", "same-session") == [
        {"role": "user", "content": "其他"}
    ]
    assert store.read_messages("user-a", "other-session") == [
        {"role": "user", "content": "另一会话"}
    ]


def test_delete_session_memory_removes_messages_and_summary() -> None:
    redis = FakeRedis()
    store = ShortTermMemoryStore(redis, ttl_seconds=3600, max_messages=4)

    store.append_message("user-1", "session-1", {"role": "user", "content": "问题"})
    store.write_summary("user-1", "session-1", "摘要")
    store.delete_session_memory("user-1", "session-1")

    assert store.read_messages("user-1", "session-1") == []
    assert store.read_summary("user-1", "session-1") is None


def test_redis_exception_propagates() -> None:
    redis = FakeRedis()
    redis.should_raise = True
    store = ShortTermMemoryStore(redis, ttl_seconds=3600, max_messages=4)

    with pytest.raises(ConnectionError, match="模拟 Redis 连接失败"):
        store.read_messages("user-1", "session-1")


def test_message_serialization_is_json() -> None:
    redis = FakeRedis()
    store = ShortTermMemoryStore(redis, ttl_seconds=3600, max_messages=4)

    message = {"role": "user", "content": "中文"}
    store.append_message("user-1", "session-1", message)

    stored = redis.lists["short_memory:user-1:session-1:messages"][0]
    assert json.loads(stored) == message
