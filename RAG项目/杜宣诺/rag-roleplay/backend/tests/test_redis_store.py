import pytest
import fakeredis.aioredis

from app.store.redis_store import RedisStore


@pytest.fixture
async def store():
    s = RedisStore.__new__(RedisStore)
    s.redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return s


async def test_push_and_recent(store):
    await store.push_message(1, "user", "你好")
    await store.push_message(1, "assistant", "你好呀")
    recent = await store.get_recent(1, 10)
    assert len(recent) == 2
    assert recent[0]["role"] == "user"


async def test_summary_roundtrip(store):
    assert await store.get_summary(1) is None
    await store.set_summary(1, "用户喜欢猫")
    assert await store.get_summary(1) == "用户喜欢猫"
