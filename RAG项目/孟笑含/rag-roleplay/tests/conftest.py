# -*- coding: utf-8 -*-
"""测试公共设施：假 LLM、真实 Redis 15 号库的短期记忆。"""
import os

import pytest
import redis

from app.store.memory import RedisMemoryStore


class FakeLLM:
    """记录请求、返回预设回复的假大模型。"""

    def __init__(self, reply="你好呀，有什么可以帮你？"):
        self.reply = reply
        self.last_messages = None

    async def chat(self, messages):
        self.last_messages = messages
        return self.reply

    async def chat_stream(self, messages):
        self.last_messages = messages
        for chunk in ["你", "好", "呀"]:
            yield chunk


@pytest.fixture()
def test_redis():
    client = redis.Redis(
        host=os.getenv("REDIS_HOST", "127.0.0.1"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        db=15,
    )
    client.ping()
    client.flushdb()
    yield client
    client.flushdb()
    client.close()


@pytest.fixture()
def test_memory(test_redis):
    return RedisMemoryStore(test_redis, max_rounds=10, ttl_seconds=3600)


@pytest.fixture()
def fake_llm():
    return FakeLLM()
