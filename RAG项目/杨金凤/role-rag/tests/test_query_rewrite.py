"""query_rewrite 单元测试：改写、降级、缓存、history 截断。"""
from unittest.mock import MagicMock

import query_rewrite


HISTORY = [
    {"role": "user", "content": "我最近血压 150/95"},
    {"role": "assistant", "content": "属于 1 级高血压，建议低盐饮食。"},
]
REWRITTEN = "我血压 150/95 属于几级高血压，该怎么办"


def _fake_client(content):
    """构造 (client, model)，client 的 create 返回含 content 的假响应。"""
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    client.chat.completions.create.return_value = resp
    return client, "deepseek-flash"


def _patch_client(monkeypatch, content=REWRITTEN):
    client, model = _fake_client(content)
    monkeypatch.setattr(query_rewrite, "_client", lambda: (client, model))
    return client


def _patch_redis(monkeypatch, cached=None):
    redis = MagicMock()
    redis.get.return_value = cached
    monkeypatch.setattr(query_rewrite, "_redis", lambda: redis)
    return redis


def test_rewrite_query_empty_history(monkeypatch):
    """history 为空（第一轮）→ 原样返回，不调 LLM。"""
    client = _patch_client(monkeypatch)
    assert query_rewrite.rewrite_query("那我该怎么办", []) == "那我该怎么办"
    client.chat.completions.create.assert_not_called()


def test_rewrite_query_success(monkeypatch):
    """有历史 → 调用 LLM，temperature=0、max_tokens=200、单条 user 消息。"""
    client = _patch_client(monkeypatch)
    assert query_rewrite.rewrite_query("那我该怎么办", HISTORY) == REWRITTEN
    client.chat.completions.create.assert_called_once()
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] == query_rewrite.REWRITE_MAX_TOKENS
    assert kwargs["messages"][0]["role"] == "user"


def test_rewrite_query_llm_error_falls_back(monkeypatch):
    """LLM 调用抛异常 → 降级返回原始 query。"""

    def boom():
        raise RuntimeError("api down")

    monkeypatch.setattr(query_rewrite, "_client", boom)
    assert query_rewrite.rewrite_query("那我该怎么办", HISTORY) == "那我该怎么办"


def test_rewrite_query_empty_result_falls_back(monkeypatch):
    """LLM 返回空内容 → 降级返回原始 query。"""
    _patch_client(monkeypatch, content="")
    assert query_rewrite.rewrite_query("那我该怎么办", HISTORY) == "那我该怎么办"


def test_get_rewritten_query_cache_hit(monkeypatch):
    """缓存命中 → 直接返回缓存值，不调 LLM。"""
    redis = _patch_redis(monkeypatch, cached="缓存结果")
    client = _patch_client(monkeypatch)
    assert query_rewrite.get_rewritten_query("那我该怎么办", HISTORY, "s1") == "缓存结果"
    client.chat.completions.create.assert_not_called()
    redis.get.assert_called_once()


def test_get_rewritten_query_cache_miss_writes(monkeypatch):
    """缓存未命中 → 改写后写回，key 前缀 rewrite:{session_id}:，TTL 1 小时。"""
    redis = _patch_redis(monkeypatch, cached=None)
    _patch_client(monkeypatch)
    assert query_rewrite.get_rewritten_query("那我该怎么办", HISTORY, "s1") == REWRITTEN
    redis.set.assert_called_once()
    key, value = redis.set.call_args.args[:2]
    assert key.startswith("rewrite:s1:")
    assert value == REWRITTEN
    assert redis.set.call_args.kwargs["ex"] == query_rewrite.REWRITE_CACHE_TTL


def test_get_rewritten_query_empty_history(monkeypatch):
    """第一轮 → 不查缓存、不改写，直接返回原 query。"""
    client = _patch_client(monkeypatch)
    redis = _patch_redis(monkeypatch)
    assert query_rewrite.get_rewritten_query("那我该怎么办", [], "s1") == "那我该怎么办"
    client.chat.completions.create.assert_not_called()
    redis.get.assert_not_called()


def test_rewrite_key_uses_md5():
    """缓存 key 为 rewrite:{session_id}:{md5(query)}，不含原始 query 明文。"""
    key = query_rewrite._rewrite_key("s1", "那我该怎么办")
    assert key.startswith("rewrite:s1:")
    assert "那我该怎么办" not in key
    assert len(key) == len("rewrite:s1:") + 32


def test_format_history_takes_last_n_rounds():
    """_format_history 只取最近 REWRITE_HISTORY_ROUNDS 轮（6 条消息）。"""
    history = []
    for i in range(10):
        history.append({"role": "user", "content": f"问{i}"})
        history.append({"role": "assistant", "content": f"答{i}"})
    text = query_rewrite._format_history(history)
    assert "问0" not in text
    assert "答0" not in text
    assert "问9" in text
    assert "答9" in text
    assert len(text.split("\n")) == query_rewrite.REWRITE_HISTORY_ROUNDS * 2
