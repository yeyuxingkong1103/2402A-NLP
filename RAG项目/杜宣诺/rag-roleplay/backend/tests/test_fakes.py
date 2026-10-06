import pytest
from app.core.fakes import FakeLLM, FakeEmbedding, FakeRerank


async def test_fake_llm_echo():
    llm = FakeLLM()
    out = await llm.chat([{"role": "user", "content": "你好"}])
    assert out == "echo: 你好"


async def test_fake_llm_chat_stream_is_async_generator():
    llm = FakeLLM()
    chunks = [c async for c in llm.chat_stream([{"role": "user", "content": "你好"}])]
    assert "".join(chunks) == "echo: 你好"


async def test_fake_embedding_shape():
    emb = FakeEmbedding()
    dense = await emb.encode_dense(["abc", "abcdef"])
    assert len(dense) == 2
    assert len(dense[0]) == 1024


async def test_fake_rerank_orders_by_length():
    rr = FakeRerank()
    res = await rr.rerank("q", ["x", "xxxxx", "xx"], 2)
    assert res[0][0] == 1  # 最长的 "xxxxx" 排第一
    assert len(res) == 2  # top_m 截断生效
    assert res[1][0] == 2  # 第二长 "xx" 排第二
