import httpx

from app.core.providers.embedding.bge_m3_http import BgeM3Http
from app.core.providers.llm.openai_compat import OpenAICompatLLM
from app.core.providers.rerank.bge_rerank_http import BgeRerankHttp


class _FakeResponse:
    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class _FakeAsyncClient:
    """Mimics httpx.AsyncClient for providers that construct it internally."""
    response = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    async def post(self, url, json=None, headers=None):
        return _FakeAsyncClient.response


async def test_bge_m3_http():
    provider = BgeM3Http("http://x")
    captured = {}

    async def fake_post(payload):
        captured.update(payload)
        return {"dense": [[0.1, 0.2]], "sparse": [{"0": 1.0}]}

    provider._post = fake_post
    assert await provider.encode_dense(["hi"]) == [[0.1, 0.2]]
    assert await provider.encode_sparse(["hi"]) == [{0: 1.0}]
    dq, sq = await provider.encode_query("hi")
    assert dq == [0.1, 0.2]
    assert sq == {0: 1.0}
    assert captured["texts"] == ["hi"]


async def test_bge_rerank_http(monkeypatch):
    _FakeAsyncClient.response = _FakeResponse({"results": [{"index": 2, "score": 0.9}]})
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    p = BgeRerankHttp("http://x")
    res = await p.rerank("q", ["a", "b", "c"], 1)
    assert res == [(2, 0.9)]


async def test_openai_compat_chat(monkeypatch):
    _FakeAsyncClient.response = _FakeResponse({"choices": [{"message": {"content": "你好"}}]})
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    llm = OpenAICompatLLM("http://x", "key", "m")
    out = await llm.chat([{"role": "user", "content": "hi"}])
    assert out == "你好"
