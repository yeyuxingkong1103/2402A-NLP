"""批次 15 任务 2：SiliconFlowEmbeddingClient 通用重试单测。

验收 a 项四场景 + 附加保障：
- 超时后重试成功
- HTTP 429 后重试成功
- HTTP 400 不重试直接抛
- 超过次数后抛
- 总耗时预算耗尽即停（不再浪费重试）
- 指数退避间隔 0.5s → 1s → 2s
- .env 可配置默认重试次数
"""

import urllib.error

import pytest

from app.models.embedding import EmbeddingApiError, SiliconFlowEmbeddingClient

DIM = 4


def make_response(texts: list[str]) -> dict:
    """构造合法 embeddings 响应（维度 DIM）。"""
    return {
        "data": [
            {"index": i, "embedding": [0.1] * DIM} for i in range(len(texts))
        ]
    }


class ScriptedTransport:
    """按脚本逐个抛错 / 返回；记录调用次数。"""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls = 0

    def __call__(self, url, headers, payload, timeout):
        self.calls += 1
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def make_client(script: list, **kwargs):
    transport = ScriptedTransport(script)
    client = SiliconFlowEmbeddingClient(
        api_url="http://embedding.test/v1/embeddings",
        api_key="test-key",
        model="test-model",
        dimension=DIM,
        transport=transport,
        **kwargs,
    )
    return client, transport


def test_retry_after_timeout_succeeds(monkeypatch):
    monkeypatch.setattr("app.models.embedding.time.sleep", lambda _s: None)
    ok = make_response(["你好"])
    client, transport = make_client([TimeoutError("read timed out"), ok])
    vectors = client.embed(["你好"])
    assert len(vectors[0]) == DIM
    assert transport.calls == 2  # 首调失败 + 重试 1 次成功


def test_retry_after_429_succeeds(monkeypatch):
    monkeypatch.setattr("app.models.embedding.time.sleep", lambda _s: None)
    ok = make_response(["你好"])
    client, transport = make_client(
        [EmbeddingApiError("Embedding HTTP 429", status_code=429), ok]
    )
    vectors = client.embed(["你好"])
    assert len(vectors[0]) == DIM
    assert transport.calls == 2


def test_http_400_not_retried(monkeypatch):
    monkeypatch.setattr("app.models.embedding.time.sleep", lambda _s: None)
    client, transport = make_client(
        [EmbeddingApiError("Embedding HTTP 400", status_code=400)]
    )
    with pytest.raises(EmbeddingApiError):
        client.embed(["你好"])
    assert transport.calls == 1  # 不可重试错误：立即抛，不浪费重试


def test_http_500_retried_401_not(monkeypatch):
    monkeypatch.setattr("app.models.embedding.time.sleep", lambda _s: None)
    # 5xx 可重试
    client, transport = make_client(
        [EmbeddingApiError("Embedding HTTP 503", status_code=503), make_response(["你好"])]
    )
    assert len(client.embed(["你好"])[0]) == DIM
    assert transport.calls == 2
    # 401 不可重试
    client2, transport2 = make_client(
        [EmbeddingApiError("Embedding HTTP 401", status_code=401)]
    )
    with pytest.raises(EmbeddingApiError):
        client2.embed(["你好"])
    assert transport2.calls == 1


def test_exhaust_retries_raises(monkeypatch):
    monkeypatch.setattr("app.models.embedding.time.sleep", lambda _s: None)
    script = [TimeoutError("timed out")] * 4  # 1 次首调 + 3 次重试
    client, transport = make_client(script, retry_attempts=3)
    with pytest.raises(EmbeddingApiError):
        client.embed(["你好"])
    assert transport.calls == 4


def test_total_budget_stops_retry(monkeypatch):
    monkeypatch.setattr("app.models.embedding.time.sleep", lambda _s: None)
    script = [TimeoutError("timed out")] * 5
    client, transport = make_client(
        script, retry_attempts=3, retry_total_budget_seconds=0.0
    )
    with pytest.raises(EmbeddingApiError):
        client.embed(["你好"])
    assert transport.calls == 1  # 预算为 0：首调失败即放弃


def test_backoff_is_exponential(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("app.models.embedding.time.sleep", sleeps.append)
    script = [TimeoutError("timed out"), TimeoutError("timed out"), make_response(["你好"])]
    client, transport = make_client(script, retry_backoff_seconds=0.5)
    client.embed(["你好"])
    assert sleeps == [0.5, 1.0]
    assert transport.calls == 3


def test_env_configurable_default_attempts(monkeypatch):
    monkeypatch.setenv("EMBEDDING_RETRY_ATTEMPTS", "2")

    class NoopTransport:
        calls = 0

        def __call__(self, *_args):
            NoopTransport.calls += 1
            return make_response(["你好"])

    client = SiliconFlowEmbeddingClient(
        api_url="u", api_key="k", model="m", dimension=DIM, transport=NoopTransport()
    )
    assert client.retry_attempts == 2


def test_urlopen_timeout_wrapped_as_retryable_timeout(monkeypatch):
    """真实 _request 路径：URLError(reason=超时) 应转成可重试的 TimeoutError。"""
    client = SiliconFlowEmbeddingClient(
        api_url="u", api_key="k", model="m", dimension=DIM, transport=None
    )

    def fake_urlopen(_request, timeout=None):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr("app.models.embedding.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(TimeoutError):
        client._request("http://embedding.test/v1/embeddings", {}, b"", 1.0)
