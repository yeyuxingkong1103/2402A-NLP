"""云端模型接口的边界行为测试，不发出真实网络请求。"""

from types import SimpleNamespace

import httpx
import pytest

from backend.app.models.embedding import EmbeddingClient
from backend.app.models.prompt import build_answer_messages
from backend.app.rag.answer import answer_token_budget
from backend.app.rag.understand import AskRequest
from backend.app.storage.mysql import UserStateStore


@pytest.mark.parametrize("status_code", [401, 403])
def test_embedding_authentication_error_fails_without_retrying(monkeypatch, status_code):
    """密钥失效时继续等待不会成功，应在第一次 401 后立即返回错误。"""
    request = httpx.Request("POST", "https://example.test/v1/embeddings")
    response = httpx.Response(status_code, request=request)
    attempts = 0
    sleeps = []

    def reject_request(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise httpx.HTTPStatusError("unauthorized", request=request, response=response)

    monkeypatch.setattr("backend.app.models.embedding.httpx.post", reject_request)
    monkeypatch.setattr("backend.app.models.embedding.time.sleep", sleeps.append)
    settings = SimpleNamespace(
        siliconflow_api_key="invalid-test-key",
        siliconflow_embedding_base_url="https://example.test/v1",
        embedding_model="test-embedding",
        embedding_batch_size=32,
        embedding_batch_max_chars=60000,
        embedding_cache_size=8,
        embedding_cache_ttl=60,
        embedding_timeout=1,
        embedding_retry_count=5,
        embedding_retry_delay=3,
    )

    with pytest.raises(httpx.HTTPStatusError):
        EmbeddingClient(settings).embed(["合同纠纷怎么办"])

    assert attempts == 1
    assert sleeps == []


def test_embedding_connection_failure_uses_short_configured_retry(monkeypatch):
    attempts = 0
    sleeps = []

    def fail_to_connect(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("network unavailable")

    monkeypatch.setattr("backend.app.models.embedding.httpx.post", fail_to_connect)
    monkeypatch.setattr("backend.app.models.embedding.time.sleep", sleeps.append)
    settings = SimpleNamespace(
        siliconflow_api_key="test-key",
        siliconflow_embedding_base_url="https://example.test/v1",
        embedding_model="test-embedding",
        embedding_batch_size=32,
        embedding_batch_max_chars=60000,
        embedding_cache_size=8,
        embedding_cache_ttl=60,
        embedding_timeout=1,
        embedding_retry_count=1,
        embedding_retry_delay=1,
        embedding_failure_cooldown=30,
    )

    with pytest.raises(httpx.ConnectError):
        EmbeddingClient(settings).embed(["合同纠纷怎么办？"])

    assert attempts == 2
    assert sleeps == [1]


def test_embedding_uses_a_short_connect_timeout(monkeypatch):
    captured = {}

    def successful_request(url, **kwargs):
        captured["timeout"] = kwargs["timeout"]
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={"data": [{"embedding": [0.1, 0.2]}]},
        )

    monkeypatch.setattr("backend.app.models.embedding.httpx.post", successful_request)
    settings = SimpleNamespace(
        siliconflow_api_key="test-key",
        siliconflow_embedding_base_url="https://example.test/v1",
        embedding_model="test-embedding",
        embedding_batch_size=32,
        embedding_batch_max_chars=60000,
        embedding_cache_size=8,
        embedding_cache_ttl=60,
        embedding_timeout=60,
        embedding_connect_timeout=5,
        embedding_retry_count=1,
        embedding_retry_delay=1,
        embedding_failure_cooldown=30,
    )

    EmbeddingClient(settings).embed(["合同纠纷"])

    assert captured["timeout"].connect == 5
    assert captured["timeout"].read == 60


def test_embedding_circuit_breaker_skips_network_during_cooldown(monkeypatch):
    attempts = 0

    def fail_to_connect(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("network unavailable")

    monkeypatch.setattr("backend.app.models.embedding.httpx.post", fail_to_connect)
    monkeypatch.setattr("backend.app.models.embedding.time.sleep", lambda _seconds: None)
    settings = SimpleNamespace(
        siliconflow_api_key="test-key",
        siliconflow_embedding_base_url="https://example.test/v1",
        embedding_model="test-embedding",
        embedding_batch_size=32,
        embedding_batch_max_chars=60000,
        embedding_cache_size=8,
        embedding_cache_ttl=60,
        embedding_timeout=1,
        embedding_retry_count=0,
        embedding_retry_delay=1,
        embedding_failure_cooldown=30,
    )
    client = EmbeddingClient(settings)

    with pytest.raises(httpx.ConnectError):
        client.embed(["第一次提问"])
    with pytest.raises(RuntimeError, match="暂时不可用"):
        client.embed(["第二次提问"])

    assert attempts == 1


def test_new_user_does_not_enable_web_search_by_default():
    settings = UserStateStore(mysql=None).normalize_settings()

    assert settings["include_web_default"] is False


def test_question_api_does_not_enable_web_search_when_option_is_omitted():
    request = AskRequest(query="合同解除需要什么条件？")

    assert request.include_web is False


def test_question_api_accepts_retrieval_mode_and_answer_detail():
    request = AskRequest(
        query="contract termination",
        retrieval_mode="local",
        answer_detail="concise",
    )

    assert request.retrieval_mode == "local"
    assert request.answer_detail == "concise"


def test_answer_detail_changes_prompt_and_token_budget():
    concise = build_answer_messages("loan dispute", "evidence", answer_detail="concise")
    detailed = build_answer_messages("loan dispute", "evidence", answer_detail="detailed")

    assert "不超过 3 项" in concise[-1]["content"]
    assert "对方可能抗辩" in detailed[-1]["content"]
    assert answer_token_budget(2200, "concise") < answer_token_budget(2200, "standard")
    assert answer_token_budget(2200, "standard") < answer_token_budget(2200, "detailed")
