"""api.py 集成测试：TestClient + mock rag.ask / rag.ask_stream（不调 DeepSeek）。"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api
import rag


@pytest.fixture
def client(monkeypatch):
    """关闭 lifespan 预热的真实模型加载，返回可用的 TestClient。"""
    monkeypatch.setattr(rag, "load_embedder", lambda: None)
    monkeypatch.setattr(rag, "load_reranker", lambda: None)
    monkeypatch.setattr(api.roles, "load_roles", lambda: None)
    with TestClient(api.app) as c:
        yield c


def _mock_ask(monkeypatch):
    """mock rag.ask，返回固定 answer + sources，并返回 mock 供断言调用参数。"""
    ask = MagicMock(return_value={
        "answer": "血压偏高需注意饮食。",
        "sources": [{"content": "指南片段", "page": 3, "similarity": 0.9}],
    })
    monkeypatch.setattr(rag, "ask", ask)
    return ask


def _mock_stream(monkeypatch):
    """mock rag.ask_stream，返回固定 SSE 生成器，并返回 mock 供断言调用参数。"""
    def fake_stream(message, session_id):
        yield "event: sources\ndata: []\n\n"
        yield "data: token\n\n"
        yield "data: [DONE]\n\n"

    stream = MagicMock(side_effect=fake_stream)
    monkeypatch.setattr(rag, "ask_stream", stream)
    return stream


def test_health_returns_ok(client):
    """GET /health 返回 200 与 {"status": "ok"}。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_returns_answer_and_sources(client, monkeypatch):
    """正常请求返回 200，body 含 answer 与非空 sources。"""
    _mock_ask(monkeypatch)
    resp = client.post("/chat", json={"message": "血压多少算高？"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "血压偏高需注意饮食。"
    assert isinstance(data["sources"], list) and data["sources"]


def test_chat_empty_message_422(client):
    """message 为空串 → 422（Pydantic min_length=1）。"""
    resp = client.post("/chat", json={"message": ""})
    assert resp.status_code == 422


def test_chat_whitespace_message_422(client):
    """message 为纯空白 → 422（手动 strip 校验）。"""
    resp = client.post("/chat", json={"message": "   "})
    assert resp.status_code == 422


def test_chat_missing_session_id_uses_default(client, monkeypatch):
    """缺 session_id 时以 "default" 调用 rag.ask。"""
    ask = _mock_ask(monkeypatch)
    client.post("/chat", json={"message": "hi"})
    ask.assert_called_once_with("hi", "default")


def test_chat_passes_explicit_session_id(client, monkeypatch):
    """显式 session_id 原样透传给 rag.ask。"""
    ask = _mock_ask(monkeypatch)
    client.post("/chat", json={"message": "hi", "session_id": "u123"})
    ask.assert_called_once_with("hi", "u123")


def test_chat_stream_content_type(client, monkeypatch):
    """POST /chat/stream 返回 200，Content-Type 为 text/event-stream。"""
    _mock_stream(monkeypatch)
    resp = client.post("/chat/stream", json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")


def test_chat_stream_event_sequence(client, monkeypatch):
    """流式响应按 event: sources → data: → data: [DONE] 顺序。"""
    _mock_stream(monkeypatch)
    body = client.post("/chat/stream", json={"message": "hi"}).text
    assert body.index("event: sources") < body.index("data: token") < body.index("data: [DONE]")


def test_chat_stream_missing_session_id_uses_default(client, monkeypatch):
    """缺 session_id 时以 "default" 调用 rag.ask_stream。"""
    stream = _mock_stream(monkeypatch)
    client.post("/chat/stream", json={"message": "hi"})
    stream.assert_called_once_with("hi", "default")
