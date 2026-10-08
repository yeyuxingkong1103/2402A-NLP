# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
tests/test_api_v6.py —— 工单六 API v6 接口测试（mock 引擎，不依赖模型）
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from src import api_v6

    class FakeEngine:
        def health(self):
            return {"milvus_images_rows": 3, "fulltext_docs": 1141}

        def ask(self, question, doc_id=None, company=None,
                lang_override=None, retrieval_config=None, **kw):
            return {
                "mode": "rag_v6", "query": question, "route": {},
                "answer": f"mock:{question}", "references": [],
                "retrieved_text_chunks": [], "retrieved_tables": [],
                "retrieved_images": [], "latency_ms": 120.0,
                "breakdown": {"retrieve_ms": 80.0, "llm_ms": 40.0},
                "retrieval": (retrieval_config or {}),
                "token_usage": {},
            }

        def ask_llm(self, question):
            return {"mode": "pure_llm", "query": question, "answer": "llm",
                    "references": [], "latency_ms": 50.0, "breakdown": {},
                    "retrieved_text_chunks": [], "retrieved_tables": [],
                    "retrieved_images": [], "route": {}, "token_usage": {}}

    class FakeSession:
        def __init__(self):
            self.sid = "s1"

        def chat(self, question, session_id=None, **kw):
            r = FakeEngine().ask(question, **kw)
            r.update({"session_id": "s1", "resolved_query": question,
                      "entity": "测试公司", "is_followup": False,
                      "coref_strategy": "direct"})
            return r

        def get_history(self, sid):
            return {"session_id": sid, "history": [],
                    "current_entity": None, "current_doc": None}

        def reset_session(self, sid):
            return None

    monkeypatch.setattr(api_v6, "get_engine", lambda: FakeEngine())
    monkeypatch.setattr(api_v6, "get_conv", lambda: FakeSession())
    return TestClient(api_v6.app)


class TestHealth:
    def test_health(self, client):
        body = client.get("/api/v6/health").json()
        assert body["status"] == "ok"
        assert body["fulltext_docs"] == 1141
        assert body["work_order"] == "人工智能NLP-RAG-混合检索任务"


class TestConfig:
    def test_config_options(self, client):
        body = client.get("/api/v6/config").json()
        assert set(body["modes"]) == {"vector", "fulltext", "hybrid"}
        assert set(body["fusions"]) == {"rrf", "weighted"}
        assert set(body["rerankers"]) == {"llm", "tfidf", "adaptive"}
        assert "and" in body["matches"] and "fuzzy" in body["matches"]
        assert len(body["presets"]) >= 3
        assert body["defaults"]["mode"] == "hybrid"


class TestAsk:
    def test_ask_default(self, client):
        body = client.post("/api/v6/ask",
                           json={"question": "军用收入"}).json()
        assert body["answer"] == "mock:军用收入"
        assert body["mode"] == "rag_v6"

    def test_ask_with_retrieval_config(self, client):
        body = client.post("/api/v6/ask", json={
            "question": "军用收入",
            "retrieval": {"mode": "fulltext", "reranker": "tfidf",
                          "match": "or", "fusion": "weighted",
                          "vector_weight": 0.3, "fulltext_weight": 0.7},
        }).json()
        rt = body["retrieval"]
        assert rt["mode"] == "fulltext"
        assert rt["reranker"] == "tfidf"
        assert rt["match"] == "or"


class TestChat:
    def test_chat_returns_session(self, client):
        body = client.post("/api/v6/chat",
                           json={"question": "法定代表人是谁"}).json()
        assert body["session_id"] == "s1"
        assert body["entity"] == "测试公司"

    def test_history(self, client):
        body = client.get("/api/v6/history/s1").json()
        assert body["session_id"] == "s1"

    def test_reset(self, client):
        assert client.post("/api/v6/reset/s1").json()["ok"] is True

    def test_feedback(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = client.post("/api/v6/feedback", json={
            "session_id": "s1", "query": "q", "answer": "a",
            "rating": "up"})
        assert r.json()["ok"] is True
