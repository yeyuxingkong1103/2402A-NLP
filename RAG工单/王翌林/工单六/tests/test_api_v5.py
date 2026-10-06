# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query 理解优化任务
tests/test_api_v5.py —— 工单五 API v5 接口测试（新增文件）

使用 FastAPI TestClient 验证：health / chat / history / feedback / reset。
chat 端点 mock RAGEngineV4（不依赖真实模型与 Milvus）。
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """工单五：注入 mock 引擎，避免加载真实 RAG/Milvus"""
    from src import api_v5

    class FakeSession:
        def __init__(self):
            self.history = []
            self.current_entity = "武汉兴图新科电子股份有限公司"
            self.current_doc = "招股说明书1"
            self.last_intent = "法定代表人是谁"

        def to_dict(self):
            return {"session_id": "mock_sid", "history": self.history,
                    "current_entity": self.current_entity,
                    "current_doc": self.current_doc}

    class FakeEngine:
        def __init__(self):
            self._sessions = {"mock_sid": FakeSession()}

        def chat(self, question, session_id=None, **kw):
            return {
                "session_id": "mock_sid",
                "resolved_query": question,
                "entity": "武汉兴图新科电子股份有限公司",
                "doc_id": "招股说明书1",
                "is_followup": False,
                "coref_strategy": "direct",
                "mode": "rag_v5", "query": question,
                "route": {}, "answer": "mock answer",
                "references": [], "retrieved_text_chunks": [],
                "retrieved_tables": [], "retrieved_images": [],
                "latency_ms": 100.0, "conversation_latency_ms": 100.0,
                "breakdown": {}, "token_usage": {},
            }

        def get_history(self, sid):
            return self._sessions[sid].to_dict()

        def reset_session(self, sid):
            self._sessions.pop(sid, None)

        def health(self):
            return {"sessions": 1, "milvus_images_rows": 3}

    monkeypatch.setattr(api_v5, "get_engine", lambda: FakeEngine())
    return TestClient(api_v5.app)


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/api/v5/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["engine"] == "conv_v5"
        assert body["sessions"] == 1
        assert body["work_order"] == "人工智能NLP-RAG-Query 理解优化任务"


class TestChat:
    def test_chat_returns_answer(self, client):
        r = client.post("/api/v5/chat",
                        json={"question": "这个公司的法定代表人是谁？"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "mock answer"
        assert body["session_id"] == "mock_sid"
        assert body["entity"] == "武汉兴图新科电子股份有限公司"

    def test_chat_with_session(self, client):
        r = client.post("/api/v5/chat",
                        json={"question": "test", "session_id": "mock_sid"})
        assert r.status_code == 200
        assert r.json()["session_id"] == "mock_sid"

    def test_chat_empty_question(self, client):
        r = client.post("/api/v5/chat", json={"question": ""})
        assert r.status_code == 200  # 空串也走通（由 RAG 层兜底）


class TestHistory:
    def test_get_history(self, client):
        r = client.get("/api/v5/history/mock_sid")
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == "mock_sid"
        assert body["current_entity"] == "武汉兴图新科电子股份有限公司"


class TestFeedback:
    def test_feedback(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = client.post("/api/v5/feedback", json={
            "session_id": "mock_sid", "question": "q", "answer": "a",
            "rating": "up", "comment": "", "latency_ms": 100.0})
        assert r.status_code == 200
        assert r.json()["ok"] is True


class TestReset:
    def test_reset_session(self, client):
        r = client.post("/api/v5/reset/mock_sid")
        assert r.status_code == 200
        assert r.json()["ok"] is True
