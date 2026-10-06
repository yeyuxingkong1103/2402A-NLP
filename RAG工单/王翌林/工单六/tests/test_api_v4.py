# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
tests/test_api_v4.py —— 工单四 API v4 单测（TestClient + Mock 引擎，不依赖模型/Milvus）
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

ASK_RESULT = {
    "mode": "rag_v4", "query": "q", "route": {"mode": "image_first"},
    "answer": "销售部由4个部门构成，大客户销售部下设6个销售处。",
    "references": [{"type": "image", "ref_id": "图1", "image_id": "img_008",
                    "path": "p.png", "score": 0.8}],
    "retrieved_text_chunks": [], "retrieved_tables": [],
    "retrieved_images": [{"image_id": "img_008"}],
    "latency_ms": 123.4, "breakdown": {"retrieve_ms": 20.0, "llm_ms": 100.0},
    "token_usage": {},
}


@pytest.fixture()
def client(monkeypatch):
    """工单四：TestClient + Mock RAGEngineV4（隔离模型与 Milvus）"""
    import src.api_v4 as m
    fake = MagicMock()
    fake.ask.return_value = ASK_RESULT
    fake.ask_llm.return_value = {**ASK_RESULT, "mode": "pure_llm",
                                 "answer": "纯LLM", "references": [],
                                 "retrieved_images": []}
    fake.health.return_value = {"milvus_images_rows": 3}
    monkeypatch.setattr(m, "get_engine", lambda: fake)
    return TestClient(m.app)


def test_health(client):
    """工单四：健康检查含工单编号与图像库行数"""
    r = client.get("/api/v4/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["engine"] == "rag_v4"
    assert body["milvus_images_rows"] == 3
    assert "人工智能NLP-RAG-图像内容解析及检索优化" in body["work_order"]


def test_ask_v4_contract(client):
    """工单四：/ask 契约——answer/references/retrieved_images/latency_ms"""
    r = client.post("/api/v4/ask", json={
        "question": "武汉力源组织结构图中，销售部有几个部门构成？",
        "doc_id": "招股说明书2", "use_image": True, "use_rag": True})
    assert r.status_code == 200
    body = r.json()
    for k in ("answer", "references", "retrieved_images",
              "retrieved_text_chunks", "retrieved_tables", "latency_ms"):
        assert k in body
    assert "4个部门" in body["answer"] and "6个销售处" in body["answer"]
    assert body["references"][0]["type"] == "image"


def test_ask_use_rag_false(client):
    """工单四：use_rag=False 走纯 LLM 模式"""
    r = client.post("/api/v4/ask", json={"question": "你好", "use_rag": False})
    assert r.status_code == 200 and r.json()["mode"] == "pure_llm"


def test_ask_engine_error_500(client, monkeypatch):
    """工单四：引擎异常 → 500 且带错误信息（容错规范）"""
    import src.api_v4 as m
    fake = MagicMock()
    fake.ask.side_effect = RuntimeError("boom")
    monkeypatch.setattr(m, "get_engine", lambda: fake)
    r = client.post("/api/v4/ask", json={"question": "x"})
    assert r.status_code == 500 and "问答失败" in r.json()["detail"]


def test_evaluate_endpoint(client):
    """工单四：/evaluate 批量评估返回 accuracy 与逐题命中"""
    r = client.post("/api/v4/evaluate", json={"questions": [
        {"id": 5, "question": "组织结构图 销售部", "doc_id": "招股说明书2",
         "expected_keywords": ["4个部门", "6个销售处"], "qtype": "image"}],
        "use_rag": True})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1 and body["correct"] == 1
    assert body["accuracy"] == 1.0
    assert body["items"][0]["hit_keywords"] == ["4个部门", "6个销售处"]


def test_images_endpoint_empty(monkeypatch):
    """工单四：/images/{doc_id} 在无 collection 时返回空列表（容错）"""
    import src.api_v4 as m

    class FakeStore:
        collection = "rag_images"

        class _c:
            @staticmethod
            def has_collection(name):
                return False

        client = _c()

    monkeypatch.setattr("src.image_parser.image_store.ImageStore",
                        lambda **kw: FakeStore())
    c = TestClient(m.app)
    r = c.get("/api/v4/images/招股说明书2")
    assert r.status_code == 200 and r.json()["total"] == 0
