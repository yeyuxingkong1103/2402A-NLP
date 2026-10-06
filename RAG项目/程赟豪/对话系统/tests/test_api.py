"""API 集成测试（不依赖 Milvus/Redis/大模型，仅覆盖无外部依赖的接口）"""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_role_list():
    resp = client.get("/api/role/list")
    assert resp.status_code == 200
    body = resp.json()
    assert "roles" in body
    assert "available_types" in body
    assert "doctor" in body["available_types"]


def test_index_page():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "RAG 角色扮演" in resp.text
