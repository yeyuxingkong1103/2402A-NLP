"""集成测试：需要后端在 8000 端口运行；未运行时自动跳过。"""
import os

import pytest
import requests

BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


def _server_up() -> bool:
    try:
        return requests.get(f"{BASE}/health", timeout=3).status_code == 200
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason="后端未启动")


def test_health():
    resp = requests.get(f"{BASE}/health", timeout=10)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_roles():
    resp = requests.get(f"{BASE}/roles", timeout=10)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_chat():
    resp = requests.post(
        f"{BASE}/chat",
        json={"user_id": "test", "role_id": 1, "message": "猎捕国家重点保护野生动物怎么处罚？"},
        timeout=120,
    )
    assert resp.status_code == 200
    assert len(resp.json()["reply"]) > 10


def test_clear():
    resp = requests.post(f"{BASE}/clear", params={"user_id": "test", "role_id": 1}, timeout=10)
    assert resp.status_code == 200


def test_ops_redis_stats():
    resp = requests.get(f"{BASE}/ops/redis/stats", timeout=10)
    assert resp.status_code == 200
    assert "dbsize" in resp.json()


def test_ops_milvus_stats():
    resp = requests.get(f"{BASE}/ops/milvus/stats", timeout=30)
    assert resp.status_code == 200
    assert "num_entities" in resp.json()
