"""接口冒烟测试（独立脚本，需后端运行）。

用法：``python test_rag.py``
"""
import os

import requests

BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


def test_health():
    resp = requests.get(f"{BASE}/health", timeout=10)
    assert resp.status_code == 200, f"health 失败: {resp.status_code}"
    print("✅ /health:", resp.json())


def test_roles():
    resp = requests.get(f"{BASE}/roles", timeout=10)
    assert resp.status_code == 200, f"roles 失败: {resp.status_code}"
    roles = resp.json()
    print(f"✅ /roles: 共 {len(roles)} 个角色")
    return roles


def test_chat(role_id=1):
    resp = requests.post(
        f"{BASE}/chat",
        json={"user_id": "smoke", "role_id": role_id, "message": "猎捕国家重点保护野生动物怎么处罚？"},
        timeout=120,
    )
    assert resp.status_code == 200, f"chat 失败: {resp.status_code}"
    reply = resp.json()["reply"]
    assert len(reply) > 20, "回复太短"
    print("✅ /chat 回复前 80 字:", reply[:80], "...")


def test_stream(role_id=1):
    with requests.post(
        f"{BASE}/chat_stream",
        json={"user_id": "smoke", "role_id": role_id, "message": "简单打个招呼"},
        stream=True, timeout=120,
    ) as resp:
        assert resp.status_code == 200
        got = "".join(chunk for chunk in resp.iter_content(chunk_size=None, decode_unicode=True) if chunk)
    print("✅ /chat_stream 收到", len(got), "字符")


def test_clear():
    resp = requests.post(f"{BASE}/clear", params={"user_id": "smoke", "role_id": 1}, timeout=10)
    assert resp.status_code == 200, f"clear 失败: {resp.status_code}"
    print("✅ /clear 通过")


def test_ops():
    r1 = requests.get(f"{BASE}/ops/redis/stats", timeout=10)
    r2 = requests.get(f"{BASE}/ops/milvus/stats", timeout=30)
    assert r1.status_code == 200 and r2.status_code == 200
    print("✅ /ops/redis/stats 通过")
    print("✅ /ops/milvus/stats 通过")


if __name__ == "__main__":
    print("=== 开始测试 ===")
    test_health()
    test_roles()
    test_chat()
    test_stream()
    test_clear()
    test_ops()
    print("=== 全部通过 ✅ ===")
