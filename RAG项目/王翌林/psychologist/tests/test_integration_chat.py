"""集成测试：问答主链路（非流式 / SSE 流式 / 危机注入 / 会话隔离与删除）。

说明：
1. LLM 由 conftest 全局屏蔽；检索层在本文件内 monkeypatch（不加载 BGE-M3、不连 Milvus）。
2. MySQL 必须可用；Redis 不强依赖（并发锁与短期记忆均有降级路径）。
"""
import json

import pytest


@pytest.fixture(autouse=True)
def stub_retrieval(monkeypatch):
    """屏蔽真实检索（G3 后 Query 改写在 services 层），返回空知识片段。"""
    from src.services import retrieval_service

    def _fake(persona_id, question, history=None, top_k=None, rerank_top_n=None):
        return [], question

    monkeypatch.setattr(retrieval_service, "search_with_query_rewrite", _fake)


def _persona_id(personas_by_code) -> int:
    for code in ("cbt_chen", "humanistic_lin", "mindfulness_zhou"):
        if code in personas_by_code:
            return personas_by_code[code]["id"]
    pytest.skip("三个心理医生角色均未就绪，请先初始化角色")


def _message_items(resp_json: dict) -> list:
    data = resp_json["data"]
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    return data


def test_chat_non_stream_flow(client, test_user, personas_by_code):
    persona_id = _persona_id(personas_by_code)
    resp = client.post("/api/v1/chat", headers=test_user["headers"], json={
        "persona_id": persona_id, "message": "最近工作压力很大，晚上总是睡不着。",
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["answer"].strip()
    assert data["conversation_id"] > 0
    assert data["persona_id"] == persona_id
    assert data["crisis_detected"] is False
    assert data["references"] == []

    msgs = client.get(f"/api/v1/conversations/{data['conversation_id']}/messages",
                      headers=test_user["headers"])
    assert msgs.status_code == 200, msgs.text
    roles = [m["role"] for m in _message_items(msgs.json())]
    assert roles == ["user", "assistant"]


def test_chat_crisis_injection(client, test_user, personas_by_code):
    persona_id = _persona_id(personas_by_code)
    resp = client.post("/api/v1/chat", headers=test_user["headers"], json={
        "persona_id": persona_id, "message": "我不想活了，觉得一切都没有意义。",
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["crisis_detected"] is True
    assert data["crisis_notice"] and "12356" in data["crisis_notice"]
    assert "12356" in data["answer"]


def test_chat_stream_sse(client, test_user, personas_by_code):
    persona_id = _persona_id(personas_by_code)
    resp = client.post("/api/v1/chat/stream", headers=test_user["headers"], json={
        "persona_id": persona_id, "message": "我最近情绪低落，可以聊聊吗？",
    })
    assert resp.status_code == 200, resp.text
    body = resp.text
    assert "event: meta" in body and "event: delta" in body and "event: done" in body

    first_data = next(ln for ln in body.splitlines() if ln.startswith("data: "))
    meta = json.loads(first_data[len("data: "):])
    conv_id = meta["conversation_id"]
    assert conv_id > 0
    assert meta["persona_id"] == persona_id

    msgs = client.get(f"/api/v1/conversations/{conv_id}/messages", headers=test_user["headers"])
    assert msgs.status_code == 200, msgs.text
    assert [m["role"] for m in _message_items(msgs.json())] == ["user", "assistant"]


def test_conversation_isolation_between_users(client, test_user, other_user, personas_by_code):
    persona_id = _persona_id(personas_by_code)
    created = client.post("/api/v1/chat", headers=test_user["headers"], json={
        "persona_id": persona_id, "message": "这条消息只属于 test_user。",
    })
    conv_id = created.json()["data"]["conversation_id"]

    forbidden = client.get(f"/api/v1/conversations/{conv_id}/messages",
                           headers=other_user["headers"])
    assert forbidden.status_code == 403

    delete_forbidden = client.delete(f"/api/v1/conversations/{conv_id}",
                                     headers=other_user["headers"])
    assert delete_forbidden.status_code == 403


def test_delete_conversation(client, test_user, personas_by_code):
    persona_id = _persona_id(personas_by_code)
    created = client.post("/api/v1/chat", headers=test_user["headers"], json={
        "persona_id": persona_id, "message": "会话将被删除。",
    })
    conv_id = created.json()["data"]["conversation_id"]

    deleted = client.delete(f"/api/v1/conversations/{conv_id}", headers=test_user["headers"])
    assert deleted.status_code == 200

    gone = client.get(f"/api/v1/conversations/{conv_id}/messages", headers=test_user["headers"])
    assert gone.status_code == 404
