"""集成测试（integration）：核心接口——/personas、/conversations、/users/me。"""
import pytest

pytestmark = pytest.mark.integration


# ---------------- /personas ----------------
def test_list_personas_returns_three(client, test_user):
    resp = client.get("/api/v1/personas", headers=test_user["headers"])
    assert resp.status_code == 200
    items = resp.json()["data"]["items"]
    assert len(items) >= 3
    codes = {i["persona_code"] for i in items}
    assert {"cbt_chen", "humanistic_lin", "mindfulness_zhou"} <= codes


def test_persona_detail_fields(client, test_user, personas_by_code):
    pid = personas_by_code["cbt_chen"]["id"]
    resp = client.get(f"/api/v1/personas/{pid}", headers=test_user["headers"])
    assert resp.status_code == 200
    detail = resp.json()["data"]
    assert detail["persona_code"] == "cbt_chen"
    assert detail["greeting"].strip()
    assert "我不会提供" in detail["system_prompt"] or "诊断" in detail["system_prompt"]


def test_persona_404(client, test_user):
    resp = client.get("/api/v1/personas/99999999", headers=test_user["headers"])
    assert resp.status_code == 404


# ---------------- /conversations ----------------
def test_conversation_create_list_messages_delete(client, test_user, personas_by_code):
    pid = personas_by_code["humanistic_lin"]["id"]

    created = client.post("/api/v1/conversations", headers=test_user["headers"],
                          json={"persona_id": pid, "title": "pytest 会话"})
    assert created.status_code == 200, created.text
    conv = created.json()["data"]
    cid = conv["id"]
    assert conv["persona_id"] == pid

    listed = client.get("/api/v1/conversations", headers=test_user["headers"])
    ids = [c["id"] for c in listed.json()["data"]["items"]]
    assert cid in ids

    filtered = client.get(f"/api/v1/conversations?persona_id={pid}", headers=test_user["headers"])
    assert all(c["persona_id"] == pid for c in filtered.json()["data"]["items"])

    messages = client.get(f"/api/v1/conversations/{cid}/messages", headers=test_user["headers"])
    assert messages.status_code == 200
    payload = messages.json()["data"]
    items = payload["items"] if isinstance(payload, dict) else payload
    assert items == []                              # 新会话无消息

    assert client.delete(f"/api/v1/conversations/{cid}",
                         headers=test_user["headers"]).status_code == 200
    assert client.get(f"/api/v1/conversations/{cid}",
                      headers=test_user["headers"]).status_code == 404


def test_conversation_invalid_persona_404(client, test_user):
    resp = client.post("/api/v1/conversations", headers=test_user["headers"],
                       json={"persona_id": 99999999})
    assert resp.status_code == 404


# ---------------- /users/me ----------------
def test_users_me(client, test_user):
    resp = client.get("/api/v1/users/me", headers=test_user["headers"])
    assert resp.status_code == 200
    me = resp.json()["data"]
    assert me["username"] == test_user["username"]
    assert "id" in me and "created_at" in me


def test_users_me_requires_token(client):
    assert client.get("/api/v1/users/me").status_code == 401
