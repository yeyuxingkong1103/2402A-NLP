from datetime import datetime

import pytest

from backend.app.api.v1.chat import chat_service
from backend.app.core.security import create_access_token
from backend.app.models.message import Message
from backend.app.models.user import User
from backend.app.services import memory_service
from backend.app.services.auth_service import get_auth_store, reset_auth_store
from backend.app.services.feedback_service import get_alert, get_feedback, reset_feedback_store_for_tests


@pytest.fixture(autouse=True)
def feedback_api_env(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum-value")
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum-value")
    reset_auth_store()
    reset_feedback_store_for_tests()
    memory_service.reset_memory_store_for_tests()
    chat_service.reset_for_tests()


def _ensure_user(user_id: str) -> None:
    now = datetime.utcnow()
    get_auth_store().save_user(
        User(
            id=user_id,
            encrypted_phone="ciphertext",
            phone_nonce="nonce",
            phone_encrypted_data_key="derived-hkdf-sha256",
            phone_key_version="v1",
            phone_hmac=f"phone-{user_id}",
            agreement_version="v1",
            privacy_policy_version="v1",
            adult_confirmed=True,
            created_at=now,
            updated_at=now,
        )
    )


def _headers(user_id: str) -> dict[str, str]:
    _ensure_user(user_id)
    token = create_access_token(user_id, "session-test")
    return {"Authorization": f"Bearer {token}"}


def test_submit_negative_feedback_with_reason(client):
    _ensure_user("user-1")
    chat_service.messages["msg-1"] = Message(id="msg-1", conversation_id="c1", user_id="user-1", user_text="问题", answer="回答", status="answered")

    response = client.post(
        "/api/v1/feedback",
        headers=_headers("user-1"),
        json={"message_id": "msg-1", "rating": "down", "category": "wrong_legal_basis", "reason": "引用了不存在的法条"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["risk_level"] == "high"
    assert body["alert_id"]
    assert get_feedback(body["id"]).reason == "引用了不存在的法条"
    assert get_alert(body["alert_id"]).notification_sent is True


def test_submit_feedback_requires_authentication(client):
    response = client.post(
        "/api/v1/feedback",
        json={"message_id": "msg-1", "rating": "down", "category": "not_helpful", "reason": "没有解决问题"},
    )

    assert response.status_code == 401


def test_submit_feedback_rejects_cross_user_message(client):
    _ensure_user("owner")
    chat_service.messages["msg-1"] = Message(id="msg-1", conversation_id="c1", user_id="owner", user_text="问题", answer="回答", status="answered")

    response = client.post(
        "/api/v1/feedback",
        headers=_headers("attacker"),
        json={"message_id": "msg-1", "rating": "down", "category": "not_helpful", "reason": "无权"},
    )

    assert response.status_code == 404


def test_submit_feedback_rejects_unanswered_message_and_long_reason(client):
    chat_service.messages["msg-1"] = Message(id="msg-1", conversation_id="c1", user_id="user-1", user_text="问题")

    pending = client.post(
        "/api/v1/feedback",
        headers=_headers("user-1"),
        json={"message_id": "msg-1", "rating": "down", "category": "not_helpful"},
    )
    too_long = client.post(
        "/api/v1/feedback",
        headers=_headers("user-1"),
        json={"message_id": "msg-1", "rating": "down", "reason": "x" * 501},
    )

    assert pending.status_code == 409
    assert too_long.status_code == 422
