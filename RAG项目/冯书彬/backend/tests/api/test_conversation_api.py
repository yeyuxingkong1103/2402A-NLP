from datetime import datetime

import pytest

from backend.app.api.v1.chat import chat_service
from backend.app.core.security import create_access_token
from backend.app.models.conversation import Conversation
from backend.app.models.user import User
from backend.app.services.auth_service import get_auth_store, reset_auth_store
from backend.app.services import memory_service
from backend.app.services.memory_service import create_memory_from_conversation, list_user_memories


@pytest.fixture(autouse=True)
def conversation_api_env(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum-value")
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum-value")
    reset_auth_store()
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


def _auth_header(user_id: str) -> dict[str, str]:
    _ensure_user(user_id)
    token = create_access_token(user_id, "session-test")
    return {"Authorization": f"Bearer {token}"}


def test_delete_conversation_api_uses_authenticated_user(client):
    chat_service.conversations["c1"] = Conversation(id="c1", user_id="owner")
    memory = create_memory_from_conversation("owner", "c1", {"case_type": "divorce"}, "离婚摘要")

    denied = client.request("DELETE", "/api/v1/conversations/c1", headers=_auth_header("attacker"))

    assert denied.status_code == 403
    assert "c1" in chat_service.conversations
    assert [item.id for item in list_user_memories("owner")] == [memory.id]

    allowed = client.request("DELETE", "/api/v1/conversations/c1", headers=_auth_header("owner"))
    assert allowed.status_code == 200
    assert allowed.json()["deleted_memory_ids"] == [memory.id]
