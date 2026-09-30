import io
import json
import zipfile
from datetime import datetime

import pytest

from backend.app.api.v1.chat import chat_service
from backend.app.core.crypto import EncryptedValue, decrypt_text
from backend.app.core.security import create_access_token
from backend.app.models.conversation import Conversation
from backend.app.models.message import Message
from backend.app.models.user import User
from backend.app.services import memory_service
from backend.app.services.auth_service import get_auth_store, reset_auth_store
from backend.app.services.export_service import claim_export_password, download_export_job, get_export_job, reset_export_jobs_for_tests
from backend.app.services.feedback_service import get_alert, get_feedback, reset_feedback_store_for_tests, submit_feedback
from backend.app.services.memory_service import create_memory_from_conversation


@pytest.fixture(autouse=True)
def user_data_env(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum-value")
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum-value")
    memory_service.reset_memory_store_for_tests()
    reset_export_jobs_for_tests()
    reset_auth_store()
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


def _create_owned_conversation(user_id: str, conversation_id: str = "c1") -> None:
    conversation = Conversation(id=conversation_id, user_id=user_id)
    message = Message(
        id=f"msg-{conversation_id}",
        conversation_id=conversation_id,
        user_id=user_id,
        user_text="我的手机号是13800138000，想咨询离婚财产问题",
        answer="已记录手机号13800138000并生成回答",
        status="answered",
    )
    conversation.message_ids.append(message.id)
    chat_service.conversations[conversation.id] = conversation
    chat_service.messages[message.id] = message


def _create_export(client, user_id: str) -> dict:
    challenge = client.post(f"/api/v1/users/{user_id}/export/challenge", headers=_auth_header(user_id))
    assert challenge.status_code == 200
    token = challenge.json()["second_factor_token"]
    response = client.post(f"/api/v1/users/{user_id}/export", json={"second_factor_token": token}, headers=_auth_header(user_id))
    assert response.status_code == 200
    return response.json()


def _claim_export_password(client, user_id: str, job_id: str) -> str:
    challenge = client.post(f"/api/v1/users/{user_id}/export/challenge", headers=_auth_header(user_id))
    assert challenge.status_code == 200
    token = challenge.json()["second_factor_token"]
    response = client.post(f"/api/v1/users/{user_id}/export/{job_id}/password", json={"second_factor_token": token}, headers=_auth_header(user_id))
    assert response.status_code == 200
    return response.json()["zip_password"]


def _read_payload_from_download(body: dict) -> dict:
    encrypted = EncryptedValue(**body["encrypted_zip"])
    plaintext = decrypt_text(encrypted, purpose="user-export")
    with zipfile.ZipFile(io.BytesIO(plaintext.encode("latin1"))) as archive:
        return json.loads(archive.read("data.json").decode("utf-8"))


def test_export_excludes_internal_prompt_and_audit_logs(client):
    create_memory_from_conversation("user-1", "c1", {"case_type": "divorce"}, "离婚案件摘要")
    _create_owned_conversation("user-1", "c1")

    export_body = _create_export(client, "user-1")
    job = get_export_job(export_body["id"])
    download = client.get(f"/api/v1/users/user-1/export/{job.id}/download", headers=_auth_header("user-1"))

    assert download.status_code == 200
    download_body = download.json()
    assert "encrypted_zip" in download_body
    assert download_body["password_delivery"] == "claim_required"
    assert "zip_password" not in download_body
    assert _claim_export_password(client, "user-1", job.id)
    with pytest.raises(zipfile.BadZipFile):
        zipfile.ZipFile(io.BytesIO(json.dumps(download_body["encrypted_zip"]).encode("utf-8")))
    payload = _read_payload_from_download(download_body)
    assert payload["conversations"][0]["id"] == "c1"
    assert payload["messages"][0]["user_text"] == "我的手机号是[手机号]，想咨询离婚财产问题"
    assert payload["messages"][0]["answer"] == "已记录手机号[手机号]并生成回答"
    assert "audit_logs" not in payload
    assert "internal_prompts" not in payload
    assert export_body["max_downloads"] == 3
    assert not hasattr(job, "zip_password")
    assert job.password_hmac


def test_export_password_requires_second_factor_and_single_claim(client):
    body = _create_export(client, "user-1")

    missing_token = client.post(f"/api/v1/users/user-1/export/{body['id']}/password", json={}, headers=_auth_header("user-1"))
    assert missing_token.status_code == 403
    assert _claim_export_password(client, "user-1", body["id"])
    second_challenge = client.post("/api/v1/users/user-1/export/challenge", headers=_auth_header("user-1"))
    denied = client.post(
        f"/api/v1/users/user-1/export/{body['id']}/password",
        json={"second_factor_token": second_challenge.json()["second_factor_token"]},
        headers=_auth_header("user-1"),
    )
    assert denied.status_code == 403


def test_export_requires_server_second_factor_token(client):
    response = client.post("/api/v1/users/user-1/export", json={"second_factor_verified": True}, headers=_auth_header("user-1"))

    assert response.status_code == 403


def test_user_data_api_rejects_cross_user_access(client):
    response = client.get("/api/v1/users/user-2/memories", headers=_auth_header("user-1"))

    assert response.status_code == 403


def test_export_download_enforces_owner_and_download_limit(client):
    body = _create_export(client, "user-1")

    denied = client.get(f"/api/v1/users/user-2/export/{body['id']}/download", headers=_auth_header("user-2"))
    assert denied.status_code == 403

    assert _claim_export_password(client, "user-1", body["id"])
    for _ in range(3):
        response = client.get(f"/api/v1/users/user-1/export/{body['id']}/download", headers=_auth_header("user-1"))
        assert response.status_code == 200
        assert response.json()["password_delivery"] == "claim_required"
        assert "zip_password" not in response.json()
    limited = client.get(f"/api/v1/users/user-1/export/{body['id']}/download", headers=_auth_header("user-1"))
    assert limited.status_code == 403


def test_export_download_rejects_revoked_job(client):
    body = _create_export(client, "user-1")
    job = get_export_job(body["id"])
    job.revoked = True

    with pytest.raises(PermissionError):
        download_export_job(job.id, "user-1")


def test_delete_account_removes_linkable_memory_and_rejects_old_token(client):
    create_memory_from_conversation("user-1", "c1", {"case_type": "rent"}, "租赁摘要")
    _create_owned_conversation("user-1", "c1")
    feedback = submit_feedback("user-1", "msg-c1", "down", "引用了不存在的法条", "wrong_legal_basis")
    headers = _auth_header("user-1")
    export_body = _create_export(client, "user-1")

    response = client.request("DELETE", "/api/v1/users/user-1", json={"confirmed": True}, headers=headers)

    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    assert memory_service.list_user_memories("user-1") == []
    assert get_feedback(feedback.id) is None
    assert get_alert(feedback.alert_id) is None
    with pytest.raises(KeyError):
        get_export_job(export_body["id"])
    with pytest.raises(PermissionError):
        download_export_job(export_body["id"], "user-1")

    denied = client.get("/api/v1/users/user-1/memories", headers=headers)
    assert denied.status_code == 401
