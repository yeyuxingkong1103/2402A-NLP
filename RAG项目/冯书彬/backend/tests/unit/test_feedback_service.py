import pytest

from backend.app.api.v1.chat import chat_service
from backend.app.core.security import create_access_token
from backend.app.models.message import Message
from backend.app.models.user import User
from backend.app.services.auth_service import get_auth_store, reset_auth_store
from backend.app.services.feedback_service import classify_feedback_risk
from backend.app.services.notification_service import AdminAlert, send_admin_alert


@pytest.fixture(autouse=True)
def feedback_env(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum-value")
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum-value")
    reset_auth_store()
    chat_service.reset_for_tests()


def _create_user(user_id: str) -> None:
    from datetime import datetime

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


def test_wrong_legal_basis_is_high_risk_feedback():
    assert classify_feedback_risk("wrong_legal_basis", "引用了不存在的法条") == "high"


def test_unhelpful_answer_is_normal_feedback():
    assert classify_feedback_risk("not_helpful", "没有解决我的问题") == "normal"


def test_admin_alert_redacts_sensitive_metadata():
    result = send_admin_alert(
        AdminAlert(
            feedback_id="fb-1",
            risk_level="high",
            category="privacy_leak",
            metadata={"phone": "13800138000", "reason": "泄露了手机号"},
        )
    )

    assert result.sent is True
    assert result.channel == "email"
    assert result.metadata["phone"] == "[已脱敏]"
    assert "13800138000" not in str(result.metadata)


def test_submit_feedback_requires_owned_message():
    _create_user("owner")
    chat_service.messages["msg-1"] = Message(id="msg-1", conversation_id="c1", user_id="owner", user_text="问题", answer="回答", status="answered")
    from backend.app.services.feedback_service import FeedbackServiceError, submit_feedback

    with pytest.raises(FeedbackServiceError) as error:
        submit_feedback("attacker", "msg-1", "down", "越权", "not_helpful")

    assert error.value.status_code == 404
