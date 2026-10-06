import logging

from backend.app.services.audit_service import AuditAction, record_audit


def test_record_audit_masks_sensitive_metadata():
    log = record_audit(
        action=AuditAction.DATA_EXPORT,
        actor_id="user-1",
        target_type="user",
        target_id="user-1",
        metadata={"phone": "13800138000", "reason": "user export"},
    )

    assert log.action == "DATA_EXPORT"
    assert "13800138000" not in str(log.metadata)
    assert log.metadata["reason"] == "user export"


def test_record_audit_masks_secret_like_fields():
    log = record_audit(
        action=AuditAction.FULL_CONVERSATION_ACCESS,
        actor_id="admin-1",
        target_type="conversation",
        target_id="conv-1",
        metadata={
            "token": "secret-token-value",
            "prompt": "用户原始提问 13800138000",
            "nested": {"answer": "完整回复内容"},
        },
    )

    metadata_text = str(log.metadata)
    assert "secret-token-value" not in metadata_text
    assert "用户原始提问" not in metadata_text
    assert "完整回复内容" not in metadata_text


def test_record_audit_masks_ciphertext_fields():
    log = record_audit(
        action=AuditAction.DATA_EXPORT,
        actor_id="admin-1",
        target_type="api_key",
        target_id="key-1",
        metadata={
            "encrypted_value": "cipher-value",
            "encrypted_payload": "cipher-payload",
            "encrypted_data_key": "cipher-data-key",
        },
    )

    metadata_text = str(log.metadata)
    assert "cipher-value" not in metadata_text
    assert "cipher-payload" not in metadata_text
    assert "cipher-data-key" not in metadata_text


def test_record_audit_logs_safe_context(caplog):
    with caplog.at_level(logging.INFO, logger="backend.app.services.audit_service"):
        record_audit(
            action=AuditAction.DATA_DELETE,
            actor_id="admin-1",
            target_type="user",
            target_id="user-1",
            metadata={"phone": "13800138000", "reason": "user request"},
        )

    record = caplog.records[-1]
    assert record.action == "DATA_DELETE"
    assert record.actor == "admin-1"
    assert record.target_type == "user"
    assert record.target_id == "user-1"
    assert record.metadata_keys == ["phone", "reason"]
    assert "13800138000" not in record.getMessage()


def test_record_audit_is_append_only():
    first = record_audit(
        action=AuditAction.ADMIN_LOGIN,
        actor_id="admin-1",
        target_type="admin",
        target_id="admin-1",
        metadata={"ip": "127.0.0.1"},
    )
    second = record_audit(
        action=AuditAction.WHITELIST_CHANGE,
        actor_id="admin-1",
        target_type="source_whitelist",
        target_id="source-1",
        metadata={"url": "https://example.gov.cn/law"},
    )

    assert first.id != second.id
    assert first.created_at <= second.created_at
