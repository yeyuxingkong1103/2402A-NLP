import pytest

from backend.app.services.model_log_service import ModelCallMetadata, record_model_call, sanitize_model_metadata


def test_model_log_metadata_excludes_prompt_and_answer():
    metadata = ModelCallMetadata(
        provider="deepseek",
        model="deepseek-chat",
        prompt_version="v1",
        knowledge_base_version="kb-1",
        token_count=120,
        duration_ms=300,
        status="success",
        error_type=None,
        trace_id="trace-1",
        prompt="完整用户问题不应记录",
        answer="完整回答不应记录",
    )

    sanitized = sanitize_model_metadata(metadata)

    assert "prompt" not in sanitized
    assert "answer" not in sanitized


def test_model_log_redacts_pii_and_records_flags():
    metadata = ModelCallMetadata(
        provider="deepseek",
        model="deepseek-chat",
        prompt_version="v1",
        knowledge_base_version="kb-1",
        token_count=20,
        duration_ms=15,
        status="failure",
        error_type="timeout",
        trace_id="trace-2",
        model_version="deepseek-chat-2026-01",
        redaction_applied=True,
        memory_used=False,
        case_citation_used=True,
        citation_validation_passed=False,
        reason_code="transport_or_stream_error",
    )

    sanitized = sanitize_model_metadata(metadata)

    assert sanitized["redaction_applied"] is True
    assert sanitized["memory_used"] is False
    assert sanitized["case_citation_used"] is True
    assert sanitized["citation_validation_passed"] is False
    assert sanitized["reason_code"] == "transport_or_stream_error"
    assert "13800138000" not in str(sanitized)


def test_model_log_metadata_rejects_free_text_safe_note():
    with pytest.raises(TypeError):
        ModelCallMetadata(
            provider="deepseek",
            model="deepseek-chat",
            prompt_version="v1",
            knowledge_base_version="kb-1",
            token_count=20,
            duration_ms=15,
            status="failure",
            error_type="timeout",
            trace_id="trace-2",
            safe_note="用户手机号13800138000已处理",
        )


def test_record_model_call_stores_only_sanitized_metadata():
    metadata = ModelCallMetadata(
        provider="deepseek",
        model="deepseek-chat",
        prompt_version="v1",
        knowledge_base_version="kb-1",
        token_count=1,
        duration_ms=1,
        status="success",
        error_type=None,
        trace_id="trace-3",
        api_key="sk-secret",
    )

    recorded = record_model_call(metadata)

    assert recorded["provider"] == "deepseek"
    assert "api_key" not in recorded
    assert "sk-secret" not in str(recorded)
