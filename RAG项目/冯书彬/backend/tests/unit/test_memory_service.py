import json

import pytest

from backend.app.core.crypto import decrypt_text
from backend.app.services import memory_service
from backend.app.services.memory_service import (
    confirm_memory_update,
    create_memory_from_conversation,
    delete_exclusive_memories_for_conversation,
    list_user_memories,
    mark_candidate_update,
    set_memory_auto_extraction,
)


@pytest.fixture(autouse=True)
def memory_env(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum-value")
    memory_service.reset_memory_store_for_tests()


def test_delete_conversation_removes_only_exclusive_memory():
    exclusive = create_memory_from_conversation("user-1", "c1", {"case_type": "divorce"}, "已脱敏摘要")
    shared = create_memory_from_conversation("user-1", "c1", {"case_type": "property"}, "共同摘要")
    shared.source_conversation_ids.append("c2")

    deleted = delete_exclusive_memories_for_conversation("user-1", "c1")

    assert exclusive.id in deleted
    assert shared.id not in deleted
    assert [memory.id for memory in list_user_memories("user-1")] == [shared.id]


def test_memory_creation_encrypts_facts_and_redacted_summary():
    memory = create_memory_from_conversation("user-1", "c1", {"children": "one"}, "手机号13800138000已脱敏")

    assert memory.encrypted_facts.ciphertext != '{"children": "one"}'
    assert "13800138000" not in decrypt_text(memory.encrypted_summary, purpose="memory-summary")
    assert decrypt_text(memory.encrypted_summary, purpose="memory-summary") == "手机号[手机号]已脱敏"
    assert memory.prompt_required is True


def test_disabling_auto_extraction_keeps_existing_memory():
    memory = create_memory_from_conversation("user-1", "c1", {"case_type": "labor"}, "劳动争议摘要")

    set_memory_auto_extraction("user-1", False)
    skipped = create_memory_from_conversation("user-1", "c2", {"case_type": "rent"}, "租赁摘要")

    assert skipped is None
    assert [item.id for item in list_user_memories("user-1")] == [memory.id]


def test_structured_facts_are_redacted_before_storage():
    memory = create_memory_from_conversation("user-1", "c1", {"phone": "13800138000", "nested": {"bank": "6222021234567890123"}}, "摘要")

    facts = json.loads(decrypt_text(memory.encrypted_facts, purpose="memory-facts"))

    assert facts["phone"] == "[手机号]"
    assert facts["nested"]["bank"] == "[银行卡号]"


def test_memory_candidate_missing_ids_raise_controlled_errors():
    with pytest.raises(ValueError, match="记忆不存在"):
        mark_candidate_update("missing-memory", {"case_type": "divorce"})

    with pytest.raises(ValueError, match="候选记忆不存在"):
        confirm_memory_update("missing-candidate", "user-1")
