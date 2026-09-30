import pytest

from backend.app.api.v1.chat import chat_service
from backend.app.models.conversation import Conversation
from backend.app.models.message import Message
from backend.app.services import memory_service
from backend.app.services.conversation_service import delete_conversation_and_exclusive_memories
from backend.app.services.memory_service import create_memory_from_conversation, list_user_memories


@pytest.fixture(autouse=True)
def memory_flow_env(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum-value")
    memory_service.reset_memory_store_for_tests()
    chat_service.reset_for_tests()


def test_delete_conversation_cleans_exclusive_memory_and_messages():
    chat_service.conversations["c1"] = Conversation(id="c1", user_id="user-flow")
    chat_service.messages["m1"] = Message(id="m1", conversation_id="c1", user_id="user-flow", user_text="问题")
    chat_service.conversations["c1"].message_ids.append("m1")
    exclusive = create_memory_from_conversation("user-flow", "c1", {"case_type": "divorce"}, "离婚摘要")
    shared = create_memory_from_conversation("user-flow", "c1", {"case_type": "property"}, "财产摘要")
    shared.source_conversation_ids.append("c2")

    deleted = delete_conversation_and_exclusive_memories("user-flow", "c1")

    assert exclusive.id in deleted
    assert shared.id not in deleted
    assert "m1" not in chat_service.messages
    assert [memory.id for memory in list_user_memories("user-flow")] == [shared.id]


def test_delete_conversation_rejects_cross_user_without_deleting():
    chat_service.conversations["c1"] = Conversation(id="c1", user_id="owner")
    memory = create_memory_from_conversation("owner", "c1", {"case_type": "divorce"}, "离婚摘要")

    with pytest.raises(PermissionError):
        delete_conversation_and_exclusive_memories("attacker", "c1")

    assert "c1" in chat_service.conversations
    assert [item.id for item in list_user_memories("owner")] == [memory.id]
