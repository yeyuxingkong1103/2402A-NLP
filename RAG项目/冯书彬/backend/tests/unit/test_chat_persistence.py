import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from backend.app.models.conversation import Conversation
from backend.app.models.message import Message
from backend.app.rag.pipeline import RetrievalDecision
from backend.app.repositories.chat_repository import SQLAlchemyChatRepository, create_chat_tables
from backend.app.services.chat_service import ChatService


class FakeLlm:
    def __init__(self, answer="回答"):
        self.answer = answer

    async def stream_chat(self, _request):
        yield self.answer


def _allowed_decision():
    return RetrievalDecision(can_answer=True, reason="ok", top_documents=[], citations=[], context="婚姻家庭依据")


def _repository(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum")
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum")
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True)
    create_chat_tables(engine)
    return SQLAlchemyChatRepository(engine)


@pytest.mark.asyncio
async def test_chat_service_persists_finished_message_with_repository(monkeypatch):
    repository = _repository(monkeypatch)
    service = ChatService(rag_decider=lambda _text: _allowed_decision(), llm_client=FakeLlm("持久化回答"), repository=repository)

    result = await service.send_message("user-1", "c1", "北京已婚有孩子，想咨询离婚时孩子抚养权怎么办，无危险")
    stored = repository.get_message("user-1", result.message_id)
    conversation = repository.get_conversation("user-1", "c1")

    assert stored is not None
    assert stored.answer.startswith("持久化回答")
    assert conversation is not None
    assert result.message_id in conversation.message_ids


@pytest.mark.asyncio
async def test_chat_service_loads_message_from_repository_for_regeneration(monkeypatch):
    repository = _repository(monkeypatch)
    repository.save_conversation(Conversation(id="c1", user_id="user-1"))
    repository.save_message(Message(id="m1", conversation_id="c1", user_id="user-1", user_text="北京已婚有孩子，想咨询离婚时孩子抚养权怎么办，无危险", answer="旧回答", status="answered"))
    service = ChatService(rag_decider=lambda _text: _allowed_decision(), llm_client=FakeLlm("新回答"), repository=repository)

    result = await service.regenerate_answer("user-1", "m1")
    stored = repository.get_message("user-1", "m1")

    assert result.answer.startswith("新回答")
    assert stored is not None
    assert stored.regeneration_count == 1
    assert stored.answer.startswith("新回答")
