from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from backend.app.core.crypto import decrypt_text
from backend.app.models.conversation import Conversation
from backend.app.models.message import Message
from backend.app.repositories.chat_repository import SQLAlchemyChatRepository, _deserialize_encrypted, create_chat_tables, messages_table


def _repository(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum")
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum")
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True)
    create_chat_tables(engine)
    return SQLAlchemyChatRepository(engine), engine


def test_chat_repository_persists_and_round_trips_encrypted_message(monkeypatch):
    repository, engine = _repository(monkeypatch)
    conversation = Conversation(id="c1", user_id="user-1")
    message = Message(id="m1", conversation_id="c1", user_id="user-1", user_text="手机号13800138000", answer="回答")

    repository.save_conversation(conversation)
    repository.save_message(message)
    restored = SQLAlchemyChatRepository(engine).get_message("user-1", "m1")

    assert restored == message
    with engine.begin() as connection:
        stored = connection.execute(select(messages_table.c.encrypted_user_text)).scalar_one()
    assert "13800138000" not in stored
    assert decrypt_text(_deserialize_encrypted(stored), purpose="chat-user-text") == "手机号13800138000"


def test_chat_repository_enforces_owner_on_reads_and_deletes(monkeypatch):
    repository, _ = _repository(monkeypatch)
    repository.save_conversation(Conversation(id="c1", user_id="owner"))
    repository.save_message(Message(id="m1", conversation_id="c1", user_id="owner", user_text="问题"))

    assert repository.get_conversation("other", "c1") is None
    assert repository.get_message("other", "m1") is None
    repository.delete_conversation("other", "c1")
    assert repository.get_conversation("owner", "c1") is not None
    assert repository.get_message("owner", "m1") is not None




def test_chat_repository_accepts_frontend_prefixed_conversation_id(monkeypatch):
    repository, _ = _repository(monkeypatch)
    conversation_id = "web-fa965820-38c6-4960-a505-ff6501fd071a"

    repository.save_conversation(Conversation(id=conversation_id, user_id="real-rag-e2e-user"))
    repository.save_message(Message(id="m-real-rag", conversation_id=conversation_id, user_id="real-rag-e2e-user", user_text="抚养权如何判断？"))

    restored = repository.get_conversation("real-rag-e2e-user", conversation_id)

    assert restored is not None
    assert restored.id == conversation_id
