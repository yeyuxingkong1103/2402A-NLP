from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, MetaData, String, Table, Text, delete, insert, select, update
from sqlalchemy.engine import Engine

from backend.app.core.crypto import EncryptedValue, decrypt_text, encrypt_text
from backend.app.models.conversation import Conversation
from backend.app.models.message import Message


chat_metadata = MetaData()

conversations_table = Table(
    "conversations",
    chat_metadata,
    Column("id", String(64), primary_key=True),
    Column("user_id", String(36), nullable=False, index=True),
    Column("follow_up_count", Integer, nullable=False),
    Column("active", Boolean, nullable=False),
    Column("deleted_at", DateTime(timezone=True), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

messages_table = Table(
    "messages",
    chat_metadata,
    Column("id", String(36), primary_key=True),
    Column("conversation_id", String(64), ForeignKey("conversations.id"), nullable=False, index=True),
    Column("user_id", String(36), nullable=False, index=True),
    Column("encrypted_user_text", Text, nullable=False),
    Column("encrypted_answer", Text, nullable=False),
    Column("status", String(32), nullable=False, index=True),
    Column("citations", JSON, nullable=False),
    Column("regeneration_count", Integer, nullable=False),
    Column("corrected", Boolean, nullable=False),
    Column("source_message_id", String(64), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


def create_chat_tables(engine: Engine) -> None:
    # 测试和本地验证可显式创建聊天表；生产环境使用 Alembic 迁移。
    chat_metadata.create_all(engine, tables=[conversations_table, messages_table], checkfirst=True)


def _as_utc(value: datetime | None) -> datetime | None:
    # SQLite 读取时可能丢失时区，统一恢复 UTC 以避免时间比较异常。
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _serialize_encrypted(value: EncryptedValue) -> str:
    # 将 nonce、版本和密文作为一个 JSON 信封保存，避免明文落库。
    return json.dumps(value.__dict__, separators=(",", ":"))


def _deserialize_encrypted(value: str) -> EncryptedValue:
    # 只接受由本仓库生成的完整加密信封。
    data = json.loads(value)
    return EncryptedValue(
        ciphertext=str(data["ciphertext"]),
        nonce=str(data["nonce"]),
        encrypted_data_key=str(data["encrypted_data_key"]),
        key_version=str(data["key_version"]),
    )


def _row_to_conversation(row) -> Conversation | None:
    if row is None:
        return None
    data = row._mapping
    return Conversation(
        id=data["id"],
        user_id=data["user_id"],
        follow_up_count=int(data["follow_up_count"]),
        active=bool(data["active"]),
    )


def _row_to_message(row) -> Message | None:
    if row is None:
        return None
    data = row._mapping
    return Message(
        id=data["id"],
        conversation_id=data["conversation_id"],
        user_id=data["user_id"],
        user_text=decrypt_text(_deserialize_encrypted(data["encrypted_user_text"]), purpose="chat-user-text"),
        answer=decrypt_text(_deserialize_encrypted(data["encrypted_answer"]), purpose="chat-answer"),
        status=data["status"],
        citations=data["citations"],
        regeneration_count=int(data["regeneration_count"]),
        corrected=bool(data["corrected"]),
        source_message_id=data["source_message_id"],
    )


def _conversation_values(conversation: Conversation) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "id": conversation.id,
        "user_id": conversation.user_id,
        "follow_up_count": conversation.follow_up_count,
        "active": conversation.active,
        "deleted_at": None,
        "created_at": now,
        "updated_at": now,
    }


def _message_values(message: Message) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "user_id": message.user_id,
        "encrypted_user_text": _serialize_encrypted(encrypt_text(message.user_text, purpose="chat-user-text")),
        "encrypted_answer": _serialize_encrypted(encrypt_text(message.answer, purpose="chat-answer")),
        "status": message.status,
        "citations": message.citations,
        "regeneration_count": message.regeneration_count,
        "corrected": message.corrected,
        "source_message_id": message.source_message_id,
        "created_at": now,
        "updated_at": now,
    }


class SQLAlchemyChatRepository:
    # 会话和消息共享事务引擎；聊天正文通过 AES-GCM 加密后落库。
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def get_conversation(self, user_id: str, conversation_id: str) -> Conversation | None:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(conversations_table)
                .where(conversations_table.c.id == conversation_id)
                .where(conversations_table.c.user_id == user_id)
                .where(conversations_table.c.deleted_at.is_(None))
            ).first()
            message_ids = connection.execute(
                select(messages_table.c.id)
                .where(messages_table.c.conversation_id == conversation_id)
                .where(messages_table.c.user_id == user_id)
                .order_by(messages_table.c.created_at)
            ).scalars().all()
        conversation = _row_to_conversation(row)
        if conversation is not None:
            conversation.message_ids = list(message_ids)
        return conversation

    def save_conversation(self, conversation: Conversation) -> None:
        values = _conversation_values(conversation)
        with self.engine.begin() as connection:
            exists = connection.execute(select(conversations_table.c.id).where(conversations_table.c.id == conversation.id)).first()
            if exists:
                values.pop("created_at")
                connection.execute(update(conversations_table).where(conversations_table.c.id == conversation.id).values(**values))
                return
            connection.execute(insert(conversations_table).values(**values))

    def get_message(self, user_id: str, message_id: str) -> Message | None:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(messages_table)
                .where(messages_table.c.id == message_id)
                .where(messages_table.c.user_id == user_id)
            ).first()
        return _row_to_message(row)

    def save_message(self, message: Message) -> None:
        values = _message_values(message)
        with self.engine.begin() as connection:
            exists = connection.execute(select(messages_table.c.id).where(messages_table.c.id == message.id)).first()
            if exists:
                values.pop("created_at")
                connection.execute(update(messages_table).where(messages_table.c.id == message.id).values(**values))
                return
            connection.execute(insert(messages_table).values(**values))

    def delete_message(self, user_id: str, message_id: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(messages_table).where(messages_table.c.id == message_id).where(messages_table.c.user_id == user_id))

    def delete_conversation(self, user_id: str, conversation_id: str) -> None:
        now = datetime.now(timezone.utc)
        with self.engine.begin() as connection:
            connection.execute(
                delete(messages_table)
                .where(messages_table.c.conversation_id == conversation_id)
                .where(messages_table.c.user_id == user_id)
            )
            connection.execute(
                update(conversations_table)
                .where(conversations_table.c.id == conversation_id)
                .where(conversations_table.c.user_id == user_id)
                .values(deleted_at=now, active=False, updated_at=now)
            )

    def delete_user(self, user_id: str) -> None:
        with self.engine.begin() as connection:
            conversation_ids = select(conversations_table.c.id).where(conversations_table.c.user_id == user_id)
            connection.execute(delete(messages_table).where(messages_table.c.conversation_id.in_(conversation_ids)))
            connection.execute(delete(conversations_table).where(conversations_table.c.user_id == user_id))
