# -*- coding: utf-8 -*-
"""ORM 模型：6 张表。

简化取舍：枚举字段用 String 承载并在业务层校验，避免 MySQL ENUM 变更时的迁移成本。
"""
from datetime import datetime

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer,
    JSON, String, Text, UniqueConstraint,
)

from .core.db import Base


class User(Base):
    """用户。多用户隔离的根。"""
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    username = Column(String(64), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(32), nullable=False)
    display_name = Column(String(64))
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)


class Character(Base):
    """角色：人格三层（身份/风格/约束）+ 知识库绑定。"""
    __tablename__ = "characters"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    slug = Column(String(64), nullable=False, unique=True)
    name = Column(String(64), nullable=False)
    category = Column(String(32))
    avatar = Column(String(255))
    description = Column(String(512))

    identity_block = Column(Text)            # 身份层
    style_json = Column(JSON)                # 风格层 {tone, length, ...}
    domain_constraints = Column(Text)        # 约束层（免责声明等）
    prompt_template = Column(Text)           # 完整模板骨架

    kb_collection = Column(String(64), nullable=False)
    recall_top_k = Column(Integer, default=20)
    rerank_top_k = Column(Integer, default=5)
    temperature = Column(Float, default=0.3)
    is_builtin = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)


class Conversation(Base):
    """会话：user_id 外键是多用户隔离的核心。"""
    __tablename__ = "conversations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False)
    character_id = Column(BigInteger, ForeignKey("characters.id"), nullable=False)
    title = Column(String(128))
    message_count = Column(Integer, default=0, nullable=False)
    last_message_at = Column(DateTime)
    is_deleted = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    __table_args__ = (
        Index("idx_conv_user_char", "user_id", "character_id", "is_deleted", "last_message_at"),
    )


class Message(Base):
    """消息：长期记忆 + 链路追溯（sources/trace 用 JSON 列）。"""
    __tablename__ = "messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id = Column(BigInteger, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(16), nullable=False)          # user / assistant / system
    content = Column(Text, nullable=False)
    rewritten_query = Column(Text)
    sources_json = Column(JSON)
    trace_json = Column(JSON)
    latency_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    __table_args__ = (Index("idx_msg_conv", "conversation_id", "id"),)


class SearchHistory(Base):
    """检索调试台的检索历史（按用户归属）。"""
    __tablename__ = "search_history"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False)
    query = Column(String(512), nullable=False)
    collection = Column(String(64))
    top_k = Column(Integer)
    use_rerank = Column(Boolean, default=True)
    result_json = Column(JSON)
    elapsed_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    __table_args__ = (Index("idx_search_user_time", "user_id", "created_at"),)


class KbDocument(Base):
    """知识库文档：(collection, file_hash) 唯一 → 入库幂等。"""
    __tablename__ = "kb_documents"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    collection = Column(String(64), nullable=False)
    source_path = Column(String(512), nullable=False)
    file_hash = Column(String(40), nullable=False)
    doc_type = Column(String(8), nullable=False)         # pdf / txt / md
    chunk_strategy = Column(String(16), default="auto")
    chunk_count = Column(Integer, default=0)
    status = Column(String(16), default="pending", nullable=False)
    error_msg = Column(Text)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        UniqueConstraint("collection", "file_hash", name="uk_coll_hash"),
        Index("idx_kb_coll_status", "collection", "status"),
    )
