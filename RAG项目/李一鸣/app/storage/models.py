from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Role(Base):
    """角色配置表：保存人物设定、专业领域、安全策略和知识范围。"""
    __tablename__ = "roles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid4().hex)
    name: Mapped[str] = mapped_column(String(120), index=True)
    category: Mapped[str] = mapped_column(String(64), default="friend")
    description: Mapped[str] = mapped_column(Text, default="")
    personality: Mapped[str] = mapped_column(Text, default="")
    expertise: Mapped[str] = mapped_column(Text, default="")
    speaking_style: Mapped[str] = mapped_column(Text, default="")
    safety_policy: Mapped[str] = mapped_column(Text, default="")
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    knowledge_scope: Mapped[str] = mapped_column(String(120), default="global")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Document(Base):
    """知识库文档元数据表；实际文本块和向量存放在向量索引中。"""
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid4().hex)
    filename: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(500), default="upload")
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="processing")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class User(Base):
    """用户基础信息表。"""
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(120), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class ConversationMessage(Base):
    """永久聊天记录表；citations_json 保存本轮引用的 JSON 快照。"""
    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid4().hex)
    user_id: Mapped[str] = mapped_column(String(120), index=True)
    role_id: Mapped[str] = mapped_column(String(64), index=True)
    conversation_id: Mapped[str] = mapped_column(String(120), index=True)
    message_type: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    citations_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
