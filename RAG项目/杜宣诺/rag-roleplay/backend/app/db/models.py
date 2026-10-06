from datetime import datetime
from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base

# SQLite 只对 INTEGER PRIMARY KEY（ROWID 别名）自动自增；BIGINT PRIMARY KEY 不自动赋值。
# 用 with_variant 让 MySQL 保持 BIGINT、SQLite 退化为 INTEGER，从而主键在两种库都正确自增。
BigIntPK = BigInteger().with_variant(Integer, "sqlite")


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Character(Base):
    __tablename__ = "characters"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    owner_user_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    avatar: Mapped[str] = mapped_column(String(512), default="")
    persona: Mapped[str] = mapped_column(Text, default="")
    worldview: Mapped[str] = mapped_column(Text, default="")
    relationship: Mapped[str] = mapped_column(Text, default="")
    hidden_setting: Mapped[str] = mapped_column(Text, default="")
    greeting: Mapped[str] = mapped_column(Text, default="")
    sample_dialogue: Mapped[str] = mapped_column(Text, default="")
    is_preset: Mapped[bool] = mapped_column(default=False)
    tags: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    character_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("characters.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("sessions.id"), nullable=False)
    role: Mapped[str] = mapped_column(Enum("user", "assistant"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    sources: Mapped[str] = mapped_column(Text, default="")  # JSON 字符串：该回答引用的来源列表
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Favorite(Base):
    __tablename__ = "favorites"
    __table_args__ = (UniqueConstraint("user_id", "message_id", name="uq_favorite_user_message"),)
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    message_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("messages.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class MemoryTask(Base):
    __tablename__ = "memory_tasks"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("sessions.id"), nullable=False)
    last_processed_msg_id: Mapped[int] = mapped_column(BigInteger, default=0)
    status: Mapped[str] = mapped_column(Enum("pending", "running", "done", "failed"), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    ext: Mapped[str] = mapped_column(String(16), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(Enum("pending", "running", "done", "failed"), default="pending")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
