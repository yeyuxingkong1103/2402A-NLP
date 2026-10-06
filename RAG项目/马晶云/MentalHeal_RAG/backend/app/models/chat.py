from datetime import datetime

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), unique=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    display_name: Mapped[str] = mapped_column(String(80))
    account_role: Mapped[str] = mapped_column(String(20), default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class AIRole(Base):
    __tablename__ = "ai_roles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    role_id: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(80))
    description: Mapped[str] = mapped_column(String(255))
    system_instruction: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True)
    user_id: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(120))
    last_message: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    role_id: Mapped[str] = mapped_column(String(64), default="mental-health")
    status: Mapped[str] = mapped_column(String(20), default="active")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(String(64), unique=True)
    session_id: Mapped[str] = mapped_column(String(64))
    user_id: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    citations_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class LongTermMemory(Base):
    __tablename__ = "long_term_memories"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    memory_id: Mapped[str] = mapped_column(String(64), unique=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    memory_type: Mapped[str] = mapped_column(String(32), default="preference")
    content: Mapped[str] = mapped_column(Text)
    is_confirmed: Mapped[bool] = mapped_column(Boolean, default=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    knowledge_base_id: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(String(500), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(256), unique=True)
    source_file: Mapped[str] = mapped_column(String(512))
    raw_path: Mapped[str] = mapped_column(String(1024))
    processed_path: Mapped[str] = mapped_column(String(1024))
    markdown_path: Mapped[str] = mapped_column(String(1024))
    page_count: Mapped[int] = mapped_column(default=0)
    total_text_chars: Mapped[int] = mapped_column(default=0)
    weak_page_count: Mapped[int] = mapped_column(default=0)
    non_rag_page_count: Mapped[int] = mapped_column(default=0)
    table_page_count: Mapped[int] = mapped_column(default=0)
    content_type_counts: Mapped[dict] = mapped_column(JSON, default=dict)
    cleaning_flag_counts: Mapped[dict] = mapped_column(JSON, default=dict)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(256), unique=True)
    document_id: Mapped[str] = mapped_column(String(256))
    source_file: Mapped[str] = mapped_column(String(512))
    page_start: Mapped[int] = mapped_column(default=0)
    page_end: Mapped[int] = mapped_column(default=0)
    chunk_index: Mapped[int] = mapped_column(default=0)
    chunk_type: Mapped[str] = mapped_column(String(32), default="text")
    char_count: Mapped[int] = mapped_column(default=0)
    token_estimate: Mapped[int] = mapped_column(default=0)
    has_table: Mapped[bool] = mapped_column(Boolean, default=False)
    quality_score: Mapped[float] = mapped_column(default=1.0)
    milvus_collection: Mapped[str] = mapped_column(String(128), default="knowledge_chunks")
    knowledge_version: Mapped[str] = mapped_column(String(128), default="")
    text_preview: Mapped[str] = mapped_column(String(512), default="")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class RagEvaluationRun(Base):
    __tablename__ = "rag_evaluation_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    dataset_path: Mapped[str] = mapped_column(String(512))
    sample_count: Mapped[int] = mapped_column(default=0)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict] = mapped_column(JSON, default=dict)
    error_message: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="queued")
    raw_dir: Mapped[str] = mapped_column(String(1024))
    processed_dir: Mapped[str] = mapped_column(String(1024))
    chunks_dir: Mapped[str] = mapped_column(String(1024))
    embeddings_dir: Mapped[str] = mapped_column(String(1024))
    document_count: Mapped[int] = mapped_column(default=0)
    page_count: Mapped[int] = mapped_column(default=0)
    chunk_count: Mapped[int] = mapped_column(default=0)
    note: Mapped[str] = mapped_column(Text, default="")
    summary_json: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime)
