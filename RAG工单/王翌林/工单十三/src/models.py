# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-基于PDF文档的问答系统
src/models.py — SQLAlchemy ORM 模型
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from src.db import Base

class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (Index("ix_doc_hash", "file_hash"), {"mysql_charset": "utf8mb4"})
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(256), nullable=False)
    file_path = Column(String(1024), nullable=False)
    file_hash = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    total_pages = Column(Integer, default=0)
    total_chars = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    error_msg = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (Index("ix_chunk_doc_id", "doc_id"), Index("ix_chunk_chunk_id", "chunk_id"), {"mysql_charset": "utf8mb4"})
    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(String(128), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    global_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    char_count = Column(Integer, default=0)
    page = Column(Integer, nullable=False)
    vector_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    document = relationship("Document", back_populates="chunks")

class QALog(Base):
    __tablename__ = "qa_logs"
    __table_args__ = (Index("ix_qa_log_created", "created_at"), {"mysql_charset": "utf8mb4"})
    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    query_rewrite = Column(Text, nullable=True)
    rag_answer = Column(Text, nullable=True)
    llm_answer = Column(Text, nullable=True)
    retrieved_chunks = Column(JSON, nullable=True)
    latency_ms = Column(Float, default=0.0)
    llm_latency_ms = Column(Float, default=0.0)
    token_usage = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    feedbacks = relationship("Feedback", back_populates="qa_log", cascade="all, delete-orphan")

class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (Index("ix_feedback_qa_log", "qa_log_id"), {"mysql_charset": "utf8mb4"})
    id = Column(Integer, primary_key=True, autoincrement=True)
    qa_log_id = Column(Integer, ForeignKey("qa_logs.id", ondelete="CASCADE"), nullable=False)
    rating = Column(Integer, nullable=False)
    is_correct = Column(Integer, nullable=True)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    qa_log = relationship("QALog", back_populates="feedbacks")

class EvalResult(Base):
    __tablename__ = "eval_results"
    __table_args__ = (Index("ix_eval_question_id", "question_id"), {"mysql_charset": "utf8mb4"})
    id = Column(Integer, primary_key=True, autoincrement=True)
    question_id = Column(String(64), nullable=False)
    question = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=True)
    rag_answer = Column(Text, nullable=False)
    llm_answer = Column(Text, nullable=True)
    metrics_json = Column(JSON, nullable=True)
    evaluator = Column(String(32), default="ragas")
    created_at = Column(DateTime, default=datetime.utcnow)
