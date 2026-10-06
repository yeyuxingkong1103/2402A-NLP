# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-基于PDF文档的问答系统
src/schemas.py — Pydantic 请求/响应模型
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    use_rag: bool = Field(True)
    doc_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    qa_log_id: int
    rating: int = Field(..., ge=1, le=5)
    is_correct: Optional[int] = None
    comment: Optional[str] = Field(None, max_length=500)

class ReferenceItem(BaseModel):
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    score: Optional[float] = None
    preview: Optional[str] = None

class AskResponse(BaseModel):
    mode: str
    answer: str
    qa_log_id: Optional[int] = None
    references: List[ReferenceItem] = []
    latency_ms: float = 0.0
    token_usage: Dict[str, int] = {}
    query_understanding: Optional[Dict[str, Any]] = None
    breakdown: Optional[Dict[str, float]] = None
    rag_answer: Optional[str] = None
    llm_answer: Optional[str] = None
    rag_latency_ms: Optional[float] = None
    llm_latency_ms: Optional[float] = None

class FeedbackResponse(BaseModel):
    id: int
    ok: bool = True

class HealthResponse(BaseModel):
    status: str
    mysql: str = "unknown"
    milvus: str = "unknown"
    embedding_model: str = ""
    llm_model: str = ""
    uptime_seconds: float = 0.0

class StatsResponse(BaseModel):
    total_documents: int = 0
    total_chunks: int = 0
    total_qa_logs: int = 0
    total_feedback: int = 0
    collection_entities: Optional[int] = None
