# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
src/schemas_v6.py —— 工单六 API v6 Pydantic 模型（新增文件）

在 v5 多轮对话模型基础上增加检索策略配置字段。
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.retrieval.retrieval_config import (
    FUSION_RRF, MODE_HYBRID, RERANKER_LLM, MATCH_AND,
)


class RetrievalOptions(BaseModel):
    """工单六：检索策略选项（全部可选，不传走默认）"""
    mode: Optional[str] = None               # vector / fulltext / hybrid
    fusion: Optional[str] = None             # rrf / weighted
    reranker: Optional[str] = None           # llm / tfidf / adaptive
    vector_weight: Optional[float] = None
    fulltext_weight: Optional[float] = None
    match: Optional[str] = None              # and / or / phrase / fuzzy
    fields: Optional[List[str]] = None
    top_k: Optional[int] = None
    use_table: Optional[bool] = None
    use_image: Optional[bool] = None


class AskV6Request(BaseModel):
    """工单六：问答请求（含检索策略覆盖）"""
    question: str
    doc_id: Optional[str] = None
    company: Optional[str] = None
    use_rag: bool = True
    lang_override: Optional[str] = None
    retrieval: Optional[RetrievalOptions] = None


class ChatV6Request(BaseModel):
    """工单六：多轮对话请求（继承工单五会话能力 + 检索策略）"""
    question: str
    session_id: Optional[str] = None
    doc_id: Optional[str] = None
    use_image: bool = True
    retrieval: Optional[RetrievalOptions] = None


class RetrievalMeta(BaseModel):
    """工单六：实际生效的检索策略回显"""
    mode: str = MODE_HYBRID
    fusion: str = FUSION_RRF
    reranker: str = RERANKER_LLM
    match: str = MATCH_AND
    vector_weight: float = 0.6
    fulltext_weight: float = 0.4
    vector_hits: int = 0
    fulltext_hits: int = 0
    elapsed_ms: float = 0.0


class AskV6Response(BaseModel):
    mode: str = "rag_v6"
    query: str
    route: Dict[str, Any] = {}
    answer: str
    references: List[Dict[str, Any]] = []
    retrieved_text_chunks: List[Dict[str, Any]] = []
    retrieved_tables: List[Dict[str, Any]] = []
    retrieved_images: List[Dict[str, Any]] = []
    latency_ms: float = 0.0
    breakdown: Dict[str, float] = {}
    retrieval: Dict[str, Any] = {}
    token_usage: Dict[str, Any] = {}
    # 工单六：多轮对话附加字段（/chat 端点）
    session_id: Optional[str] = None
    resolved_query: Optional[str] = None
    entity: Optional[str] = None
    is_followup: bool = False
    coref_strategy: str = "direct"


class ConfigV6Response(BaseModel):
    """工单六：可选策略与预设"""
    modes: List[str]
    fusions: List[str]
    rerankers: List[str]
    matches: List[str]
    fields: List[str]
    presets: Dict[str, Any]
    defaults: Dict[str, Any]


class HealthV6Response(BaseModel):
    status: str = "ok"
    engine: str = "rag_v6"
    milvus_images_rows: int = 0
    fulltext_docs: int = 0
    work_order: str = Field(default="人工智能NLP-RAG-混合检索任务")
