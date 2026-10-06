# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""API 请求/响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
class HealthOut(BaseModel):
    ok: bool
    ollama: str = ""
    milvus: str = ""
    collection_rows: int = 0
    llm_model: str = ""
    embed_model: str = ""


# ----------------------------------------------------------------------
class CitationOut(BaseModel):
    page_label: str
    page_no: int
    chunk_type: str
    section_path: str = ""
    snippet: str = ""
    score: float = 0.0


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    history: list[dict[str, str]] | None = None
    # 工单01「纯 LLM vs RAG 对比」：true 时不走检索，用于对照
    no_rag: bool = False
    stream: bool = True


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut] = []
    mode: Literal["rag", "no_rag"] = "rag"
    ttft_ms: int = 0          # 首 token 延迟 —— 工单01 ≤3s 的验收口径
    total_ms: int = 0         # 完整答案耗时（主动披露）
    retrieval_ms: int = 0
    n_candidates: int = 0
    n_filtered: int = 0


# ----------------------------------------------------------------------
class IngestRequest(BaseModel):
    pdf_path: str | None = None
    rebuild: bool = False
    # 只解析前 N 页，供快速验证；None = 全量
    limit_pages: int | None = None


class IngestStatus(BaseModel):
    running: bool = False
    stage: str = "idle"
    done: int = 0
    total: int = 0
    message: str = ""
    error: str = ""
    stats: dict[str, Any] = {}


# ----------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: Literal["up", "down"]
    comment: str = ""
    citations: list[CitationOut] = []


class FeedbackOut(BaseModel):
    ok: bool
    received: int


# ----------------------------------------------------------------------
class EvalItemOut(BaseModel):
    id: int
    question: str
    category: str = ""
    rag_answer: str = ""
    no_rag_answer: str = ""
    rule_hit: bool | None = None
    rule_detail: str = ""
    context_relevance: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None
    citations: list[str] = []
    ttft_ms: int = 0


class EvalSummary(BaseModel):
    n: int = 0
    rule_hit_rate: float = 0.0
    rag_rule_hits: int = 0
    no_rag_rule_hits: int = 0
    avg_context_relevance: float = 0.0
    avg_faithfulness: float = 0.0
    avg_answer_relevance: float = 0.0
    avg_context_recall: float = 0.0
    avg_answer_correctness: float = 0.0
    ttft_p50_ms: int = 0
    ttft_p95_ms: int = 0
    total_p50_ms: int = 0
    total_p95_ms: int = 0
    generated_at: str = ""


class EvalReport(BaseModel):
    summary: EvalSummary
    items: list[EvalItemOut] = []


# ----------------------------------------------------------------------
class KBCollectionOut(BaseModel):
    name: str
    rows: int
    description: str = ""


class KBListOut(BaseModel):
    collections: list[KBCollectionOut] = []


class DeleteDocRequest(BaseModel):
    doc_id: str
