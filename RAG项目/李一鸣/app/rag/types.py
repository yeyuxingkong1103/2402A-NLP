from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    """知识库中的最小检索单元，包含原文、元数据和可选向量。"""
    id: str
    document_id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass(slots=True)
class RetrievedChunk:
    """一次检索结果，同时保留融合、向量、词法和重排分数。"""
    chunk: ChunkRecord
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float | None = None
    retrieval_method: str = "hybrid"


@dataclass(slots=True)
class ParsedDocument:
    """文档解析后的统一结构，供分块器继续处理。"""
    text: str
    pages: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatResult:
    """聊天服务返回给 API 层的答案、引用、追踪 ID 和耗时。"""
    answer: str
    citations: list[RetrievedChunk]
    trace_id: str
    timings: dict[str, float]
