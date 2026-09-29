from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    id: str
    document_id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass(slots=True)
class RetrievedChunk:
    chunk: ChunkRecord
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float | None = None
    retrieval_method: str = "hybrid"


@dataclass(slots=True)
class ParsedDocument:
    text: str
    pages: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatResult:
    answer: str
    citations: list[RetrievedChunk]
    trace_id: str
    timings: dict[str, float]
