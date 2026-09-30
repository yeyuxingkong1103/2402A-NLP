"""Data contracts and deterministic helpers for document chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

ChunkStrategy = Literal["fixed", "sentence", "paragraph", "heading", "semantic"]
Summarizer = Callable[[str, int], str]
EmbeddingFunction = Callable[[list[str]], list[list[float]]]
SENTENCE_RE = re.compile(r"(?<=[。！？!?；;\.])\s+|(?<=[。！？!?；;\.])")


@dataclass(slots=True, frozen=True)
class ChunkingConfig:
    """Configuration shared by all chunking strategies."""

    strategy: ChunkStrategy = "paragraph"
    chunk_size: int = 800
    chunk_overlap: int = 120
    min_chunk_chars: int = 20
    parent_chunk_size: int = 2000
    summary_max_chars: int = 220
    semantic_similarity_threshold: float = 0.72
    create_parent_chunks: bool = True
    deduplicate: bool = True


@dataclass(slots=True, frozen=True)
class ParentBlock:
    """Larger context block associated with one or more child chunks."""

    parent_id: str
    text: str
    summary: str
    child_chunk_ids: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Chunk:
    """Child knowledge chunk ready for embedding and indexing."""

    chunk_id: str
    text: str
    summary: str
    source: str
    page_start: int | None
    page_end: int | None
    chunk_index: int
    text_hash: str
    parent_id: str | None = None
    parent_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ChunkingResult:
    """Child chunks plus their optional parent context blocks."""

    chunks: tuple[Chunk, ...]
    parents: tuple[ParentBlock, ...]


def stable_hash(text: str) -> str:
    """Return a stable SHA-256 hash for normalized text."""
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def default_summarize(text: str, max_chars: int) -> str:
    """Create a deterministic extractive summary without invoking an LLM."""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    summary = ""
    for sentence in (part.strip() for part in SENTENCE_RE.split(compact)):
        if not sentence:
            continue
        if len(summary) + len(sentence) > max_chars:
            break
        summary += sentence
    return (summary or compact[:max_chars]).strip()
