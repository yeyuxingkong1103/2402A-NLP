"""Immutable result models shared by hybrid retrieval components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class SearchResult:
    """One normalized dense, sparse, or fused retrieval hit."""

    chunk_id: str
    text: str
    source: str
    score: float
    summary: str = ""
    doc_id: str | None = None
    metadata: dict[str, Any] | None = None
    search_type: str = "dense"


@dataclass(slots=True, frozen=True)
class HybridSearchResult:
    """Final ranking plus the two source rankings used to build it."""

    results: tuple[SearchResult, ...]
    dense_results: tuple[SearchResult, ...]
    sparse_results: tuple[SearchResult, ...]
    fusion_method: str
    total_found: int
