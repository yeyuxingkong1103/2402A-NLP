"""Contracts for multi-source retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class MultiSourceResult:
    """One normalized result returned by a retrieval source."""

    chunk_id: str
    text: str
    source: str
    score: float
    retrieval_source: str
    summary: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class MultiPathRecallResult:
    """Final results plus source and deduplication diagnostics."""

    results: tuple[MultiSourceResult, ...]
    source_counts: dict[str, int]
    total_candidates: int
    dedup_removed: int
