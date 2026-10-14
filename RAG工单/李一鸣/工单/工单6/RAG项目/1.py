"""Work order 06: dense, sparse and metadata-filtered hybrid retrieval."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Chunk, dense_search, lexical_search, rrf


@dataclass
class RetrievalConfig:
    dense_k: int = 20
    lexical_k: int = 20
    final_k: int = 8
    rrf_k: int = 60


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], config: RetrievalConfig | None = None):
        self.chunks = chunks
        self.config = config or RetrievalConfig()

    def _eligible(self, filters: dict[str, Any] | None) -> list[Chunk]:
        if not filters:
            return self.chunks
        return [chunk for chunk in self.chunks if all(chunk.metadata.get(k) == v for k, v in filters.items())]

    def search(self, query: str, filters: dict[str, Any] | None = None) -> list[tuple[Chunk, float]]:
        eligible = self._eligible(filters)
        dense = dense_search(query, eligible, self.config.dense_k)
        lexical = lexical_search(query, eligible, self.config.lexical_k)
        return rrf(dense, lexical, k=self.config.rrf_k, top_k=self.config.final_k)

    def explain(self, query: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        eligible = self._eligible(filters)
        dense = dense_search(query, eligible, self.config.dense_k)
        lexical = lexical_search(query, eligible, self.config.lexical_k)
        fused = rrf(dense, lexical, k=self.config.rrf_k, top_k=self.config.final_k)
        return {
            "query": query,
            "filters": filters or {},
            "dense": [{"id": c.id, "score": s} for c, s in dense],
            "lexical": [{"id": c.id, "score": s} for c, s in lexical],
            "fused": [{"id": c.id, "score": s} for c, s in fused],
        }
