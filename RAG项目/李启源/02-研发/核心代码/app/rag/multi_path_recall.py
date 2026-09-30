"""Multi-path recall facade retaining the historical public API."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from app.rag.multi_path_models import MultiPathRecallResult, MultiSourceResult
from app.rag.multi_path_sources import (
    InternetSearchRetriever,
    MySQLFullTextRetriever,
    Neo4JGraphRetriever,
    RedisCacheRetriever,
)

logger = logging.getLogger(__name__)


class MultiPathRecaller:
    """Combine local vector, SQL, cache, graph, and optional internet sources."""

    def __init__(
        self,
        *,
        milvus_retriever: Any,
        mysql_retriever: MySQLFullTextRetriever | None = None,
        redis_retriever: RedisCacheRetriever | None = None,
        neo4j_retriever: Neo4JGraphRetriever | None = None,
        internet_retriever: InternetSearchRetriever | None = None,
    ) -> None:
        self.milvus_retriever = milvus_retriever
        self.mysql_retriever = mysql_retriever
        self.redis_retriever = redis_retriever
        self.neo4j_retriever = neo4j_retriever
        self.internet_retriever = internet_retriever

    def recall(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        enable_cache: bool = True,
        enable_mysql: bool = True,
        enable_graph: bool = False,
        enable_internet: bool = False,
    ) -> MultiPathRecallResult:
        """Recall from enabled sources, preferring a cache hit."""
        if enable_cache and self.redis_retriever:
            cached = self.redis_retriever.search(query, top_k=top_k, filters=filters)
            if cached:
                return MultiPathRecallResult(tuple(cached[:top_k]), {"redis": len(cached)}, len(cached), 0)

        all_results: list[MultiSourceResult] = []
        source_counts: dict[str, int] = {}
        self._extend(all_results, source_counts, "milvus", self._milvus_search(query, top_k, filters))
        if enable_mysql and self.mysql_retriever:
            self._extend(all_results, source_counts, "mysql", self.mysql_retriever.search(query, top_k=max(1, top_k // 2), filters=filters))
        if enable_graph and self.neo4j_retriever:
            self._extend(all_results, source_counts, "neo4j", self.neo4j_retriever.search(query, top_k=max(1, top_k // 4)))
        if enable_internet and self.internet_retriever and len(all_results) < 5:
            self._extend(all_results, source_counts, "internet", self.internet_retriever.search(query, top_k=5))

        deduped = self._deduplicate(all_results)
        final_results = sorted(deduped, key=lambda result: result.score, reverse=True)[:top_k]
        if self.redis_retriever and final_results:
            self.redis_retriever.cache_results(
                query,
                [
                    {
                        "chunk_id": result.chunk_id,
                        "text": result.text,
                        "source": result.source,
                        "summary": result.summary,
                        "score": result.score,
                    }
                    for result in final_results
                ],
                filters=filters,
            )
        return MultiPathRecallResult(
            results=tuple(final_results),
            source_counts=source_counts,
            total_candidates=len(all_results),
            dedup_removed=len(all_results) - len(deduped),
        )

    @staticmethod
    def _extend(
        target: list[MultiSourceResult], counts: dict[str, int], name: str, results: list[MultiSourceResult]
    ) -> None:
        target.extend(results)
        counts[name] = len(results)

    def _milvus_search(
        self, query: str, top_k: int, filters: dict[str, Any] | None
    ) -> list[MultiSourceResult]:
        """Normalize the primary retriever's records without assuming one shape."""
        try:
            raw = self.milvus_retriever.search(query, top_k=top_k, filters=filters)
        except TypeError:
            raw = self.milvus_retriever.search(query, top_k=top_k)
        if hasattr(raw, "results"):
            raw = raw.results
        return [
            item
            if isinstance(item, MultiSourceResult)
            else MultiSourceResult(
                chunk_id=item["chunk_id"],
                text=item["text"],
                source=item.get("source", "milvus"),
                score=float(item.get("score", 0.0)),
                retrieval_source="milvus",
                summary=item.get("summary", ""),
                metadata=item.get("metadata"),
            )
            for item in raw
        ]

    @staticmethod
    def _deduplicate(results: list[MultiSourceResult]) -> list[MultiSourceResult]:
        """Keep the highest-scoring result for normalized duplicate text."""
        seen: dict[str, MultiSourceResult] = {}
        for result in results:
            key = hashlib.md5(result.text.strip().lower().encode()).hexdigest()
            if key not in seen or result.score > seen[key].score:
                seen[key] = result
        return list(seen.values())


def build_multi_path_recaller(
    milvus_retriever: Any,
    mysql_engine: Any | None = None,
    redis_client: Any | None = None,
    neo4j_driver: Any | None = None,
) -> MultiPathRecaller:
    """Build a recaller from the optional configured sources."""
    return MultiPathRecaller(
        milvus_retriever=milvus_retriever,
        mysql_retriever=MySQLFullTextRetriever(mysql_engine) if mysql_engine else None,
        redis_retriever=RedisCacheRetriever(redis_client) if redis_client else None,
        neo4j_retriever=Neo4JGraphRetriever(neo4j_driver) if neo4j_driver else None,
    )


__all__ = [
    "InternetSearchRetriever",
    "MultiPathRecallResult",
    "MultiPathRecaller",
    "MultiSourceResult",
    "MySQLFullTextRetriever",
    "Neo4JGraphRetriever",
    "RedisCacheRetriever",
    "build_multi_path_recaller",
]
