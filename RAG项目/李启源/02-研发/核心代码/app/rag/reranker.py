"""Reranking facade preserving the original imports."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from app.rag.reranker_providers import (
    BGERerankerHTTPClient,
    LexicalReranker,
    LocalCrossEncoderReranker,
    MockReranker,
    RerankResult,
    Reranker,
    RerankerError,
    RerankerOutput,
    build_reranker_from_env,
    tokenize_mixed,
)

logger = logging.getLogger(__name__)


class RerankingRetriever:
    """Apply a provider, threshold low scores, and preserve source metadata."""

    def __init__(self, reranker: Reranker, *, rerank_top_k: int = 10, score_threshold: float = 0.3) -> None:
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k
        self.score_threshold = score_threshold

    def rerank_results(self, query: str, results: Sequence[Any], *, top_k: int | None = None) -> RerankerOutput:
        if not results:
            return RerankerOutput(tuple(), 0, 0, 0)
        try:
            scores = self.reranker.rerank(query, [result.text for result in results], top_k=top_k)
        except Exception as exc:
            logger.error("Reranking failed: %s", exc)
            fallback = tuple(
                RerankResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    source=result.source,
                    rerank_score=result.score,
                    original_score=result.score,
                    summary=getattr(result, "summary", ""),
                    metadata=getattr(result, "metadata", None),
                )
                for result in results[: top_k or len(results)]
            )
            return RerankerOutput(fallback, len(results), 0, 0)
        reranked = [
            RerankResult(
                chunk_id=result.chunk_id,
                text=result.text,
                source=result.source,
                rerank_score=score,
                original_score=result.score,
                summary=getattr(result, "summary", ""),
                metadata={**(getattr(result, "metadata", None) or {}), "original_rank": index, "rerank_score": score},
            )
            for index, (result, score) in enumerate(zip(results, scores, strict=False))
            if score >= self.score_threshold
        ]
        reranked.sort(key=lambda result: result.rerank_score, reverse=True)
        final = reranked[: top_k or self.rerank_top_k]
        return RerankerOutput(tuple(final), len(results), len(reranked), len(results) - len(reranked))


__all__ = [
    "BGERerankerHTTPClient", "LexicalReranker", "LocalCrossEncoderReranker",
    "MockReranker", "RerankResult", "Reranker", "RerankerError", "RerankerOutput",
    "RerankingRetriever", "build_reranker_from_env", "tokenize_mixed",
]
