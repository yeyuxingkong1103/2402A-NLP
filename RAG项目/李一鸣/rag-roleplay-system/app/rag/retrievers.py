import logging
import re
from collections import Counter

from app.core.config import Settings
from app.rag.embeddings import EmbeddingService
from app.rag.types import ChunkRecord, RetrievedChunk

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


class HybridRetriever:
    """Dense + BM25-style lexical retrieval with reciprocal rank fusion."""

    def __init__(self, vector_store, embedding_service: EmbeddingService, settings: Settings):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.settings = settings
        self._index: list[ChunkRecord] = []
        self._bm25 = None

    def refresh(self) -> None:
        self._index = self.vector_store.all_records()
        self._bm25 = None
        if self._index:
            try:
                from rank_bm25 import BM25Okapi  # type: ignore

                self._bm25 = BM25Okapi([tokenize(record.text) for record in self._index])
                logger.info("lexical index refreshed with rank-bm25: %s records", len(self._index))
            except ImportError:
                logger.info("rank-bm25 unavailable; using overlap lexical fallback: %s records", len(self._index))
        else:
            logger.info("lexical index refreshed: 0 records")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.top_k_final
        if not self._index:
            self.refresh()
        dense_hits = self.vector_store.search(
            self.embedding_service.embed_query(query),
            top_k=self.settings.top_k_vector,
            score_threshold=self.settings.vector_score_threshold,
            filters=filters,
        )
        lexical_hits = self._lexical_search(query, self.settings.top_k_bm25, filters)
        fused = self._rrf(dense_hits, lexical_hits)
        logger.info(
            "hybrid retrieval complete: dense=%s lexical=%s fused=%s",
            len(dense_hits),
            len(lexical_hits),
            len(fused),
        )
        return fused[:top_k]

    def _lexical_search(
        self, query: str, top_k: int, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_counts = Counter(query_tokens)
        scored: list[RetrievedChunk] = []
        eligible = [
            record
            for record in self._index
            if not filters
            or all(record.metadata.get(key) == value for key, value in filters.items())
        ]
        bm25_scores: dict[str, float] = {}
        if self._bm25 is not None and not filters and len(eligible) == len(self._index):
            bm25_scores = {
                record.id: float(score)
                for record, score in zip(self._index, self._bm25.get_scores(query_tokens))
            }

        for record in eligible:
            tokens = tokenize(record.text)
            if not tokens:
                continue
            counts = Counter(tokens)
            overlap = sum(min(query_counts[token], counts[token]) for token in query_counts)
            if overlap == 0:
                continue
            overlap_score = overlap / max(1, len(query_tokens))
            score = bm25_scores.get(record.id, 0.0) + overlap_score
            scored.append(
                RetrievedChunk(
                    chunk=record,
                    score=score,
                    lexical_score=score,
                    retrieval_method="bm25",
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _rrf(
        dense_hits: list[RetrievedChunk], lexical_hits: list[RetrievedChunk], k: int = 60
    ) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            item = merged.setdefault(hit.chunk.id, RetrievedChunk(chunk=hit.chunk, score=0.0))
            item.dense_score = hit.dense_score
            item.score += 1.0 / (k + rank)
            item.retrieval_method = "vector"
        for rank, hit in enumerate(lexical_hits, start=1):
            item = merged.setdefault(hit.chunk.id, RetrievedChunk(chunk=hit.chunk, score=0.0))
            item.lexical_score = hit.lexical_score
            item.score += 1.0 / (k + rank)
            item.retrieval_method = "hybrid" if item.retrieval_method == "vector" else "bm25"
        output = list(merged.values())
        output.sort(key=lambda item: item.score, reverse=True)
        return output
