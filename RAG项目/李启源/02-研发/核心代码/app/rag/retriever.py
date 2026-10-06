"""Vector retrieval, reranking, and hybrid search for RAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """Raised when retrieval fails."""


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """A retrieved chunk with its similarity score."""

    chunk_id: str
    text: str
    source: str
    score: float
    summary: str = ""
    parent_id: str | None = None
    parent_summary: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    """Results from a retrieval operation."""

    chunks: tuple[RetrievedChunk, ...]
    query: str
    total_found: int
    reranked: bool = False


class VectorRetriever(Protocol):
    """Interface for vector similarity search."""

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors and return raw results."""


class MilvusRetriever:
    """Milvus vector retriever with filtering support."""

    def __init__(
        self,
        collection_name: str,
        *,
        uri: str = "http://localhost:19530",
        token: str | None = None,
        metric_type: str = "COSINE",
    ) -> None:
        try:
            from pymilvus import Collection, connections  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RetrievalError("pymilvus is required for MilvusRetriever") from exc

        connections.connect(alias="default", uri=uri, token=token or "")
        self.collection = Collection(collection_name)
        self.collection.load()
        self.metric_type = metric_type

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        filter_expr = self._build_filter_expression(filters) if filters else None
        search_params = {"metric_type": self.metric_type, "params": {"ef": 64}}

        try:
            results = self.collection.search(
                data=[list(query_vector)],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["text", "source", "doc_id", "chunk_id", "summary", "tenant_id", "kb_id"],
            )
        except Exception as exc:
            raise RetrievalError(f"Milvus search failed: {exc}") from exc

        chunks: list[dict[str, Any]] = []
        for hit in results[0]:
            chunks.append({
                "chunk_id": hit.entity.get("chunk_id") or str(hit.id),
                "text": hit.entity.get("text", ""),
                "source": hit.entity.get("source", ""),
                "score": float(hit.score),
                "summary": hit.entity.get("summary", ""),
                "doc_id": hit.entity.get("doc_id"),
                "tenant_id": hit.entity.get("tenant_id"),
                "kb_id": hit.entity.get("kb_id"),
            })
        return chunks

    def _build_filter_expression(self, filters: dict[str, Any]) -> str:
        allowed = {"tenant_id", "kb_id", "doc_id", "role_id"}
        expressions: list[str] = []
        for key, value in filters.items():
            if key not in allowed:
                raise RetrievalError(f"unsupported filter field: {key}")
            if isinstance(value, bool):
                expressions.append(f"{key} == {str(value).lower()}")
            elif isinstance(value, int):
                expressions.append(f"{key} == {value}")
            elif key == "doc_id" and isinstance(value, str):
                safe_value = value.replace("\\", "\\\\").replace('"', '\\"')
                expressions.append(f'{key} == "{safe_value}"')
            else:
                raise RetrievalError(f"unsupported filter value for: {key}")
        return " && ".join(expressions) if expressions else ""


class InMemoryRetriever:
    """In-memory retriever for testing."""

    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self.records = records

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        candidates = list(self.records.values())
        if filters:
            candidates = [
                rec for rec in candidates
                if all(rec.get(key) == value for key, value in filters.items())
            ]

        scored: list[tuple[float, dict[str, Any]]] = []
        for record in candidates:
            vector = record.get("vector", [])
            if not vector:
                continue
            similarity = self._cosine_similarity(query_vector, vector)
            scored.append((similarity, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "chunk_id": rec.get("chunk_id") or rec.get("id", ""),
                "text": rec.get("text", ""),
                "source": rec.get("source", ""),
                "score": score,
                "summary": rec.get("summary", ""),
                "doc_id": rec.get("doc_id"),
            }
            for score, rec in scored[:top_k]
        ]

    def _cosine_similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)


class RAGRetriever:
    """High-level RAG retriever with embedding and optional reranking."""

    def __init__(
        self,
        embedding_client: Any,
        vector_retriever: VectorRetriever,
        *,
        reranker: Any | None = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_retriever = vector_retriever
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        rerank: bool = False,
        rerank_top_k: int = 3,
        similarity_threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query."""
        query_vectors = self.embedding_client.embed_texts([query])
        if not query_vectors or not query_vectors[0]:
            raise RetrievalError("Failed to embed query")

        query_vector = query_vectors[0]
        raw_results = self.vector_retriever.search(
            query_vector,
            top_k=top_k if not rerank else top_k * 2,
            filters=filters,
        )

        filtered_results = [
            result for result in raw_results
            if result["score"] >= similarity_threshold
        ]

        chunks = [
            RetrievedChunk(
                chunk_id=result["chunk_id"],
                text=result["text"],
                source=result["source"],
                score=result["score"],
                summary=result.get("summary", ""),
                metadata={"doc_id": result.get("doc_id")},
            )
            for result in filtered_results
        ]

        if rerank and self.reranker and chunks:
            chunks = self._rerank_chunks(query, chunks, rerank_top_k)
            logger.info("chunks reranked", extra={"count": len(chunks)})

        return RetrievalResult(
            chunks=tuple(chunks[:top_k]),
            query=query,
            total_found=len(raw_results),
            reranked=rerank and self.reranker is not None,
        )

    def _rerank_chunks(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank chunks using a reranker model."""
        try:
            scores = self.reranker.rank(query, [chunk.text for chunk in chunks])
            reranked = sorted(
                zip(chunks, scores, strict=False),
                key=lambda x: x[1],
                reverse=True,
            )
            return [
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    score=float(score),
                    summary=chunk.summary,
                    parent_id=chunk.parent_id,
                    parent_summary=chunk.parent_summary,
                    metadata=chunk.metadata,
                )
                for chunk, score in reranked[:top_k]
            ]
        except Exception as exc:
            logger.warning("reranking failed, returning original order", extra={"error": str(exc)})
            return chunks[:top_k]
