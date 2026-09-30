"""Reranker contracts and provider implementations."""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

logger = logging.getLogger(__name__)
_CJK_RUN = re.compile(r"[一-鿿㐀-䶿]+")
_ASCII_WORD = re.compile(r"[a-z0-9]+")


def tokenize_mixed(text: str) -> list[str]:
    """Tokenize CJK as character bigrams and ASCII as words."""
    terms = _ASCII_WORD.findall(text.lower())
    for run in _CJK_RUN.findall(text.lower()):
        terms.extend(run if len(run) == 1 else (run[index : index + 2] for index in range(len(run) - 1)))
    return terms


class RerankerError(RuntimeError):
    """Raised when a configured reranker cannot score documents."""


@dataclass(slots=True, frozen=True)
class RerankResult:
    """One result with original and reranker scores."""

    chunk_id: str
    text: str
    source: str
    rerank_score: float
    original_score: float
    summary: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class RerankerOutput:
    """Reranking diagnostics and final results."""

    results: tuple[RerankResult, ...]
    original_count: int
    reranked_count: int
    filtered_count: int


class Reranker(Protocol):
    """Common provider interface."""

    def rerank(self, query: str, documents: Sequence[str], *, top_k: int | None = None) -> list[float]: ...


class BGERerankerHTTPClient:
    """HTTP adapter for an authorized BGE reranker service."""

    def __init__(self, base_url: str, api_key: str | None = None, model: str = "BAAI/bge-reranker-v2-m3", timeout_seconds: int = 30) -> None:
        self.base_url, self.api_key, self.model, self.timeout_seconds = base_url, api_key, model, timeout_seconds

    def rerank(self, query: str, documents: Sequence[str], *, top_k: int | None = None) -> list[float]:
        if not documents:
            return []
        try:
            import httpx
        except ImportError as exc:
            raise RerankerError("httpx is required for BGERerankerHTTPClient") from exc
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {"model": self.model, "query": query, "documents": list(documents)}
        if top_k is not None:
            payload["top_k"] = top_k
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(self.base_url.rstrip("/") + "/rerank", json=payload, headers=headers)
                response.raise_for_status()
                body = response.json()
        except Exception as exc:
            raise RerankerError(f"Reranker request failed: {exc}") from exc
        results = body.get("results")
        if not isinstance(results, list):
            raise RerankerError("Invalid reranker response format")
        scores = [0.0] * len(documents)
        for item in results:
            if isinstance(item, dict) and 0 <= item.get("index", -1) < len(scores):
                scores[item["index"]] = float(item.get("relevance_score", 0.0))
        return scores


class LexicalReranker:
    """Chinese-aware IDF lexical scorer that needs no model download."""

    def __init__(self, *, phrase_bonus: float = 0.15) -> None:
        self.phrase_bonus = phrase_bonus

    def rerank(self, query: str, documents: Sequence[str], *, top_k: int | None = None) -> list[float]:
        if not documents:
            return []
        query_terms = set(tokenize_mixed(query))
        if not query_terms:
            return [0.0] * len(documents)
        doc_terms = [set(tokenize_mixed(doc)) for doc in documents]
        total = len(documents)
        idf = {term: math.log(1.0 + total / (1.0 + sum(term in terms for terms in doc_terms))) for term in query_terms}
        denominator = sum(idf.values())
        if denominator <= 0:
            return [0.5] * len(documents)
        query_lower = query.lower()
        scores = []
        for doc, terms in zip(documents, doc_terms, strict=True):
            coverage = sum(idf[term] for term in query_terms if term in terms) / denominator
            score = math.sqrt(coverage) + (self.phrase_bonus if query_lower in doc.lower() else 0.0)
            scores.append(min(1.0, max(0.0, score)))
        return scores


class LocalCrossEncoderReranker:
    """Lazy-loading local sentence-transformers cross encoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", *, device: str | None = None, batch_size: int = 16, max_length: int = 512) -> None:
        self.model_name, self.device, self.batch_size, self.max_length = model_name, device, batch_size, max_length
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, max_length=self.max_length, device=self.device)
            except ImportError as exc:
                raise RerankerError("sentence-transformers is required for LocalCrossEncoderReranker") from exc
            except Exception as exc:
                raise RerankerError(f"Failed to load cross-encoder {self.model_name}: {exc}") from exc
        return self._model

    def rerank(self, query: str, documents: Sequence[str], *, top_k: int | None = None) -> list[float]:
        if not documents:
            return []
        try:
            scores = [float(value) for value in self._load().predict([(query, doc) for doc in documents], batch_size=self.batch_size, show_progress_bar=False)]
        except Exception as exc:
            raise RerankerError(f"Cross-encoder scoring failed: {exc}") from exc
        return [1.0 / (1.0 + math.exp(-score)) for score in scores] if any(score < 0 or score > 1 for score in scores) else scores


class MockReranker:
    """Explicit inert provider for offline tests."""

    def __init__(self, *, score: float = 0.5) -> None:
        self.score = score

    def rerank(self, query: str, documents: Sequence[str], *, top_k: int | None = None) -> list[float]:
        return [self.score] * len(documents)


def build_reranker_from_env() -> Reranker:
    """Select a provider from environment variables."""
    configured = os.getenv("RERANKER_PROVIDER")
    provider = configured.lower() if configured else "lexical"
    if provider in {"lexical", "bm25"}:
        return LexicalReranker()
    if provider in {"local", "cross-encoder", "local-bge"}:
        return LocalCrossEncoderReranker(
            model_name=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            device=os.getenv("RERANKER_DEVICE") or None,
            batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "16")),
            max_length=int(os.getenv("RERANKER_MAX_LENGTH", "512")),
        )
    if provider == "mock":
        return MockReranker()
    if provider in {"bge", "bge-reranker", "http"}:
        base_url = os.getenv("RERANKER_BASE_URL")
        if not base_url:
            raise RerankerError("RERANKER_BASE_URL is required for HTTP reranker")
        return BGERerankerHTTPClient(
            base_url=base_url,
            api_key=os.getenv("RERANKER_API_KEY"),
            model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            timeout_seconds=int(os.getenv("RERANKER_TIMEOUT_SECONDS", "30")),
        )
    raise RerankerError(f"Unsupported reranker provider: {provider}")
