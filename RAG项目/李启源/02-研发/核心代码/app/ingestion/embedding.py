"""Embedding clients for BGE-m3 and deterministic local tests."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot return valid vectors."""


class EmbeddingClient(Protocol):
    """Common interface for embedding providers."""

    @property
    def dimension(self) -> int:
        """Return vector dimension."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts and return dense vectors."""


@dataclass(slots=True)
class MockEmbeddingClient:
    """Deterministic embedding client for tests and no-API-key development."""

    dim: int = 32

    @property
    def dimension(self) -> int:
        return self.dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = []
            for index in range(self.dim):
                byte = digest[index % len(digest)]
                values.append((byte / 127.5) - 1.0)
            norm = sum(value * value for value in values) ** 0.5 or 1.0
            vectors.append([value / norm for value in values])
        return vectors


@dataclass(slots=True)
class BgeM3HttpEmbeddingClient:
    """HTTP adapter for a BGE-m3 embedding service.

    The service is expected to expose an OpenAI-compatible embedding endpoint:
    ``POST /v1/embeddings`` with ``model`` and ``input`` fields. If your BGE-m3
    service uses a different contract, keep this class boundary and replace only
    ``embed_texts``.
    """

    base_url: str
    api_key: str | None = None
    model: str = "BAAI/bge-m3"
    dim: int = 1024
    timeout_seconds: int = 30
    batch_size: int = 32

    @property
    def dimension(self) -> int:
        return self.dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            import httpx
        except ImportError as exc:
            raise EmbeddingError("httpx is required for BgeM3HttpEmbeddingClient") from exc

        if self.batch_size < 1:
            raise EmbeddingError("embedding batch size must be positive")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = self.base_url.rstrip("/") + "/v1/embeddings"
        vectors: list[list[float]] = []

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                for start in range(0, len(texts), self.batch_size):
                    batch = texts[start : start + self.batch_size]
                    response = client.post(
                        url,
                        json={"model": self.model, "input": batch},
                        headers=headers,
                    )
                    response.raise_for_status()
                    body = response.json()
                    data = body.get("data")
                    if not isinstance(data, list):
                        raise EmbeddingError("embedding response missing data list")
                    if len(data) != len(batch):
                        raise EmbeddingError(
                            f"embedding count mismatch: expected {len(batch)}, got {len(data)}"
                        )
                    for item in data:
                        vector = item.get("embedding") if isinstance(item, dict) else None
                        if not isinstance(vector, list):
                            raise EmbeddingError(
                                "embedding response item missing embedding vector"
                            )
                        if len(vector) != self.dim:
                            raise EmbeddingError(
                                f"embedding dimension mismatch: expected {self.dim}, got {len(vector)}"
                            )
                        vectors.append([float(value) for value in vector])
        except EmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider errors need wrapping
            raise EmbeddingError(f"embedding request failed: {exc}") from exc

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"embedding count mismatch: expected {len(texts)}, got {len(vectors)}"
            )
        logger.info("embedded texts", extra={"count": len(texts), "model": self.model})
        return vectors


@dataclass(slots=True)
class OpenAIEmbeddingClient:
    """OpenAI embedding client."""

    api_key: str
    model: str = "text-embedding-3-small"
    dim: int = 1536
    timeout_seconds: int = 30

    @property
    def dimension(self) -> int:
        return self.dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingError("openai package is required for OpenAIEmbeddingClient") from exc

        try:
            client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
            response = client.embeddings.create(input=texts, model=self.model)
            vectors = [item.embedding for item in response.data]
            logger.info("embedded texts", extra={"count": len(texts), "model": self.model})
            return vectors
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc


def build_embedding_client_from_env() -> EmbeddingClient:
    """Create an embedding client from environment variables."""
    provider = os.getenv("EMBEDDING_PROVIDER", "mock").lower()
    if provider == "mock":
        return MockEmbeddingClient(dim=int(os.getenv("EMBEDDING_DIM", "32")))
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("EMBEDDING_API_KEY")
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY is required for OpenAI embeddings")
        return OpenAIEmbeddingClient(
            api_key=api_key,
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            dim=int(os.getenv("EMBEDDING_DIM", "1536")),
            timeout_seconds=int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30")),
        )
    if provider in {"bge-m3", "http", "http_service", "online_api"}:
        base_url = os.getenv("EMBEDDING_BASE_URL")
        if not base_url:
            raise EmbeddingError("EMBEDDING_BASE_URL is required for HTTP embeddings")
        return BgeM3HttpEmbeddingClient(
            base_url=base_url,
            api_key=os.getenv("EMBEDDING_API_KEY") or None,
            model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            timeout_seconds=int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        )
    raise EmbeddingError(f"unsupported embedding provider: {provider}")
