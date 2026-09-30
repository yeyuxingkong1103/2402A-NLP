"""Application configuration management."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""

    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: list[str] | None = None
    max_upload_size_mb: int = 50
    api_key: str | None = None
    require_api_key: bool = False
    rate_limit_per_minute: int = 60


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding service configuration."""

    provider: Literal["mock", "bge-m3", "http"] = "mock"
    base_url: str | None = None
    api_key: str | None = None
    model: str = "BAAI/bge-m3"
    dimension: int = 1024
    timeout_seconds: int = 30


@dataclass(frozen=True)
class VectorStoreConfig:
    """Vector store configuration."""

    provider: Literal["milvus", "inmemory"] = "milvus"
    uri: str = "http://localhost:19530"
    collection_name: str = "kf_chunks"
    token: str | None = None
    enable_sparse: bool = False


@dataclass(frozen=True)
class DatabaseConfig:
    """Relational database configuration."""

    url: str | None = None
    pool_size: int = 5
    max_overflow: int = 10


@dataclass(frozen=True)
class LLMConfig:
    """LLM service configuration."""

    provider: Literal["openai", "anthropic", "mock"] = "mock"
    api_key: str | None = None
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout_seconds: int = 60


@dataclass(frozen=True)
class RAGConfig:
    """RAG pipeline configuration."""

    top_k: int = 5
    rerank: bool = False
    rerank_top_k: int = 3
    similarity_threshold: float = 0.5
    max_context_tokens: int = 4000
    enable_parent_retrieval: bool = False


def load_config_from_env() -> tuple[
    AppConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    DatabaseConfig,
    LLMConfig,
    RAGConfig,
]:
    """Load all configuration from environment variables."""
    app_config = AppConfig(
        environment=os.getenv("APP_ENV", "development"),  # type: ignore[arg-type]
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("API_PORT", "8000")),
        api_prefix=os.getenv("API_PREFIX", "/api/v1"),
        cors_origins=os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else None,
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")),
        api_key=os.getenv("API_KEY") or None,
        require_api_key=os.getenv("REQUIRE_API_KEY", "false").lower() == "true",
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
    )

    embedding_config = EmbeddingConfig(
        provider=os.getenv("EMBEDDING_PROVIDER", "mock"),  # type: ignore[arg-type]
        base_url=os.getenv("EMBEDDING_BASE_URL"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        dimension=int(os.getenv("EMBEDDING_DIM", "32" if os.getenv("EMBEDDING_PROVIDER", "mock").lower() == "mock" else "1024")),
        timeout_seconds=int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30")),
    )

    vector_config = VectorStoreConfig(
        provider=os.getenv("VECTOR_STORE_PROVIDER", "milvus"),  # type: ignore[arg-type]
        uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        collection_name=os.getenv("MILVUS_COLLECTION", "kf_chunks"),
        token=os.getenv("MILVUS_TOKEN"),
        enable_sparse=os.getenv("MILVUS_ENABLE_SPARSE", "false").lower() == "true",
    )

    database_config = DatabaseConfig(
        url=os.getenv("DATABASE_URL"),
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
    )

    llm_config = LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "mock"),  # type: ignore[arg-type]
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
        timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
    )

    rag_config = RAGConfig(
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        rerank=os.getenv("RAG_RERANK", "false").lower() == "true",
        rerank_top_k=int(os.getenv("RAG_RERANK_TOP_K", "3")),
        similarity_threshold=float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.5")),
        max_context_tokens=int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "4000")),
        enable_parent_retrieval=os.getenv("RAG_ENABLE_PARENT_RETRIEVAL", "false").lower() == "true",
    )

    return app_config, embedding_config, vector_config, database_config, llm_config, rag_config
