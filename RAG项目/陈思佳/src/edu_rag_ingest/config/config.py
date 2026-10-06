from __future__ import annotations

"""应用配置定义与加载逻辑，统一管理爬虫、解析、检索、模型、缓存和数据库配置。"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CrawlerConfig:
    seed_urls: list[str]
    allowed_domains: list[str]
    max_pages: int
    request_timeout_seconds: int
    delay_seconds: float
    user_agent: str
    output_dir: Path
    follow_links: bool
    download_file_types: list[str]
    allowed_file_urls: list[str]


@dataclass(frozen=True)
class MinerUConfig:
    enabled: bool
    api_url: str
    api_key: str
    timeout_seconds: int
    upload_field: str
    extra_form: dict[str, Any]
    response_text_fields: list[str]
    parsed_output_dir: Path
    artifact_output_dir: Path


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    min_chunk_chars: int
    output_path: Path


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model_name: str
    device: str
    batch_size: int
    normalize_embeddings: bool
    max_seq_length: int


@dataclass(frozen=True)
class MilvusConfig:
    uri: str
    token: str
    collection_name: str
    vector_dim: int
    drop_existing: bool
    metric_type: str
    index_type: str
    index_params: dict[str, Any]
    search_params: dict[str, Any]


@dataclass(frozen=True)
class QAConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    temperature: float
    top_k: int
    max_context_chars: int
    timeout_seconds: int


@dataclass(frozen=True)
class HybridSearchConfig:
    enabled: bool
    keyword_top_k: int
    candidate_multiplier: int
    rrf_k: int
    vector_weight: float
    keyword_weight: float


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool
    model_name: str
    device: str
    batch_size: int
    max_candidates: int


@dataclass(frozen=True)
class MemoryConfig:
    redis_url: str
    namespace: str
    recent_message_limit: int
    session_ttl_seconds: int
    long_term_limit: int


@dataclass(frozen=True)
class DatabaseConfig:
    url: str
    echo: bool


@dataclass(frozen=True)
class AppConfig:
    """应用所有子系统配置的聚合对象。"""
    crawler: CrawlerConfig
    mineru: MinerUConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    milvus: MilvusConfig
    qa: QAConfig
    hybrid_search: HybridSearchConfig
    rerank: RerankConfig
    memory: MemoryConfig
    database: DatabaseConfig
    metadata_defaults: dict[str, str]


def _load_dotenv(path: Path) -> None:
    """读取简单的 KEY=VALUE 文件，不覆盖已有环境变量。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(config_path: str | Path) -> AppConfig:
    """读取 YAML 和环境变量，构造类型化的应用配置。"""
    path = Path(config_path)
    project_root = path.parent.parent if path.parent.name == "configs" else Path.cwd()
    _load_dotenv(project_root / ".env")
    _load_dotenv(project_root / "frontend" / ".env")
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    crawler = raw["crawler"]
    mineru = raw["mineru"]
    chunking = raw["chunking"]
    embedding = raw.get("embedding", {})
    milvus = raw.get("milvus", {})
    qa = raw.get("qa", {})
    hybrid_search = raw.get("hybrid_search", {})
    rerank = raw.get("rerank", {})
    memory = raw.get("memory", {})
    database = raw.get("database", {})

    return AppConfig(
        crawler=CrawlerConfig(
            seed_urls=list(crawler.get("seed_urls", [])),
            allowed_domains=list(crawler.get("allowed_domains", [])),
            max_pages=int(crawler.get("max_pages", 3)),
            request_timeout_seconds=int(crawler.get("request_timeout_seconds", 20)),
            delay_seconds=float(crawler.get("delay_seconds", 1)),
            user_agent=str(crawler.get("user_agent", "EducationRAGBot/0.1")),
            output_dir=Path(crawler.get("output_dir", "data/raw")),
            follow_links=bool(crawler.get("follow_links", False)),
            download_file_types=list(crawler.get("download_file_types", [".pdf"])),
            allowed_file_urls=list(crawler.get("allowed_file_urls", [])),
        ),
        mineru=MinerUConfig(
            enabled=bool(mineru.get("enabled", True)),
            api_url=str(mineru.get("api_url", "")),
            api_key=os.getenv("MINERU_API_KEY", str(mineru.get("api_key", ""))),
            timeout_seconds=int(mineru.get("timeout_seconds", 300)),
            upload_field=str(mineru.get("upload_field", "file")),
            extra_form=dict(mineru.get("extra_form", {})),
            response_text_fields=list(mineru.get("response_text_fields", [])),
            parsed_output_dir=Path(mineru.get("parsed_output_dir", "data/parsed")),
            artifact_output_dir=Path(mineru.get("artifact_output_dir", "data/mineru")),
        ),
        chunking=ChunkingConfig(
            chunk_size=int(chunking.get("chunk_size", 900)),
            chunk_overlap=int(chunking.get("chunk_overlap", 120)),
            min_chunk_chars=int(chunking.get("min_chunk_chars", 80)),
            output_path=Path(chunking.get("output_path", "data/chunks/document_chunks.jsonl")),
        ),
        embedding=EmbeddingConfig(
            provider=str(embedding.get("provider", "sentence_transformers")),
            model_name=str(embedding.get("model_name", "BAAI/bge-m3")),
            device=str(embedding.get("device", "cpu")),
            batch_size=int(embedding.get("batch_size", 8)),
            normalize_embeddings=bool(embedding.get("normalize_embeddings", True)),
            max_seq_length=int(embedding.get("max_seq_length", 8192)),
        ),
        milvus=MilvusConfig(
            uri=str(milvus.get("uri", "http://localhost:19530")),
            token=str(os.getenv("MILVUS_TOKEN", milvus.get("token", ""))),
            collection_name=str(milvus.get("collection_name", "edu_rag_teacher_chunks")),
            vector_dim=int(milvus.get("vector_dim", 1024)),
            drop_existing=bool(milvus.get("drop_existing", False)),
            metric_type=str(milvus.get("metric_type", "COSINE")),
            index_type=str(milvus.get("index_type", "AUTOINDEX")),
            index_params=dict(milvus.get("index_params", {})),
            search_params=dict(milvus.get("search_params", {})),
        ),
        qa=QAConfig(
            provider=str(qa.get("provider", "dashscope")),
            api_key=os.getenv("DASHSCOPE_API_KEY", str(qa.get("api_key", ""))),
            base_url=str(qa.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            model=str(qa.get("model", "qwen-plus")),
            temperature=float(qa.get("temperature", 0.2)),
            top_k=int(qa.get("top_k", 5)),
            max_context_chars=int(qa.get("max_context_chars", 6000)),
            timeout_seconds=int(qa.get("timeout_seconds", 120)),
        ),
        hybrid_search=HybridSearchConfig(
            enabled=bool(hybrid_search.get("enabled", True)),
            keyword_top_k=int(hybrid_search.get("keyword_top_k", 20)),
            candidate_multiplier=int(hybrid_search.get("candidate_multiplier", 4)),
            rrf_k=int(hybrid_search.get("rrf_k", 60)),
            vector_weight=float(hybrid_search.get("vector_weight", 1.0)),
            keyword_weight=float(hybrid_search.get("keyword_weight", 1.0)),
        ),
        rerank=RerankConfig(
            enabled=bool(rerank.get("enabled", False)),
            model_name=str(rerank.get("model_name", "")),
            device=str(rerank.get("device", "cpu")),
            batch_size=int(rerank.get("batch_size", 16)),
            max_candidates=int(rerank.get("max_candidates", 20)),
        ),
        memory=MemoryConfig(
            redis_url=os.getenv("REDIS_URL", str(memory.get("redis_url", "redis://localhost:6379/0"))),
            namespace=str(memory.get("namespace", "edu_rag")),
            recent_message_limit=int(memory.get("recent_message_limit", 12)),
            session_ttl_seconds=int(memory.get("session_ttl_seconds", 604800)),
            long_term_limit=int(memory.get("long_term_limit", 20)),
        ),
        database=DatabaseConfig(
            url=os.getenv("DATABASE_URL", str(database.get("url", "sqlite:///data/education_rag.db"))),
            echo=bool(database.get("echo", False)),
        ),
        metadata_defaults=dict(raw.get("metadata_defaults", {})),
    )
