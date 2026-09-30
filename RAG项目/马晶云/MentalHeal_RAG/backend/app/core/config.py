from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "MentalHeal RAG"
    secret_key: str = "change-this-secret-key"
    access_token_expire_minutes: int = 1440
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048
    llm_timeout_seconds: int = 60
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    embedding_model_name: str = "BAAI/bge-m3"
    rerank_model_name: str = "BAAI/bge-reranker-v2-m3"
    embedding_batch_size: int = 16
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "mentalheal_rag"
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    chat_history_max_rounds: int = 10
    chat_history_ttl_seconds: int = 86400
    rag_cache_ttl_seconds: int = 600
    document_status_ttl_seconds: int = 86400
    upload_max_size_mb: int = 100
    raw_data_dir: Path = PROJECT_ROOT / "data/raw"
    processed_data_dir: Path = PROJECT_ROOT / "data/processed"
    cleaned_data_dir: Path = PROJECT_ROOT / "data/cleaned"
    chunks_data_dir: Path = PROJECT_ROOT / "data/chunks"
    vectorized_data_dir: Path = PROJECT_ROOT / "data/vectorized"
    failed_data_dir: Path = PROJECT_ROOT / "data/failed"
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_collection_knowledge: str = "knowledge_chunks"
    rag_top_k: int = 20
    bm25_top_k: int = 20
    rag_rerank_top_k: int = 5
    rag_rerank_candidates: int = 8
    rag_score_threshold: float = 0.35
    rag_context_max_chars: int = 8000
    rag_vector_weight: float = 0.25
    rag_rerank_weight: float = 0.30
    rag_lexical_weight: float = 0.45
    rag_bm25_weight: float = 0.20
    rag_max_chunks_per_document: int = 2
    rag_max_chunks_per_page: int = 2

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def mysql_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}?charset=utf8mb4"
        )

    @property
    def embedding_model_path(self) -> Path:
        return Path(self.embedding_model_name).expanduser()

    @property
    def rerank_model_path(self) -> Path:
        return Path(self.rerank_model_name).expanduser()


@lru_cache

def get_settings() -> Settings:
    return Settings()
