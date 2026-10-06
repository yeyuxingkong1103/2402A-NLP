from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "legal-rag"
    app_env: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000

    database_url: str = "mysql+pymysql://law_user:law_password@127.0.0.1:3306/law_rag"
    redis_url: str = "redis://127.0.0.1:6379/0"

    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_document_collection: str = "legal_document_chunks_v2"
    milvus_memory_collection: str = "user_long_term_memories"
    milvus_embedding_dim: int = 1024

    jwt_secret_key: str = "please-change-this-secret-to-a-long-random-string"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_timeout_seconds: int = 120

    dashscope_api_key: str = ""
    qwen_vl_model: str = "qwen-vl-max"

    mineru_api_url: str = ""
    mineru_api_key: str = ""
    mineru_timeout_seconds: int = 180
    mineru_fallback_to_local: bool = True

    embedding_model_path: str = "model/bge-m3"
    reranker_model_path: str = "model/bge-reranker-large"
    model_device: str = "cpu"
    embedding_batch_size: int = 16
    reranker_batch_size: int = 8

    vector_top_k: int = 20
    bm25_top_k: int = 20
    rrf_result_top_k: int = 10
    rerank_top_k: int = 5
    rrf_k: int = 60

    data_dir: str = "data"
    max_chunk_length: int = 512
    min_chunk_length: int = 10
    short_memory_ttl_seconds: int = 86400
    short_memory_window: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @property
    def data_root(self) -> Path:
        return self.project_root / self.data_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
