from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "rag-roleplay"

    database_url: str = "mysql+asyncmy://root:password@localhost:3306/roleplay"
    redis_url: str = "redis://localhost:6379/0"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_connect_retries: int = 10
    milvus_connect_interval: float = 3.0

    jwt_secret: str = "change-me-in-production-please-set-a-long-random-secret"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7

    llm_provider: str = "openai_compat"
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"

    embedding_base_url: str = "http://localhost:8080"
    rerank_base_url: str = "http://localhost:8081"

    short_term_rounds: int = 20
    retrieval_top_k: int = 20
    rerank_top_m: int = 5
    doc_upload_dir: str = "./data/uploads"
    max_upload_bytes: int = 20 * 1024 * 1024
    doc_chunk_size: int = 800
    doc_chunk_overlap: int = 100
    doc_top_k: int = 5
    ocr_enabled: bool = False
    memory_extract_every_n_rounds: int = 20
    memory_extract_idle_seconds: int = 30
    memory_dedup_threshold: float = 0.92

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
