from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_name: str = "Prospectus RAG"
    environment: str = "development"
    api_prefix: str = "/api"
    admin_token: str = "change-me"
    database_url: str = "postgresql+psycopg://rag:rag@postgres:5432/rag"
    redis_url: str = "redis://redis:6379/0"
    milvus_uri: str = "http://milvus:19530"
    milvus_token: str = ""
    milvus_collection: str = "document_chunks"
    upload_dir: Path = Path("/data/uploads")
    parsed_dir: Path = Path("/data/parsed")
    pages_dir: Path = Path("/data/pages")
    embedding_model_path: str = "/models/bge-m3"
    reranker_model_path: str = "/models/bge-reranker-large"
    embedding_dimension: int = 1024
    use_local_models: bool = False
    mineru_api_url: str = ""
    mineru_base_url: str = ""
    mineru_api_key: str = ""
    mineru_timeout_seconds: float = 120.0
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_api_key: str = ""
    dashscope_vl_model: str = "qwen-vl-max"
    enable_vision: bool = False
    max_upload_mb: int = 100
    retrieval_top_k: int = 20
    rerank_top_k: int = 8
    max_context_chars: int = 18000

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for directory in (settings.upload_dir, settings.parsed_dir, settings.pages_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return settings
