from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 所有默认路径都以项目根目录为基准，避免从不同工作目录启动时产生相对路径错误。


def _sqlite_database_url() -> str:
    # 开发环境默认使用项目 data 目录中的 SQLite 文件；生产环境可通过 .env 切换到 MySQL。
    return f"sqlite+aiosqlite:///{(PROJECT_ROOT / 'data' / 'rag_roleplay.db').as_posix()}"


class Settings(BaseSettings):
    """应用配置。字段名对应 .env 中的大写环境变量。"""
    project_root: Path = Field(default=PROJECT_ROOT, alias="PROJECT_ROOT")
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="DATA_DIR")
    upload_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "uploads", alias="UPLOAD_DIR"
    )
    index_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "indexes", alias="INDEX_DIR"
    )
    local_vector_index: Path = Field(
        default=PROJECT_ROOT / "data" / "indexes" / "chunks.json",
        alias="LOCAL_VECTOR_INDEX",
    )
    log_dir: Path = Field(default=PROJECT_ROOT / "logs", alias="LOG_DIR")

    app_name: str = Field(default="RAG Roleplay System", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    database_url: str = Field(default_factory=_sqlite_database_url, alias="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_enabled: bool = Field(default=False, alias="REDIS_ENABLED")

    milvus_enabled: bool = Field(default=False, alias="MILVUS_ENABLED")
    milvus_uri: str = Field(default="http://localhost:19530", alias="MILVUS_URI")
    milvus_token: str = Field(default="", alias="MILVUS_TOKEN")
    milvus_collection: str = Field(
        default="rag_roleplay_chunks", alias="MILVUS_COLLECTION"
    )

    embedding_provider: str = Field(default="hash", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    reranker_enabled: bool = Field(default=False, alias="RERANKER_ENABLED")
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL"
    )

    llm_provider: str = Field(default="mock", alias="LLM_PROVIDER")
    llm_base_url: str = Field(
        default="https://api.deepseek.com/v1", alias="LLM_BASE_URL"
    )
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="deepseek-chat", alias="LLM_MODEL")
    llm_timeout_seconds: int = Field(default=60, alias="LLM_TIMEOUT_SECONDS")
    ragas_enabled: bool = Field(default=False, alias="RAGAS_ENABLED")

    top_k_vector: int = Field(default=8, alias="TOP_K_VECTOR")
    top_k_bm25: int = Field(default=8, alias="TOP_K_BM25")
    top_k_final: int = Field(default=6, alias="TOP_K_FINAL")
    vector_score_threshold: float = Field(default=0.05, alias="VECTOR_SCORE_THRESHOLD")
    chunk_size: int = Field(default=600, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=80, alias="CHUNK_OVERLAP")
    max_memory_messages: int = Field(default=12, alias="MAX_MEMORY_MESSAGES")
    rag_global_knowledge_categories: str = Field(
        default="doctor", alias="RAG_GLOBAL_KNOWLEDGE_CATEGORIES"
    )

    @property
    def rag_global_categories(self) -> set[str]:
        return {
            item.strip().lower()
            for item in self.rag_global_knowledge_categories.split(",")
            if item.strip()
        }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
