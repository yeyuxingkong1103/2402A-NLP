"""集中配置：读取 .env，支持多环境（dev / test / prod）。
全局配置
所有密钥与连接信息均通过环境变量注入，代码中零硬编码。
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== 应用 =====
    app_env: str = "dev"  # dev | test | prod
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    api_base: str = "http://127.0.0.1:8000"
    cors_origins: str = "*"

    # ===== 大模型 =====
    llm_provider: str = "deepseek"
    llm_api_key: str = ""
    llm_base_url: str = ""  # 为空时回退到 provider 预设
    llm_model: str = ""     # 为空时回退到 provider 预设
    llm_temperature: float = 0.5
    llm_max_tokens: int = 2048

    # ===== 向量 / 重排模型 =====
    bge_m3_path: str = "/mnt/d/model/bge-m3"
    bge_reranker_path: str = "/mnt/d/model/bge-reranker-base"
    embed_dim: int = 1024

    # ===== Milvus =====
    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    milvus_collection: str = "rag_knowledge"
    milvus_native_bm25: bool = True  # Milvus 2.5+ 原生 BM25，否则回退 rank_bm25

    # ===== Redis =====
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    memory_max_turns: int = 10
    memory_ttl: int = 3600

    # ===== MySQL =====
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "student"
    mysql_password: str = "abc12345"
    mysql_db: str = "testdb"

    # ===== MongoDB（长期记忆 / 原文归档）=====
    mongo_enabled: bool = True
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "rag"

    # ===== Neo4j（知识图谱，可选）=====
    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"

    # ===== RAG 引擎与检索参数 =====
    rag_engine: str = "native"  # native | langchain | llamaindex
    retrieve_top_k: int = 10
    rerank_top_k: int = 5
    score_threshold: float = 0.3
    enable_query_rewrite: bool = True
    enable_summary: bool = True
    enable_rerank: bool = True

    # ===== 分块 =====
    chunk_strategy: str = "sentence"  # fixed | sentence | paragraph | heading | semantic
    chunk_size: int = 500
    chunk_overlap: int = 50
    parent_child: bool = True  # 父子块检索
    max_summary_chunks: int = 50  # 单文档生成 LLM 摘要的最大块数
    enable_augment: bool = False  # 知识库数据增强（生成假设问题/扩写）
    augment_max: int = 20

    # ===== 知识库 =====
    laws_dir: str = "laws"
    auto_ingest: bool = True

    @property
    def is_prod(self) -> bool:
        return self.app_env.lower() == "prod"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
