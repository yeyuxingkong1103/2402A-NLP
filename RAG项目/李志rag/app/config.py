from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """项目统一配置。

    配置值优先从项目根目录的 .env 读取；这里的默认值用于说明类型和提供开发兜底。
    答辩时可按“应用、数据库、检索模型、文档解析、大模型、RAG 参数”六组讲解。
    """

    # 应用与认证
    app_env: str = "development"
    app_secret_key: str = "development-only-change-me"
    admin_username: str = "admin"
    admin_password: str = "ChangeMe123!"
    jwt_expire_hours: int = 24

    # 数据存储
    database_url: str = "mysql+pymysql://rag:rag_password@127.0.0.1:3307/rag"
    redis_url: str = "redis://127.0.0.1:6379/0"
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_collection: str = "medical_chunks_bge_m3"

    # 向量化与重排序模型
    embedding_model: str = (
        "D:/models_bge_ocr/huggingface/hub/models--BAAI--bge-m3/"
        "snapshots/5617a9f61b028005a4858fdac845db406aefb181"
    )
    rerank_model: str = "D:/models_bge_ocr/bge-reranker-base"
    huggingface_cache_dir: Path = Path("D:/models_bge_ocr/huggingface/hub")
    model_device: str = "cpu"
    rerank_enabled: bool = True
    embedding_dimension: int = 1024

    # PDF、OCR 与视觉理解
    ocr_enabled: bool = True
    paddlex_cache_dir: Path = Path("D:/models_bge_ocr/paddlex")
    paddleocr_vl_enabled: bool = True
    paddleocr_vl_model: Path = Path("D:/models_bge_ocr/PaddleOCR-VL-1.6")
    hf_modules_cache: Path = Path(".runtime_cache/hf_modules")
    paddleocr_vl_max_new_tokens: int = 256
    paddleocr_vl_max_pixels: int = 501760
    paddleocr_vl_max_images_per_page: int = 4
    watermark_removal_enabled: bool = True
    watermark_min_pages: int = 3
    watermark_page_ratio: float = 0.6
    watermark_max_chars: int = 80

    # DeepSeek 生成模型
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_enabled: bool = True
    general_knowledge_fallback: bool = True

    # 上传、分块与召回参数
    upload_dir: Path = Path("data/uploads")
    max_upload_mb: int = 20
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k_dense: int = 6
    top_k_sparse: int = 6
    top_k_final: int = 3
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
