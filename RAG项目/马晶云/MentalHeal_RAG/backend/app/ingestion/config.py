from dataclasses import dataclass
from pathlib import Path
import os


# 用 dataclass 集中保存文档入库流程需要的所有配置。
# frozen=True 表示创建后不能随意修改，避免运行过程中配置被意外改掉。
@dataclass(frozen=True)
class IngestionConfig:
    raw_dir: Path
    processed_dir: Path
    cleaned_dir: Path
    chunks_dir: Path
    vectorized_dir: Path
    failed_dir: Path
    mineru_api_key: str
    mineru_base_url: str
    mineru_api_version: str
    mineru_model_version: str
    mineru_poll_seconds: int
    mineru_timeout_seconds: int
    mineru_is_ocr: bool
    enable_ocr: bool
    ocr_language: str
    ocr_use_gpu: bool
    enable_vision_ocr: bool
    chunk_size: int
    chunk_overlap: int
    embedding_model_name: str
    embedding_dim: int
    embedding_batch_size: int
    milvus_host: str
    milvus_port: int
    milvus_collection_knowledge: str

    @classmethod
    def from_env(cls) -> "IngestionConfig":
        # 从环境变量读取配置；如果没有设置，就使用后面的默认值。
        # 这样同一套代码可以在不同电脑或服务器上使用不同目录和服务地址。
        return cls(
            raw_dir=Path(os.getenv("RAW_DATA_DIR", "./data/raw")),
            processed_dir=Path(os.getenv("PROCESSED_DATA_DIR", "./data/processed")),
            cleaned_dir=Path(os.getenv("CLEANED_DATA_DIR", "./data/cleaned")),
            chunks_dir=Path(os.getenv("CHUNKS_DATA_DIR", "./data/chunks")),
            vectorized_dir=Path(os.getenv("VECTORIZED_DATA_DIR", "./data/vectorized")),
            failed_dir=Path(os.getenv("FAILED_DATA_DIR", "./data/failed")),
            mineru_api_key=os.getenv("MINERU_API_KEY", ""),
            mineru_base_url=os.getenv("MINERU_BASE_URL", "https://mineru.net/api").rstrip("/"),
            mineru_api_version=os.getenv("MINERU_API_VERSION", "v4"),
            mineru_model_version=os.getenv("MINERU_MODEL_VERSION", "pipeline"),
            mineru_poll_seconds=int(os.getenv("MINERU_POLL_SECONDS", "5")),
            mineru_timeout_seconds=int(os.getenv("MINERU_TIMEOUT_SECONDS", "900")),
            mineru_is_ocr=os.getenv("MINERU_IS_OCR", "true").lower() == "true",
            enable_ocr=os.getenv("ENABLE_OCR", "true").lower() == "true",
            ocr_language=os.getenv("PADDLEOCR_LANG", "ch"),
            ocr_use_gpu=os.getenv("PADDLEOCR_USE_GPU", "false").lower() == "true",
            enable_vision_ocr=os.getenv("ENABLE_PADDLEOCR_VL", "true").lower() == "true",
            chunk_size=int(os.getenv("INGESTION_CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("INGESTION_CHUNK_OVERLAP", "160")),
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
            milvus_host=os.getenv("MILVUS_HOST", "127.0.0.1"),
            milvus_port=int(os.getenv("MILVUS_PORT", "19530")),
            milvus_collection_knowledge=os.getenv("MILVUS_COLLECTION_KNOWLEDGE", "knowledge_chunks"),
        )

    def validate(self) -> None:
        # MinerU 是 PDF 解析阶段必须用到的服务，所以 API key 不能为空。
        if not self.mineru_api_key or self.mineru_api_key.startswith("your-"):
            raise ValueError("MINERU_API_KEY is required for PDF ingestion")
        # overlap 必须小于 chunk_size，否则切块时会出现无效参数。
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("INGESTION_CHUNK_OVERLAP must be smaller than INGESTION_CHUNK_SIZE")
