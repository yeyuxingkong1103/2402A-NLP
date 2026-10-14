from dataclasses import dataclass
import os


@dataclass(frozen=True)
class VectorStoreConfig:
    source_pdf: str = os.getenv("SOURCE_PDF", "D:/RAG工单附件/招股说明书1.pdf")
    persist_directory: str = os.getenv("VECTORSTORE_DIR", "D:/rag-model-cache/news-chroma")
    collection_name: str = os.getenv("COLLECTION_NAME", "news_rag")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


config = VectorStoreConfig()
