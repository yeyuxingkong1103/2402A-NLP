from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    pdf_path: str = os.getenv("NEWS_PDF", "D:/RAG工单附件/招股说明书1.pdf")
    persist_directory: str = os.getenv("CHROMA_DIR", "D:/rag-model-cache/news-chroma")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")


config = Config()
