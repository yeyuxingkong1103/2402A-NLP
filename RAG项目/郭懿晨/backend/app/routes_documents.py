from pathlib import Path

from fastapi import APIRouter

from backend.app.config import AppSettings
from backend.app.embeddings import BgeM3Embedder
from backend.app.storage import JsonStateStore
from backend.app.vector_store import QdrantVectorStore


router = APIRouter(tags=["documents"])


def get_document_store() -> JsonStateStore:
    """获取文档存储。"""
    return JsonStateStore(Path("data/state.json"))


def get_vector_store() -> QdrantVectorStore:
    """获取向量库。"""
    settings = AppSettings()
    return QdrantVectorStore(settings.qdrant_path, settings.qdrant_collection)


@router.get("/api/documents")
def list_documents() -> list[dict]:
    """列出文档。"""
    return [document.model_dump(mode="json") for document in get_document_store().list_documents()]


@router.get("/api/vector-store")
def list_vector_store() -> list[dict]:
    """列出向量库元数据。"""
    store = get_vector_store()
    return [record.model_dump(mode="json") for record in store.client.scroll(collection_name=store.collection_name, with_payload=True, limit=1000)[0]]


@router.get("/api/categories")
def list_categories() -> list[str]:
    """列出 MinerU 标签类别。"""
    return get_vector_store().list_categories()
