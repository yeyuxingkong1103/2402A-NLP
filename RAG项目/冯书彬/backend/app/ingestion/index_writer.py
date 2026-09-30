import hashlib
import logging
from typing import Any

from backend.app.core.config import AppSettings, settings
from backend.app.database.milvus import MilvusVectorStore, create_milvus_store
from backend.app.embeddings.base import EmbeddingClient
from backend.app.embeddings.embedding_factory import get_embedding_client
from backend.app.models.knowledge_base import KnowledgeMaterial

logger = logging.getLogger(__name__)


def index_materials(
    materials: list[KnowledgeMaterial],
    embedding_client: EmbeddingClient | None = None,
    vector_store: MilvusVectorStore | None = None,
    app_settings: AppSettings = settings,
) -> int:
    # 只索引已发布且可检索的正式材料，避免抓取快照或审核中内容进入生产检索。
    searchable = [material for material in materials if material.status == "published" and material.searchable]
    if not searchable:
        logger.info("没有可写入 Milvus 的知识库材料")
        return 0
    embedding_client = embedding_client or get_embedding_client()
    vector_store = vector_store or create_milvus_store(app_settings)
    vector_store.ensure_collection(app_settings.MILVUS_VECTOR_DIMENSION)
    texts = [material.raw_text[:2000] for material in searchable]
    vectors = embedding_client.embed_texts(texts)
    rows = [_material_to_row(material, text, vector) for material, text, vector in zip(searchable, texts, vectors, strict=True)]
    written = vector_store.upsert(rows)
    logger.info("知识库材料向量索引完成", extra={"material_count": len(searchable), "written_count": written})
    return written


def _material_to_row(material: KnowledgeMaterial, text: str, vector: list[float]) -> dict[str, Any]:
    version_id = str(getattr(material, "version_id", material.id))
    metadata = {
        "source_url": material.source_url,
        "publisher": material.publisher,
        "material_type": material.material_type,
        "status": material.status,
        "searchable": material.searchable,
        "effective_from": material.effective_from,
        "article": getattr(material, "article", None),
        "paragraph": getattr(material, "paragraph", None),
        "relationship_types": getattr(material, "relationship_types", None),
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return {
        "id": f"{material.id}:{version_id}",
        "material_id": material.id,
        "version_id": version_id,
        "relationship_type": "general",
        "text": text,
        "metadata": metadata,
        "vector": vector,
    }
