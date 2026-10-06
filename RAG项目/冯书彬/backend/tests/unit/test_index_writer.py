from datetime import datetime

from backend.app.core.config import AppSettings
from backend.app.ingestion.index_writer import index_materials
from backend.app.models.knowledge_base import KnowledgeMaterial


class _EmbeddingClient:
    def embed_texts(self, texts):
        self.texts = texts
        return [[0.1, 0.2] for _ in texts]


class _VectorStore:
    def __init__(self):
        self.dimension = None
        self.rows = []

    def ensure_collection(self, dimension):
        self.dimension = dimension

    def upsert(self, rows):
        self.rows = rows
        return len(rows)


def _material(status="published", searchable=True):
    return KnowledgeMaterial(
        id="m1",
        snapshot_id="s1",
        source_url="https://example.test/law",
        publisher="最高人民法院",
        material_type="judicial_interpretation",
        raw_text="婚姻家庭法律材料正文",
        attachments=[],
        status=status,
        searchable=searchable,
        created_at=datetime(2026, 9, 18),
        updated_at=datetime(2026, 9, 18),
    )


def test_index_materials_writes_only_published_searchable_materials():
    embeddings = _EmbeddingClient()
    store = _VectorStore()

    written = index_materials(
        [_material(), _material(status="pending_review"), _material(searchable=False)],
        embedding_client=embeddings,
        vector_store=store,
        app_settings=AppSettings(MILVUS_VECTOR_DIMENSION=2),
    )

    assert written == 1
    assert store.dimension == 2
    assert embeddings.texts == ["婚姻家庭法律材料正文"]
    assert store.rows[0]["material_id"] == "m1"
    metadata = store.rows[0]["metadata"]
    assert metadata["status"] == "published"
    assert metadata["searchable"] is True
    assert metadata["source_url"] == "https://example.test/law"
