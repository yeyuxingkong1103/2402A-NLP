from uuid import uuid4

from backend.app.embeddings import EmbeddingResult
from backend.app.models import Chunk
from backend.app.vector_store import QdrantVectorStore


class TextAwareEmbedder:
    def __init__(self) -> None:
        self.vectors = {
            "第一段": EmbeddingResult(dense=[1.0, 0.0, 0.0], sparse_indices=[0], sparse_values=[1.0]),
            "第二段": EmbeddingResult(dense=[0.0, 1.0, 0.0], sparse_indices=[1], sparse_values=[1.0]),
            "第一段查询": EmbeddingResult(dense=[1.0, 0.0, 0.0], sparse_indices=[0], sparse_values=[1.0]),
        }

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.vectors[text] for text in texts]


def test_qdrant_vector_store_search_returns_real_results(tmp_path):
    store = QdrantVectorStore(tmp_path / "qdrant", "rag_documents_test", dense_size=3)
    chunks = [
        Chunk(
            chunk_id=str(uuid4()),
            document_id="doc-1",
            page=1,
            category="正文",
            text="第一段",
            source_span="page=1:block=1",
        ),
        Chunk(
            chunk_id=str(uuid4()),
            document_id="doc-1",
            page=2,
            category="表格",
            text="第二段",
            source_span="page=2:block=1",
        ),
    ]

    embedder = TextAwareEmbedder()
    store.upsert_chunks(chunks, embedder)

    results = store.search("第一段查询", embedder, limit=1)

    assert results[0].chunk.page == 1
    assert results[0].chunk.text == "第一段"
    assert set(store.list_categories()) == {"正文", "表格"}
