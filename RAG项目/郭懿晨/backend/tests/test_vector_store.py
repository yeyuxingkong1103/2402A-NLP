from backend.app.embeddings import EmbeddingResult, FakeBgeM3Embedder
from backend.app.models import Chunk
from backend.app.vector_store import InMemoryVectorStore, QdrantVectorStore


def make_chunk(chunk_id: str, text: str, category: str = "正文") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        page=1,
        category=category,
        text=text,
        source_span="page=1:block=1",
    )


def test_in_memory_vector_store_upserts_and_lists_categories():
    store = InMemoryVectorStore()
    chunks = [make_chunk("c1", "第一段", "正文"), make_chunk("c2", "表格", "表格")]

    store.upsert_chunks(chunks, FakeBgeM3Embedder())

    assert store.list_categories() == ["正文", "表格"]
    assert len(store.list_records()) == 2


class FailingQueryClient:
    def __init__(self) -> None:
        self.scroll_calls = 0

    def get_collections(self):
        return type("Collections", (), {"collections": [type("Item", (), {"name": "rag_documents"})()]})()

    def create_collection(self, *args, **kwargs):
        raise AssertionError("should not create collection")

    def upsert(self, *args, **kwargs):
        raise AssertionError("should not upsert")

    def query_points(self, *args, **kwargs):
        raise MemoryError("out of memory")

    def scroll(self, *args, **kwargs):
        self.scroll_calls += 1
        point = type(
            "Point",
            (),
            {
                "payload": {
                    "chunk_id": "c1",
                    "document_id": "doc-1",
                    "page": 1,
                    "category": "正文",
                    "text": "第一段",
                    "source_span": "page=1:block=1",
                }
            },
        )()
        return [point], None


class DummyEmbedder:
    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        return [EmbeddingResult(dense=[1.0, 0.0, 0.0], sparse_indices=[0], sparse_values=[1.0]) for _ in texts]


def test_qdrant_vector_store_reuses_client_for_same_path(tmp_path):
    first_store = QdrantVectorStore(tmp_path / "qdrant", "rag_documents", dense_size=3)
    second_store = QdrantVectorStore(tmp_path / "qdrant", "rag_documents", dense_size=3)

    assert first_store.client is second_store.client


class ReturningQueryClient(FailingQueryClient):
    def query_points(self, *args, **kwargs):
        assert kwargs.get("with_vectors") is True
        point = type(
            "Point",
            (),
            {
                "payload": {
                    "chunk_id": "c1",
                    "document_id": "doc-1",
                    "page": 1,
                    "category": "正文",
                    "text": "第一段",
                    "source_span": "page=1:block=1",
                },
                "score": 0.8,
                "vector": {"dense": [0.1, 0.2, 0.3]},
            },
        )()
        return type("Result", (), {"points": [point]})()


def test_qdrant_search_returns_dense_when_with_vectors(tmp_path):
    store = QdrantVectorStore(tmp_path / "qdrant", "rag_documents", dense_size=3)
    store.client = ReturningQueryClient()

    results = store.search("问题", DummyEmbedder(), limit=5, with_vectors=True)

    assert results[0].dense == [0.1, 0.2, 0.3]

