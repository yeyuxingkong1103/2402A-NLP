from pathlib import Path
from uuid import uuid4

from app.core.config import Settings
from app.rag.embeddings import EmbeddingService
from app.rag.retrievers import HybridRetriever
from app.rag.types import ChunkRecord
from app.rag.vector_store import LocalVectorStore


def test_hybrid_retrieval_returns_lexical_match():
    settings = Settings(
        embedding_provider="hash",
        embedding_dimension=64,
        top_k_vector=5,
        top_k_bm25=5,
        top_k_final=3,
    )
    path = Path("data/indexes") / f"test_chunks_{uuid4().hex}.json"
    store = LocalVectorStore(str(path))
    embeddings = EmbeddingService(settings)
    records = [
        ChunkRecord("1", "doc-1", "高血压患者应进行规范血压监测。", "guide.pdf", {"role_id": "global"}),
        ChunkRecord("2", "doc-2", "英语口语练习需要坚持复述。", "english.md", {"role_id": "global"}),
    ]
    vectors = embeddings.embed_documents([record.text for record in records])
    for record, vector in zip(records, vectors):
        record.embedding = vector
    try:
        store.upsert(records)
        retriever = HybridRetriever(store, embeddings, settings)
        hits = retriever.retrieve("高血压监测", top_k=2)
        assert hits
        assert hits[0].chunk.id == "1"
    finally:
        path.unlink(missing_ok=True)
