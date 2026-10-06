from app.core.fakes import FakeEmbedding
from app.services.document_ingestor import DocumentIngestor


class _StubMilvus:
    def __init__(self):
        self.deleted = []
        self.upserted = []

    async def delete_by_filter(self, collection, filter_expr):
        self.deleted.append((collection, filter_expr))

    async def upsert_documents(self, rows):
        self.upserted.extend(rows)


async def test_ingest_clears_old_chunks_then_upserts():
    milvus = _StubMilvus()
    ingestor = DocumentIngestor(FakeEmbedding(), milvus)
    data = ("段落甲\n\n段落乙" * 50).encode("utf-8")
    n = await ingestor.ingest(doc_id=7, filename="a.txt", data=data)
    assert n > 0
    # 先清空旧 chunk 再重写，保证幂等
    assert milvus.deleted == [("documents", "doc_id == 7")]
    assert len(milvus.upserted) == n
    assert all(r["doc_id"] == 7 for r in milvus.upserted)
    # chunk 顺序连续且覆盖原文
    assert [r["chunk_index"] for r in milvus.upserted] == list(range(n))
    joined = "".join(r["text"] for r in milvus.upserted)
    assert "段落甲" in joined


async def test_ingest_empty_text_returns_zero_and_clears():
    milvus = _StubMilvus()
    ingestor = DocumentIngestor(FakeEmbedding(), milvus)
    n = await ingestor.ingest(doc_id=9, filename="a.txt", data=b"   \n\n  ")
    assert n == 0
    # 空文本也先清空，避免旧数据残留
    assert milvus.deleted == [("documents", "doc_id == 9")]
    assert milvus.upserted == []
