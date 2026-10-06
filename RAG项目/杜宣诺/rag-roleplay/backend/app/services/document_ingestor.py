"""文档入库：解析 → 分块 → 向量化 → 写入 Milvus documents collection。"""
import time

from ..config import get_settings
from .document_parser import chunk_text, extract_text


class DocumentIngestor:
    def __init__(self, embedding, milvus):
        self.embedding = embedding
        self.milvus = milvus

    async def ingest(self, doc_id: int, filename: str, data: bytes) -> int:
        settings = get_settings()
        text = extract_text(filename, data)
        chunks = chunk_text(text, settings.doc_chunk_size, settings.doc_chunk_overlap)
        # 幂等：先清空该 doc 的旧 chunk，再重写，避免重试/重复触发导致堆积
        await self.milvus.delete_by_filter("documents", f"doc_id == {doc_id}")
        if not chunks:
            return 0
        now = int(time.time())
        rows = [
            {
                "doc_id": doc_id,
                "chunk_index": i,
                "source": filename,
                "text": c,
                "created_at": now,
            }
            for i, c in enumerate(chunks)
        ]
        await self.milvus.upsert_documents(rows)
        return len(chunks)
