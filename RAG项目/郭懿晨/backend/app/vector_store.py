from pathlib import Path
from typing import Protocol

from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from backend.app.embeddings import TextEmbedder
from backend.app.models import Chunk


class SearchResult(BaseModel):
    """检索结果。"""

    chunk: Chunk
    score: float
    dense: list[float] | None = None


class VectorRecord(BaseModel):
    """向量库记录摘要。"""

    chunk_id: str
    document_id: str
    page: int
    category: str
    text: str


class VectorStore(Protocol):
    """向量库协议。"""

    def upsert_chunks(self, chunks: list[Chunk], embedder: TextEmbedder) -> None:
        """写入 chunk。"""

    def search(self, query: str, embedder: TextEmbedder, limit: int, with_vectors: bool = False) -> list[SearchResult]:
        """检索 chunk。"""


class InMemoryVectorStore:
    """测试用内存向量库。"""

    def __init__(self) -> None:
        self.records: list[VectorRecord] = []
        self.chunks: list[Chunk] = []

    def upsert_chunks(self, chunks: list[Chunk], embedder: TextEmbedder) -> None:
        """写入 chunk。"""
        embedder.embed_texts([chunk.text for chunk in chunks])
        self.chunks.extend(chunks)
        self.records.extend(
            VectorRecord(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                page=chunk.page,
                category=chunk.category,
                text=chunk.text,
            )
            for chunk in chunks
        )

    def search(self, query: str, embedder: TextEmbedder, limit: int, with_vectors: bool = False) -> list[SearchResult]:
        """返回稳定测试检索结果。"""
        embedder.embed_texts([query])
        return [SearchResult(chunk=chunk, score=1.0) for chunk in self.chunks[:limit]]

    def list_records(self) -> list[VectorRecord]:
        """列出向量记录。"""
        return self.records

    def list_categories(self) -> list[str]:
        """列出类别。"""
        return sorted({record.category for record in self.records})


_QDRANT_CLIENT_CACHE: dict[str, QdrantClient] = {}


class QdrantVectorStore:
    """Qdrant 本地向量库。"""

    def __init__(self, path: Path, collection_name: str, dense_size: int = 1024) -> None:
        cache_key = str(path.resolve())
        client = _QDRANT_CLIENT_CACHE.get(cache_key)
        if client is None:
            client = QdrantClient(path=str(path))
            _QDRANT_CLIENT_CACHE[cache_key] = client
        self.client = client
        self.collection_name = collection_name
        self.dense_size = dense_size
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """确保 collection 存在。"""
        collections = self.client.get_collections().collections
        if any(item.name == self.collection_name for item in collections):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={"dense": models.VectorParams(size=self.dense_size, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

    def upsert_chunks(self, chunks: list[Chunk], embedder: TextEmbedder) -> None:
        """写入 chunk 到 Qdrant。"""
        embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            points.append(
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector={
                        "dense": embedding.dense,
                        "sparse": models.SparseVector(
                            indices=embedding.sparse_indices,
                            values=embedding.sparse_values,
                        ),
                    },
                    payload=chunk.model_dump(),
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(
        self,
        query: str,
        embedder: TextEmbedder,
        limit: int,
        with_vectors: bool = False,
    ) -> list[SearchResult]:
        """执行 dense+sparse 检索并用 RRF 融合。"""
        query_embedding = embedder.embed_texts([query])[0]
        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=query_embedding.dense, using="dense", limit=limit),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=query_embedding.sparse_indices,
                            values=query_embedding.sparse_values,
                        ),
                        using="sparse",
                        limit=limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_vectors=with_vectors,
            )
            return [
                SearchResult(
                    chunk=Chunk.model_validate(point.payload),
                    score=float(point.score or 0.0),
                    dense=self._extract_dense(point.vector) if with_vectors else None,
                )
                for point in result.points
            ]
        except MemoryError:
            return self._fallback_search(limit)

    @staticmethod
    def _extract_dense(vector: object) -> list[float] | None:
        """从 Qdrant 命名向量中提取 dense。"""
        if isinstance(vector, dict):
            dense = vector.get("dense")
            if isinstance(dense, list):
                return [float(value) for value in dense]
        return None

    def _fallback_search(self, limit: int) -> list[SearchResult]:
        """在本地混合检索内存不足时，退回到轻量 scroll。"""
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            with_vectors=False,
            limit=limit,
        )
        return [
            SearchResult(chunk=Chunk.model_validate(point.payload), score=0.0)
            for point in points
            if point.payload
        ]

    def list_categories(self) -> list[str]:
        """列出已入库类别。"""
        results = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            limit=1000,
        )[0]
        return sorted({str(point.payload.get("category", "")) for point in results if point.payload})
