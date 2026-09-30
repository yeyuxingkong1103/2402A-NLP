"""幂等文档索引编排器。

本模块负责把“文本块 → 向量 → Milvus → MySQL 元数据”串成一个可恢复流程。
Milvus 与 MySQL 无法组成真正的跨库事务，因此写入顺序和补偿删除非常重要：
先写向量，元数据失败时立即删除本次向量，尽量避免产生孤儿数据。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import replace
from typing import Any, Sequence

from app.ingestion.chunk_models import Chunk
from app.ingestion.embedding import EmbeddingClient
from app.ingestion.indexer_models import (
    ExistingDocumentStore,
    IndexingError,
    IndexingResult,
    VectorStore,
    find_document_by_hash,
    find_document_by_name,
)
from app.ingestion.metadata_store import SqlAlchemyMetadataStore
from app.ingestion.parser import ParsedDocument
from app.ingestion.vector_stores import InMemoryVectorStore, MilvusVectorStore

logger = logging.getLogger(__name__)


class IngestionIndexer:
    """Coordinate embeddings, deduplication, vector writes, and metadata writes."""

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        metadata_store: ExistingDocumentStore | None = None,
        collection_name: str = "kf_chunks",
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.collection_name = collection_name

    def index(
        self,
        document: ParsedDocument,
        chunks: Sequence[Chunk],
        *,
        document_id: int | str,
        tenant_id: int = 1,
        role_id: int = 0,
        knowledge_base_id: int = 1,
        document_version: int = 1,
        is_published: bool = False,
        force: bool = False,
        vectors: Sequence[Sequence[float]] | None = None,
    ) -> IndexingResult:
        """索引一个文档，并执行去重、维度校验和失败补偿。"""
        # 1. 非强制模式先按“知识库 + 文件 SHA256”检查，重复文档直接返回已有 ID。
        # 去重必须限定 knowledge_base_id，同一文件允许存在于不同知识库。
        if self.metadata_store and not force:
            existing = find_document_by_hash(
                self.metadata_store, document.file_sha256, knowledge_base_id
            )
            if existing:
                logger.info(
                    "document skipped because SHA-256 already exists",
                    extra={"sha256": document.file_sha256},
                )
                return IndexingResult(
                    existing["id"], document.file_sha256, 0, len(chunks), 0, self.collection_name
                )

        if not chunks:
            raise IndexingError("cannot index a document without chunks")

        # 2. 把租户、角色和知识库信息写入文档元数据，后续检索才能做服务端范围过滤。
        document = self._with_persistence_metadata(
            document, tenant_id, role_id, knowledge_base_id
        )

        # 3. 支持调用方传入预计算向量；否则统一批量调用 Embedding 服务。
        vectors = list(vectors) if vectors is not None else self._embed(chunks)
        self._validate_vectors(vectors, len(chunks))

        # 4. force 模式先记录旧版本。新版本完整写入后才淘汰旧版本，避免更新中途无数据可用。
        previous = self._find_previous(document, knowledge_base_id, force)
        deleted = self.vector_store.delete_by_document(document_id) if force else 0
        records = self._records(
            document_id=document_id,
            chunks=chunks,
            vectors=vectors,
            tenant_id=tenant_id,
            role_id=role_id,
            knowledge_base_id=knowledge_base_id,
            document_version=document_version,
            is_published=is_published,
        )

        # 5. 先写 Milvus，再写 MySQL；MySQL 写入失败时由下方方法补偿删除向量。
        self.vector_store.upsert(records)
        self._save_metadata_or_rollback(document, document_id, chunks, vectors)

        # 6. 新版本已完整落地后，再退休同名旧文档及其向量。
        if previous and str(previous["id"]) != str(document_id):
            self._retire_previous(previous["id"])
            deleted += self.vector_store.delete_by_document(previous["id"])
        logger.info(
            "document indexed", extra={"document_id": document_id, "inserted": len(records)}
        )
        return IndexingResult(
            document_id, document.file_sha256, len(records), 0, deleted, self.collection_name
        )

    @staticmethod
    def _with_persistence_metadata(
        document: ParsedDocument, tenant_id: int, role_id: int, knowledge_base_id: int
    ) -> ParsedDocument:
        metadata = {
            **document.metadata,
            "knowledge_base_id": knowledge_base_id,
            "tenant_id": tenant_id,
            "role_id": role_id,
        }
        return replace(document, metadata=metadata) if metadata != document.metadata else document

    def _embed(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        """Embed only when the caller did not provide precomputed vectors."""
        return self.embedding_client.embed_texts([chunk.text for chunk in chunks])

    def _validate_vectors(
        self, vectors: Sequence[Sequence[float]], chunk_count: int
    ) -> None:
        if len(vectors) != chunk_count:
            raise IndexingError("embedding count does not match chunk count")
        if any(len(vector) != self.embedding_client.dimension for vector in vectors):
            raise IndexingError("embedding dimension does not match client dimension")

    def _find_previous(
        self, document: ParsedDocument, knowledge_base_id: int, force: bool
    ) -> dict[str, Any] | None:
        if not force or not self.metadata_store:
            return None
        previous = find_document_by_name(
            self.metadata_store, document.file_name, knowledge_base_id
        )
        return previous or find_document_by_hash(
            self.metadata_store, document.file_sha256, knowledge_base_id
        )

    def _records(
        self,
        *,
        document_id: int | str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        tenant_id: int,
        role_id: int,
        knowledge_base_id: int,
        document_version: int,
        is_published: bool,
    ) -> list[dict[str, Any]]:
        timestamp = int(time.time() * 1000)
        return [
            {
                "id": chunk.chunk_id,
                "vector": vector,
                "text": chunk.text,
                "source": chunk.source,
                "doc_id": str(document_id),
                "chunk_id": chunk.chunk_id,
                "summary": chunk.summary,
                "created_at": timestamp,
                "updated_at": timestamp,
                "tenant_id": tenant_id,
                "role_id": role_id,
                "kb_id": knowledge_base_id,
                "document_version": document_version,
                "is_published": is_published,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def _save_metadata_or_rollback(
        self,
        document: ParsedDocument,
        document_id: int | str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """保存关系库元数据；失败时补偿删除已写入的向量。"""
        if not self.metadata_store:
            return
        try:
            self.metadata_store.save_document(document, document_id)
            self.metadata_store.save_chunks(document_id, chunks, vectors)
        except Exception as exc:
            self.vector_store.delete_by_document(document_id)
            self._try_retire(document_id)
            raise IndexingError("metadata persistence failed after vector upsert") from exc

    def _retire_previous(self, document_id: int | str) -> None:
        self._try_retire(document_id)

    def _try_retire(self, document_id: int | str) -> None:
        """Best-effort cleanup for partial or superseded metadata."""
        if not self.metadata_store:
            return
        mark_deleted = getattr(self.metadata_store, "mark_document_deleted", None)
        if mark_deleted is None:
            return
        try:
            mark_deleted(document_id)
        except Exception:
            logger.exception(
                "failed to retire document metadata", extra={"document_id": document_id}
            )


def build_milvus_store_from_env(embedding_dimension: int | None = None) -> MilvusVectorStore:
    """Build a Milvus vector store from environment configuration."""
    return MilvusVectorStore(
        uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        collection_name=os.getenv("MILVUS_COLLECTION", "kf_chunks"),
        dimension=embedding_dimension or int(os.getenv("EMBEDDING_DIM", "1024")),
        token=os.getenv("MILVUS_TOKEN") or None,
        enable_sparse=os.getenv("MILVUS_ENABLE_SPARSE", "false").lower() == "true",
    )


__all__ = [
    "ExistingDocumentStore",
    "IngestionIndexer",
    "InMemoryVectorStore",
    "IndexingError",
    "IndexingResult",
    "MilvusVectorStore",
    "SqlAlchemyMetadataStore",
    "VectorStore",
    "build_milvus_store_from_env",
]
