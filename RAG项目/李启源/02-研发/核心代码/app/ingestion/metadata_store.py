"""SQLAlchemy metadata and sparse-search adapter for ingestion."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Sequence

from app.ingestion.chunk_models import Chunk
from app.ingestion.indexer_models import IndexingError
from app.ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)


class SqlAlchemyMetadataStore:
    """Optional synchronous SQLAlchemy writer matching the stage-2 schema."""

    def __init__(self, database_url: str) -> None:
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise IndexingError("SQLAlchemy is required for SqlAlchemyMetadataStore") from exc
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self._fulltext_available = self._ensure_fulltext_index()

    def _ensure_fulltext_index(self) -> bool:
        """Ensure legacy databases expose the sparse-search index."""
        from sqlalchemy import text

        try:
            with self.engine.begin() as connection:
                existing = connection.execute(
                    text(
                        "SELECT 1 FROM information_schema.statistics "
                        "WHERE table_schema = DATABASE() AND table_name = 'chunks' "
                        "AND index_name = 'ft_chunk_content' LIMIT 1"
                    )
                ).first()
                if existing:
                    return True
                connection.execute(
                    text("ALTER TABLE chunks ADD FULLTEXT KEY ft_chunk_content (content)")
                )
                logger.info("created missing MySQL FULLTEXT index: chunks.content")
                return True
        except Exception as exc:
            logger.warning("MySQL sparse search disabled; FULLTEXT unavailable: %s", exc)
            return False

    def find_document_by_hash(
        self, file_sha256: str, knowledge_base_id: int | None = None
    ) -> dict[str, Any] | None:
        """Find the newest active document with the same content hash."""
        from sqlalchemy import text

        condition = "file_sha256=:sha AND status <> 'deleted'"
        params: dict[str, Any] = {"sha": file_sha256}
        if knowledge_base_id is not None:
            condition += " AND knowledge_base_id=:kb_id"
            params["kb_id"] = knowledge_base_id
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT id, file_sha256, version_no FROM documents "
                    f"WHERE {condition} ORDER BY version_no DESC LIMIT 1"
                ),
                params,
            ).mappings().first()
        return dict(row) if row else None

    def find_document_by_name(
        self, file_name: str, knowledge_base_id: int
    ) -> dict[str, Any] | None:
        """Find the newest active version for a force replacement."""
        from sqlalchemy import text

        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT id, file_sha256, version_no FROM documents "
                    "WHERE file_name=:file_name AND knowledge_base_id=:kb_id "
                    "AND status <> 'deleted' ORDER BY version_no DESC, updated_at DESC LIMIT 1"
                ),
                {"file_name": file_name, "kb_id": knowledge_base_id},
            ).mappings().first()
        return dict(row) if row else None

    def mark_document_deleted(self, document_id: int | str) -> None:
        """Hide an old document before deleting its vectors."""
        from sqlalchemy import text

        with self.engine.begin() as connection:
            connection.execute(
                text("UPDATE documents SET status='deleted' WHERE id=:id"),
                {"id": str(document_id)},
            )

    def search_chunks(
        self,
        keywords: list[str],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Search processed chunks through MySQL FULLTEXT."""
        if not self._fulltext_available or not keywords or top_k < 1:
            return []
        from sqlalchemy import text

        conditions = [
            "d.status = 'processed'",
            "MATCH(c.content) AGAINST (:query IN NATURAL LANGUAGE MODE)",
        ]
        params: dict[str, Any] = {"query": " ".join(keywords), "limit": top_k}
        if filters and filters.get("tenant_id") is not None:
            conditions.append("d.tenant_id = :tenant_id")
            params["tenant_id"] = int(filters["tenant_id"])
        if filters and filters.get("kb_id") is not None:
            conditions.append("d.knowledge_base_id = :kb_id")
            params["kb_id"] = int(filters["kb_id"])
        statement = text(
            "SELECT c.chunk_id, c.content AS text, "
            "COALESCE(c.source, d.file_name) AS source, c.summary, "
            "c.document_id AS doc_id, "
            "MATCH(c.content) AGAINST (:query IN NATURAL LANGUAGE MODE) AS score "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            f"WHERE {' AND '.join(conditions)} ORDER BY score DESC LIMIT :limit"
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement, params).mappings().all()
        return [dict(row) for row in rows]

    def save_document(self, document: ParsedDocument, document_id: int | str) -> None:
        """Insert or refresh a processed document metadata row."""
        from sqlalchemy import text

        statement = text(
            """INSERT INTO documents
            (id, knowledge_base_id, version_no, file_name, storage_path, mime_type,
             file_size, file_sha256, status, review_status, source_metadata,
             tenant_id, role_id)
            VALUES (:id, :kb_id, :version_no, :file_name, :storage_path, :mime_type,
                    :file_size, :sha, 'processed', 'pending', :metadata,
                    :tenant_id, :role_id)
            ON DUPLICATE KEY UPDATE status='processed', source_metadata=:metadata,
                                    updated_at=CURRENT_TIMESTAMP(6)"""
        )
        with self.engine.begin() as connection:
            version_row = connection.execute(
                text(
                    "SELECT COALESCE(MAX(version_no), 0) + 1 AS next_version "
                    "FROM documents WHERE knowledge_base_id=:kb_id AND file_sha256=:sha"
                ),
                {
                    "kb_id": int(document.metadata.get("knowledge_base_id", 1)),
                    "sha": document.file_sha256,
                },
            ).mappings().one()
            connection.execute(
                statement,
                {
                    "id": document_id,
                    "kb_id": int(document.metadata.get("knowledge_base_id", 1)),
                    "version_no": int(version_row["next_version"]),
                    "file_name": document.file_name,
                    "storage_path": document.source_path,
                    "mime_type": document.mime_type,
                    "file_size": os.path.getsize(document.source_path),
                    "sha": document.file_sha256,
                    "metadata": json.dumps(document.metadata, ensure_ascii=False),
                    "tenant_id": int(document.metadata.get("tenant_id", 1)),
                    "role_id": int(document.metadata.get("role_id", 0)),
                },
            )

    def save_chunks(
        self,
        document_id: int | str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Persist chunks, hashes, and parent context in one transaction."""
        from sqlalchemy import text

        statement = text(
            """INSERT INTO chunks
            (document_id, chunk_id, milvus_id, chunk_index, content, summary,
             source, page_start, page_end, text_hash, parent_id, parent_summary,
             embedding_status, metadata)
            VALUES (:document_id, :chunk_id, :milvus_id, :chunk_index, :content,
                    :summary, :source, :page_start, :page_end, :text_hash,
                    :parent_id, :parent_summary, 'completed', :metadata)
            ON DUPLICATE KEY UPDATE content=:content, summary=:summary,
              text_hash=:text_hash, parent_id=:parent_id, parent_summary=:parent_summary,
              embedding_status='completed', updated_at=CURRENT_TIMESTAMP(6)"""
        )
        with self.engine.begin() as connection:
            for chunk in chunks:
                connection.execute(
                    statement,
                    {
                        "document_id": document_id,
                        "chunk_id": chunk.chunk_id,
                        "milvus_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.text,
                        "summary": chunk.summary,
                        "source": chunk.source,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "text_hash": chunk.text_hash,
                        "parent_id": chunk.parent_id,
                        "parent_summary": chunk.parent_summary,
                        "metadata": json.dumps(chunk.metadata, ensure_ascii=False),
                    },
                )
