"""Offline document-to-vector indexing orchestration.

The CLI imports this class through the historical ``build_knowledge_base``
module. Keeping the workflow here makes it reusable from tests and prevents
command-line argument handling from being coupled to ingestion behavior.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from app.ingestion.chunker import ChunkingConfig, chunk_document
from app.ingestion.embedding import build_embedding_client_from_env
from app.ingestion.indexer import (
    IngestionIndexer,
    SqlAlchemyMetadataStore,
    build_milvus_store_from_env,
)
from app.ingestion.parser import parse_document

logger = logging.getLogger(__name__)


class OfflineKnowledgeBaseBuilder:
    """Build one document or a directory into the configured stores."""

    def __init__(
        self,
        *,
        embedding_client: Any | None = None,
        vector_store: Any | None = None,
        metadata_store: Any | None = None,
    ) -> None:
        self.embedding_client = embedding_client or build_embedding_client_from_env()
        self.vector_store = vector_store or build_milvus_store_from_env(
            self.embedding_client.dimension
        )
        if metadata_store is None and os.getenv("DATABASE_URL"):
            metadata_store = SqlAlchemyMetadataStore(os.getenv("DATABASE_URL"))
        self.metadata_store = metadata_store
        self.indexer = IngestionIndexer(
            embedding_client=self.embedding_client,
            vector_store=self.vector_store,
            metadata_store=self.metadata_store,
        )

    def build_document(
        self,
        file_path: str | Path,
        *,
        knowledge_base_id: int = 1,
        tenant_id: int = 1,
        role_id: int = 0,
        parser: str = "auto",
        chunking_strategy: str = "paragraph",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        force: bool = False,
    ) -> dict[str, Any]:
        """Parse, chunk, embed, and index one document."""
        started = time.time()
        path = Path(file_path)
        logger.info("Processing document: %s", path.name)

        parse_started = time.time()
        try:
            document = parse_document(path, parser=parser)
            parse_duration = time.time() - parse_started
        except Exception as exc:
            logger.error("Parse failed: %s", exc)
            return self._failure("parse", exc, started)

        if not force and self.metadata_store:
            existing = self.metadata_store.find_document_by_hash(
                document.file_sha256, knowledge_base_id
            )
            if existing:
                return {
                    "status": "skipped",
                    "reason": "duplicate",
                    "file_sha256": document.file_sha256,
                    "existing_id": existing.get("id"),
                    "duration": time.time() - started,
                }

        document_id = str(uuid.uuid4())
        chunk_started = time.time()
        try:
            config = ChunkingConfig(
                strategy=chunking_strategy,  # type: ignore[arg-type]
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunked = chunk_document(document, config=config, document_id=document_id)
            chunk_duration = time.time() - chunk_started
        except Exception as exc:
            logger.error("Chunking failed: %s", exc)
            return self._failure("chunk", exc, started)

        embed_started = time.time()
        try:
            vectors = self.embedding_client.embed_texts(
                [chunk.text for chunk in chunked.chunks]
            )
            embed_duration = time.time() - embed_started
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            return self._failure("embed", exc, started)

        index_started = time.time()
        try:
            indexed = self.indexer.index(
                document,
                chunked.chunks,
                document_id=document_id,
                tenant_id=tenant_id,
                role_id=role_id,
                knowledge_base_id=knowledge_base_id,
                force=force,
                vectors=vectors,
            )
            index_duration = time.time() - index_started
        except Exception as exc:
            logger.error("Indexing failed: %s", exc)
            return self._failure("index", exc, started)

        duration = time.time() - started
        logger.info("Completed %s: %s vectors in %.2fs", path.name, indexed.inserted, duration)
        return {
            "status": "success",
            "document_id": document_id,
            "file_name": document.file_name,
            "file_sha256": document.file_sha256,
            "pages": len(document.pages),
            "characters": document.character_count,
            "chunks": len(chunked.chunks),
            "parents": len(chunked.parents),
            "vectors_inserted": indexed.inserted,
            "duration": duration,
            "timings": {
                "parse": parse_duration,
                "chunk": chunk_duration,
                "embed": embed_duration,
                "index": index_duration,
            },
        }

    def build_directory(
        self,
        directory: str | Path,
        *,
        knowledge_base_id: int = 1,
        tenant_id: int = 1,
        role_id: int = 0,
        parser: str = "auto",
        chunking_strategy: str = "paragraph",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        force: bool = False,
        patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build every supported document below ``directory``."""
        root = Path(directory)
        if not root.is_dir():
            raise ValueError(f"Not a directory: {root}")
        patterns = patterns or ["*.pdf", "*.txt", "*.md", "*.markdown", "*.jsonl"]
        files = [path for pattern in patterns for path in root.rglob(pattern)]
        if not files:
            return {"status": "no_files", "directory": str(root)}

        results = [
            self.build_document(
                path,
                knowledge_base_id=knowledge_base_id,
                tenant_id=tenant_id,
                role_id=role_id,
                parser=parser,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                force=force,
            )
            for path in files
        ]
        return {
            "status": "completed",
            "total": len(results),
            "success": sum(item["status"] == "success" for item in results),
            "skipped": sum(item["status"] == "skipped" for item in results),
            "failed": sum(item["status"] == "failed" for item in results),
            "results": results,
        }

    @staticmethod
    def _failure(stage: str, error: Exception, started: float) -> dict[str, Any]:
        return {
            "status": "failed",
            "stage": stage,
            "error": str(error),
            "duration": time.time() - started,
        }
