"""Contracts shared by ingestion storage adapters and the indexer."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from app.ingestion.chunk_models import Chunk
from app.ingestion.parser import ParsedDocument


class IndexingError(RuntimeError):
    """Raised when vector or metadata persistence fails."""


@dataclass(slots=True, frozen=True)
class IndexingResult:
    """Summary of one idempotent indexing operation."""

    document_id: int | str
    file_sha256: str
    inserted: int
    skipped: int
    deleted: int
    collection: str


class ExistingDocumentStore(Protocol):
    """Metadata operations required for incremental updates."""

    def find_document_by_hash(
        self, file_sha256: str, knowledge_base_id: int | None = None
    ) -> dict[str, Any] | None: ...

    def find_document_by_name(
        self, file_name: str, knowledge_base_id: int
    ) -> dict[str, Any] | None: ...

    def mark_document_deleted(self, document_id: int | str) -> None: ...

    def save_document(self, document: ParsedDocument, document_id: int | str) -> None: ...

    def save_chunks(
        self,
        document_id: int | str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...


class VectorStore(Protocol):
    """Minimal vector store boundary used by the pipeline."""

    def upsert(self, records: list[dict[str, Any]]) -> None: ...

    def delete_by_document(self, document_id: int | str) -> int: ...


def find_document_by_hash(
    metadata_store: Any,
    file_sha256: str,
    knowledge_base_id: int | None,
) -> dict[str, Any] | None:
    """Support both the old one-argument and new two-argument fake stores."""
    finder = metadata_store.find_document_by_hash
    try:
        parameters = inspect.signature(finder).parameters.values()
    except (TypeError, ValueError):
        return finder(file_sha256, knowledge_base_id)
    accepts_keyword = any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters)
    positional_count = sum(
        parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        for parameter in parameters
    )
    if accepts_keyword or positional_count >= 2:
        return finder(file_sha256, knowledge_base_id)
    return finder(file_sha256)


def find_document_by_name(
    metadata_store: Any,
    file_name: str,
    knowledge_base_id: int,
) -> dict[str, Any] | None:
    """Find an active same-name document when the store supports that lookup."""
    finder = getattr(metadata_store, "find_document_by_name", None)
    return finder(file_name, knowledge_base_id) if finder else None
