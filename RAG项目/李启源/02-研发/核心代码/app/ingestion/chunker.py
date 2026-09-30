"""Public document chunking facade.

Splitting strategies and data contracts live in focused modules; this module
keeps the historical import path and owns only result assembly.
"""

from __future__ import annotations

import logging
from typing import Sequence

from app.ingestion.chunk_models import (
    Chunk,
    ChunkingConfig,
    ChunkingResult,
    EmbeddingFunction,
    ParentBlock,
    Summarizer,
    default_summarize,
    stable_hash,
)
from app.ingestion.chunk_strategies import is_low_quality, raw_chunks
from app.ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)


def _build_parent_blocks(
    chunks: Sequence[Chunk],
    *,
    config: ChunkingConfig,
    summarizer: Summarizer,
) -> tuple[ParentBlock, ...]:
    """Group adjacent child chunks into bounded context blocks."""
    parents: list[ParentBlock] = []
    current: list[Chunk] = []
    current_length = 0
    for chunk in chunks:
        if current and current_length + len(chunk.text) > config.parent_chunk_size:
            parents.append(_make_parent(current, config, summarizer, len(parents)))
            current = []
            current_length = 0
        current.append(chunk)
        current_length += len(chunk.text)
    if current:
        parents.append(_make_parent(current, config, summarizer, len(parents)))
    return tuple(parents)


def _make_parent(
    chunks: Sequence[Chunk],
    config: ChunkingConfig,
    summarizer: Summarizer,
    index: int,
) -> ParentBlock:
    """Create one stable parent record from adjacent child chunks."""
    text = "\n\n".join(chunk.text for chunk in chunks)
    return ParentBlock(
        parent_id=f"parent-{index:04d}-{stable_hash(text)[:16]}",
        text=text,
        summary=summarizer(text, config.summary_max_chars),
        child_chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
        page_start=min((chunk.page_start or 0 for chunk in chunks), default=None),
        page_end=max((chunk.page_end or 0 for chunk in chunks), default=None),
        metadata={"child_count": len(chunks)},
    )


def _assemble_chunks(
    document: ParsedDocument,
    raw: Sequence[tuple[str, int, int]],
    config: ChunkingConfig,
    document_id: str | int | None,
    summarizer: Summarizer,
) -> list[Chunk]:
    """Normalize raw strategy output and remove duplicate or unusable chunks."""
    seen_hashes: set[str] = set()
    chunks: list[Chunk] = []
    doc_key = str(document_id or document.file_sha256[:12])
    page_metadata = {page.page_number: page.metadata for page in document.pages}
    for raw_index, (text, page_start, page_end) in enumerate(raw):
        text = text.strip()
        if is_low_quality(text, config.min_chunk_chars):
            continue
        text_hash = stable_hash(text)
        if config.deduplicate and text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)
        chunks.append(
            Chunk(
                chunk_id=f"doc{doc_key}-chunk{len(chunks):06d}-{text_hash[:8]}",
                text=text,
                summary=summarizer(text, config.summary_max_chars),
                source=(
                    f"{document.file_name}:p{page_start}"
                    if page_start == page_end
                    else f"{document.file_name}:p{page_start}-{page_end}"
                ),
                page_start=page_start,
                page_end=page_end,
                chunk_index=len(chunks),
                text_hash=text_hash,
                metadata={
                    "raw_index": raw_index,
                    "strategy": config.strategy,
                    "file_name": document.file_name,
                    "file_sha256": document.file_sha256,
                    **page_metadata.get(page_start, {}),
                },
            )
        )
    return chunks


def _attach_parents(
    chunks: Sequence[Chunk], parents: Sequence[ParentBlock]
) -> tuple[Chunk, ...]:
    """Return immutable child records carrying their parent references."""
    parent_by_child = {
        child_id: parent for parent in parents for child_id in parent.child_chunk_ids
    }
    return tuple(
        Chunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            summary=chunk.summary,
            source=chunk.source,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chunk_index=chunk.chunk_index,
            text_hash=chunk.text_hash,
            parent_id=(
                parent_by_child[chunk.chunk_id].parent_id
                if chunk.chunk_id in parent_by_child
                else None
            ),
            parent_summary=(
                parent_by_child[chunk.chunk_id].summary
                if chunk.chunk_id in parent_by_child
                else None
            ),
            metadata=chunk.metadata,
        )
        for chunk in chunks
    )


def chunk_document(
    document: ParsedDocument,
    *,
    config: ChunkingConfig | None = None,
    document_id: str | int | None = None,
    summarizer: Summarizer = default_summarize,
    embedding_function: EmbeddingFunction | None = None,
) -> ChunkingResult:
    """Split a parsed document into deduplicated child and parent chunks."""
    active_config = config or ChunkingConfig()
    chunks = _assemble_chunks(
        document,
        raw_chunks(document, active_config, embedding_function),
        active_config,
        document_id,
        summarizer,
    )
    parents = (
        _build_parent_blocks(chunks, config=active_config, summarizer=summarizer)
        if active_config.create_parent_chunks
        else tuple()
    )
    attached = _attach_parents(chunks, parents)
    logger.info(
        "document chunked",
        extra={
            "file_name": document.file_name,
            "strategy": active_config.strategy,
            "chunk_count": len(attached),
            "parent_count": len(parents),
        },
    )
    return ChunkingResult(chunks=attached, parents=parents)


__all__ = [
    "Chunk",
    "ChunkingConfig",
    "ChunkingResult",
    "ParentBlock",
    "chunk_document",
]
