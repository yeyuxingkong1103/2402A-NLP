"""Text splitting strategies used by the public chunking pipeline."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence

from app.ingestion.chunk_models import ChunkingConfig, EmbeddingFunction, SENTENCE_RE, stable_hash
from app.ingestion.parser import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)
_HEADING_RE = re.compile(
    r"^(#{1,6}\s+.+|第[一二三四五六七八九十0-9]+[章节条].*|"
    r"[一二三四五六七八九十]+、.+|\d+(?:\.\d+)*\s+.+)$"
)
RawChunk = tuple[str, int, int]


def is_low_quality(text: str, min_chars: int) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < min_chars:
        return True
    return len(set(compact)) / len(compact) < 0.08


def window_text(text: str, size: int, overlap: int) -> list[str]:
    """Split oversized units while enforcing a real maximum chunk size."""
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if len(text) <= size:
        return [text]
    windows: list[str] = []
    start = 0
    while start < len(text):
        window = text[start : start + size].strip()
        if window:
            windows.append(window)
        if start + size >= len(text):
            break
        start += size - overlap
    return windows


def pack_units(
    units: Iterable[str],
    size: int,
    overlap_units: int = 0,
    *,
    char_overlap: int = 0,
    deduplicate: bool = False,
) -> list[str]:
    """Pack semantic units, deduplicating before rather than after concatenation."""
    packed: list[str] = []
    current: list[str] = []
    current_len = 0
    seen: set[str] = set()

    def flush(*, keep_overlap: bool) -> None:
        nonlocal current, current_len
        if current:
            packed.append("\n".join(current).strip())
        current = current[-overlap_units:] if keep_overlap and overlap_units else []
        current_len = len("\n".join(current))

    for unit in units:
        if deduplicate:
            key = stable_hash(unit)
            if key in seen:
                continue
            seen.add(key)
        if len(unit) > size:
            flush(keep_overlap=False)
            packed.extend(window_text(unit, size, min(char_overlap, size - 1)))
            continue
        separator = 1 if current else 0
        if current and current_len + separator + len(unit) > size:
            flush(keep_overlap=True)
            separator = 1 if current else 0
        current.append(unit)
        current_len += len(unit) + separator
    if current:
        packed.append("\n".join(current).strip())
    return packed


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; single newlines often represent soft wrapping."""
    return [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]


def _fixed_chunks(pages: Sequence[ParsedPage], config: ChunkingConfig) -> list[RawChunk]:
    return [
        (text, page.page_number, page.page_number)
        for page in pages
        for text in window_text(page.text, config.chunk_size, config.chunk_overlap)
    ]


def _sentence_chunks(pages: Sequence[ParsedPage], config: ChunkingConfig) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    for page in pages:
        sentences = [part.strip() for part in SENTENCE_RE.split(page.text) if part.strip()]
        for text in pack_units(
            sentences,
            config.chunk_size,
            1 if config.chunk_overlap > 0 else 0,
            char_overlap=config.chunk_overlap,
            deduplicate=config.deduplicate,
        ):
            chunks.append((text, page.page_number, page.page_number))
    return chunks


def _paragraph_chunks(pages: Sequence[ParsedPage], config: ChunkingConfig) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    for page in pages:
        for text in pack_units(
            split_paragraphs(page.text),
            config.chunk_size,
            char_overlap=config.chunk_overlap,
            deduplicate=config.deduplicate,
        ):
            chunks.append((text, page.page_number, page.page_number))
    return chunks


def _heading_chunks(pages: Sequence[ParsedPage], config: ChunkingConfig) -> list[RawChunk]:
    sections: list[tuple[list[str], int, int]] = []
    current: list[str] = []
    page_start: int | None = None
    page_end: int | None = None
    for page in pages:
        for raw_line in page.text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if current and _HEADING_RE.match(line):
                sections.append((current, page_start or page.page_number, page_end or page.page_number))
                current = []
                page_start = page.page_number
            page_start = page_start or page.page_number
            page_end = page.page_number
            current.append(line)
    if current:
        sections.append((current, page_start or 1, page_end or page_start or 1))

    chunks: list[RawChunk] = []
    for lines, start, end in sections:
        for text in pack_units(
            lines,
            config.chunk_size,
            char_overlap=config.chunk_overlap,
            deduplicate=config.deduplicate,
        ):
            chunks.append((text, start, end))
    return chunks


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _semantic_chunks(
    pages: Sequence[ParsedPage],
    config: ChunkingConfig,
    embedding_function: EmbeddingFunction | None,
) -> list[RawChunk]:
    units = [
        (paragraph, page.page_number)
        for page in pages
        for paragraph in split_paragraphs(page.text)
    ]
    if not units:
        return []
    if embedding_function is None:
        logger.warning("semantic chunking requested without embeddings; using paragraphs")
        return _paragraph_chunks(pages, config)
    embeddings = embedding_function([text for text, _ in units])
    chunks: list[RawChunk] = []
    texts = [units[0][0]]
    page_start = page_end = units[0][1]
    for index in range(1, len(units)):
        next_text, next_page = units[index]
        should_split = (
            _cosine(embeddings[index - 1], embeddings[index])
            < config.semantic_similarity_threshold
            or sum(len(item) for item in texts) + len(next_text) > config.chunk_size
        )
        if should_split:
            chunks.append(("\n".join(texts), page_start, page_end))
            texts = [next_text]
            page_start = next_page
        else:
            texts.append(next_text)
        page_end = next_page
    chunks.append(("\n".join(texts), page_start, page_end))
    return chunks


def raw_chunks(
    document: ParsedDocument,
    config: ChunkingConfig,
    embedding_function: EmbeddingFunction | None,
) -> list[RawChunk]:
    """Dispatch to the configured splitting strategy."""
    strategies = {
        "fixed": lambda: _fixed_chunks(document.pages, config),
        "sentence": lambda: _sentence_chunks(document.pages, config),
        "paragraph": lambda: _paragraph_chunks(document.pages, config),
        "heading": lambda: _heading_chunks(document.pages, config),
        "semantic": lambda: _semantic_chunks(document.pages, config, embedding_function),
    }
    try:
        return strategies[config.strategy]()
    except KeyError as exc:
        raise ValueError(f"unsupported chunking strategy: {config.strategy}") from exc
