"""结构感知分块。

策略（按优先级）：

1. 先按 Markdown 标题（#/##/###）切成小节，保留「标题路径」作为 section；
2. 小节内按空行切段落，表格行（以 ``|`` 开头）视为整体，不可拆散；
3. 段落累积到 ``chunk_size`` 停止，相邻块之间用 ``chunk_overlap`` 个字符的尾部重叠；
4. 单段超过 ``max_chunk_chars`` 时按句子边界二次切分；
5. 小于 ``min_chunk_chars`` 的尾块尝试并入上一块，否则丢弃。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from ..logging_conf import get_logger
from .loaders import LoadedDoc

logger = get_logger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])|(?<=\.)\s+")


@dataclass(slots=True)
class Chunk:
    """一个待入库的知识块。"""

    text: str
    section: str
    chunk_index: int
    doc_id: str
    doc_title: str
    source: str
    role: str
    scope: str
    tags: list[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}#{self.chunk_index}"

    @property
    def char_len(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class Section:
    path: str
    lines: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n".join(self.lines).strip()


def _iter_sections(text: str) -> Iterator[Section]:
    """按标题切小节，维护标题路径（H1 › H2 › H3）。"""

    stack: list[tuple[int, str]] = []
    current = Section(path="")
    for line in text.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            if current.body or current.path:
                yield current
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current = Section(path=" › ".join(item[1] for item in stack))
            continue
        current.lines.append(line)
    yield current


def _iter_blocks(body: str) -> Iterator[str]:
    """把小节内容切成段落块；连续的表格行保持为一块。"""

    buffer: list[str] = []
    table: list[str] = []

    def flush_buffer() -> Iterator[str]:
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                yield text
            buffer.clear()

    def flush_table() -> Iterator[str]:
        if table:
            text = "\n".join(table).strip()
            if text:
                yield text
            table.clear()

    for raw_line in body.split("\n"):
        line = raw_line.rstrip()
        if line.strip().startswith("|"):
            yield from flush_buffer()
            table.append(line.strip())
            continue
        yield from flush_table()
        if not line.strip():
            yield from flush_buffer()
            continue
        buffer.append(line)
    yield from flush_buffer()
    yield from flush_table()


def _split_long_text(text: str, max_chars: int) -> list[str]:
    pieces = [piece for piece in _SENTENCE_SPLIT_RE.split(text) if piece and piece.strip()]
    if len(pieces) <= 1:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    result: list[str] = []
    buffer = ""
    for piece in pieces:
        if len(buffer) + len(piece) <= max_chars:
            buffer += piece
        else:
            if buffer.strip():
                result.append(buffer.strip())
            buffer = piece if len(piece) <= max_chars else ""
            if not buffer:
                result.extend(piece[i : i + max_chars] for i in range(0, len(piece), max_chars))
    if buffer.strip():
        result.append(buffer.strip())
    return result


def _tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    for mark in ("。", "！", "？", "；", "\n", "，"):
        index = tail.find(mark)
        if index != -1:
            return tail[index + 1:].strip()
    return tail.strip()


def chunk_document(
    doc: LoadedDoc,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    min_chunk_chars: int = 80,
    max_chunk_chars: int = 1400,
) -> list[Chunk]:
    """把一篇文档切成知识块。"""

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")

    chunks: list[Chunk] = []
    index = 0
    for section in _iter_sections(doc.text):
        section_path = section.path or doc.title
        buffer = ""
        for block in _iter_blocks(section.body):
            sub_blocks = (
                _split_long_text(block, max_chunk_chars) if len(block) > max_chunk_chars else [block]
            )
            for item in sub_blocks:
                candidate = f"{buffer}\n\n{item}".strip() if buffer else item
                if len(candidate) <= chunk_size or not buffer:
                    buffer = candidate
                    continue
                chunks.append(
                    Chunk(
                        text=buffer,
                        section=section_path,
                        chunk_index=index,
                        doc_id=doc.doc_id,
                        doc_title=doc.title,
                        source=doc.source,
                        role=doc.role,
                        scope=doc.scope,
                        tags=list(doc.tags),
                    )
                )
                index += 1
                overlap = _tail_overlap(buffer, chunk_overlap)
                buffer = f"{overlap}\n\n{item}".strip() if overlap else item
        if buffer.strip():
            if (
                chunks
                and len(buffer) < min_chunk_chars
                and chunks[-1].doc_id == doc.doc_id
                and chunks[-1].section == section_path
            ):
                merged = f"{chunks[-1].text}\n\n{buffer}".strip()
                if len(merged) <= max_chunk_chars:
                    chunks[-1] = Chunk(
                        text=merged,
                        section=chunks[-1].section,
                        chunk_index=chunks[-1].chunk_index,
                        doc_id=doc.doc_id,
                        doc_title=doc.title,
                        source=doc.source,
                        role=doc.role,
                        scope=doc.scope,
                        tags=list(doc.tags),
                    )
                    continue
            if len(buffer) >= min_chunk_chars or not chunks:
                chunks.append(
                    Chunk(
                        text=buffer,
                        section=section_path,
                        chunk_index=index,
                        doc_id=doc.doc_id,
                        doc_title=doc.title,
                        source=doc.source,
                        role=doc.role,
                        scope=doc.scope,
                        tags=list(doc.tags),
                    )
                )
                index += 1

    # 重新编号，保证连续性
    for position, chunk in enumerate(chunks):
        chunk.chunk_index = position
    logger.debug("分块完成：%s → %d 块", doc.title, len(chunks))
    return chunks


def chunk_summary(chunks: Sequence[Chunk]) -> dict[str, object]:
    lengths = [chunk.char_len for chunk in chunks] or [0]
    return {
        "chunks": len(chunks),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
        "max_chars": max(lengths),
        "min_chars": min(lengths),
    }
