import re
import logging
from dataclasses import dataclass
from uuid import uuid4

from app.rag.types import ChunkRecord, ParsedDocument

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkingConfig:
    """控制知识库分块大小、重叠长度和最小块长度。"""
    chunk_size: int = 600
    overlap: int = 80
    min_chunk_size: int = 40


def normalize_text(text: str) -> str:
    # 先清除不可见字符和多余空白，避免相同内容因格式差异生成不同向量。
    text = text.replace("\u0000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class DocumentChunker:
    """Structure-aware chunker with a fixed-size fallback for long paragraphs."""

    def __init__(self, config: ChunkingConfig):
        if config.overlap >= config.chunk_size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        self.config = config

    def split(
        self,
        document: ParsedDocument,
        document_id: str,
        source: str,
        base_metadata: dict | None = None,
    ) -> list[ChunkRecord]:
        # 分块顺序：清洗全文 -> 按结构分段 -> 对过长段落按句子切分。
        base_metadata = base_metadata or {}
        text = normalize_text(document.text)
        if not text:
            return []

        sections = self._split_sections(text)
        records: list[ChunkRecord] = []
        for section_index, section in enumerate(sections):
            for piece_index, piece in enumerate(self._split_long_section(section)):
                clean_piece = normalize_text(piece)
                if len(clean_piece) < self.config.min_chunk_size and records:
                    # 太短的尾部片段并入前一块，避免产生几乎没有检索价值的碎片。
                    records[-1].text = f"{records[-1].text}\n{clean_piece}".strip()
                    continue
                metadata = {
                    **base_metadata,
                    "section_index": section_index,
                    "piece_index": piece_index,
                    "chunking": "structure_aware_fixed_overlap",
                }
                records.append(
                    ChunkRecord(
                        id=uuid4().hex,
                        document_id=document_id,
                        text=clean_piece,
                        source=source,
                        metadata=metadata,
                    )
                )
        logger.info(
            "document chunked",
            extra={"document_id": document_id, "chunk_count": len(records)},
        )
        return records

    def _split_sections(self, text: str) -> list[str]:
        # 标题通常代表新主题；保留标题附近的内容可以提高召回后的上下文完整性。
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        sections: list[str] = []
        current = ""
        for paragraph in paragraphs:
            looks_like_heading = len(paragraph) <= 80 and (
                paragraph.startswith(("第", "一、", "二、", "三、", "1.", "2.", "3.", "#"))
                or paragraph.endswith((":", "："))
            )
            if looks_like_heading and current:
                sections.append(current)
                current = paragraph
            else:
                current = f"{current}\n\n{paragraph}".strip()
        if current:
            sections.append(current)
        return sections or [text]

    def _split_long_section(self, section: str) -> list[str]:
        if len(section) <= self.config.chunk_size:
            return [section]
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[。！？!?；;])\s*", section)
            if part.strip()
        ]
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) + 1 > self.config.chunk_size:
                # 超过目标长度时保存当前块，并保留末尾 overlap 个字符到下一块。
                pieces.append(current)
                current = current[-self.config.overlap :]
            current = f"{current}{sentence}".strip()
        if current:
            pieces.append(current)
        if not pieces:
            # 没有识别出句子边界时，退回纯字符窗口切分。
            step = self.config.chunk_size - self.config.overlap
            return [section[start : start + self.config.chunk_size] for start in range(0, len(section), step)]
        return pieces
