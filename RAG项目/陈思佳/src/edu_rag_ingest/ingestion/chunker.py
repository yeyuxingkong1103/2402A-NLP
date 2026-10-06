from __future__ import annotations

"""文档分块模块：按段落和标题切分文本，并为分块附加章节元数据。"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config.config import ChunkingConfig

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_STAGE_PATTERN = re.compile(r"第[一二三四]学段(?:（[^）]+）)?|第[一二三四]学段(?:\([^)]+\))?")


@dataclass(frozen=True)
class DocumentChunk:
    """一个可被向量化和检索的文档分块。"""
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    metadata: dict[str, str]


class TextChunker:
    """按照长度、标题和段落边界生成文档分块。"""
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def split(self, document_id: str, text: str, metadata: dict[str, str]) -> list[DocumentChunk]:
        """切分文档，并在相邻分块之间保留配置的 overlap。"""
        blocks = self._build_blocks(text, metadata)
        chunks: list[DocumentChunk] = []
        current = ""
        current_metadata = dict(metadata)

        for content, block_metadata in blocks:
            if not content.strip():
                continue
            if not current:
                current = content
                current_metadata = block_metadata
                continue

            if len(current) + len(content) + 2 <= self.config.chunk_size:
                current = f"{current}\n\n{content}".strip()
                current_metadata = self._merge_metadata(current_metadata, block_metadata)
                continue

            self._append_chunk(chunks, document_id, current, current_metadata)
            current = self._with_overlap(current, content)
            current_metadata = block_metadata

        self._append_chunk(chunks, document_id, current, current_metadata)
        return chunks

    def _build_blocks(self, text: str, base_metadata: dict[str, str]) -> list[tuple[str, dict[str, str]]]:
        blocks: list[tuple[str, dict[str, str]]] = []
        heading_stack: dict[int, str] = {}
        current_stage = ""
        current_paragraph: list[str] = []
        current_metadata = dict(base_metadata)

        def flush() -> None:
            nonlocal current_paragraph, current_metadata, heading_stack, current_stage
            if current_paragraph:
                blocks.append(("\n".join(current_paragraph).strip(), dict(current_metadata)))
                current_paragraph = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                flush()
                continue

            heading_match = _HEADING_PATTERN.match(line)
            if heading_match:
                flush()
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                heading_stack = {key: value for key, value in heading_stack.items() if key < level}
                heading_stack[level] = title
                stage_match = _STAGE_PATTERN.search(title)
                if stage_match:
                    current_stage = stage_match.group(0)
                current_metadata = self._metadata_from_headings(base_metadata, heading_stack, current_stage)
                current_paragraph.append(line)
                flush()
                continue

            stage_match = _STAGE_PATTERN.search(line)
            if stage_match:
                current_stage = stage_match.group(0)
                current_metadata = self._metadata_from_headings(base_metadata, heading_stack, current_stage)

            current_paragraph.append(line)

        flush()
        return blocks

    def _metadata_from_headings(
        self,
        base_metadata: dict[str, str],
        heading_stack: dict[int, str],
        stage: str,
    ) -> dict[str, str]:
        metadata = dict(base_metadata)
        ordered_titles = [heading_stack[key] for key in sorted(heading_stack)]
        if ordered_titles:
            metadata["section"] = " > ".join(ordered_titles)
            metadata["topic"] = ordered_titles[-1]
        if stage:
            metadata["stage"] = stage
        return metadata

    def _append_chunk(
        self,
        chunks: list[DocumentChunk],
        document_id: str,
        content: str,
        metadata: dict[str, str],
    ) -> None:
        content = content.strip()
        if len(content) < self.config.min_chunk_chars:
            return
        index = len(chunks)
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document_id}-{index:04d}",
                document_id=document_id,
                chunk_index=index,
                content=content,
                metadata=metadata,
            )
        )

    def _with_overlap(self, previous: str, content: str) -> str:
        if not previous or self.config.chunk_overlap <= 0:
            return content
        overlap = previous[-self.config.chunk_overlap :]
        return f"{overlap}\n\n{content}".strip()

    @staticmethod
    def _merge_metadata(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
        merged = dict(left)
        for key, value in right.items():
            if value:
                merged[key] = value
        return merged


def write_chunks_jsonl(chunks: list[DocumentChunk], output_path: Path) -> None:
    """将分块以一行一个 JSON 对象的格式写入文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
