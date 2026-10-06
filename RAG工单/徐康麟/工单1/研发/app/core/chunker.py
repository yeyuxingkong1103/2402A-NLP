"""分块：把解析后的页面与表格切成检索单元。

工单要求（5.2）：
- 按标题、段落、表格分块；
- chunk_size 500~800，overlap 100；
- 每个 chunk 必须带元数据 chunk_id / page / section / type / content；
- 保存到 SQLite 和向量库。

实现要点：
- **按标题切段**：先用标题把页面切成小节，再在小节内按长度滑窗，
  避免把不同章节的内容混进同一个 chunk（对招股书这种强结构化文档很关键）。
- **表格整表成块**：表格被切开会破坏行列语义，因此整表作为一个 chunk。
- **关键词标注**：命中领域关键词的 chunk 会在 ``keywords`` 中记录，供检索加权。
"""

from __future__ import annotations

import re
from pathlib import Path

from app.core.config import get_settings
from app.core.logging_conf import logger, trace
from app.core.text_utils import dedupe_keep_order, looks_like_heading, normalize_text, stable_id
from app.models.schemas import Chunk, ParsedDocument

# 领域关键词：用于给 chunk 打标签并支撑加权检索
DOMAIN_KEYWORDS: tuple[str, ...] = (
    "军用领域",
    "主营业务收入",
    "占比",
    "比重",
    "注册资本",
    "法定代表人",
    "募集资金",
    "补充流动资金",
    "上游",
    "下游",
    "供应商",
    "客户",
    "技术标准",
    "技术规范",
    "科技进步奖",
    "C4ISR",
    "视频指挥",
)


class Chunker:
    """文本与表格分块器。"""

    def __init__(self, settings=None) -> None:
        self.settings = settings or get_settings()
        self.chunk_size = self.settings.chunk.chunk_size
        self.chunk_overlap = self.settings.chunk.chunk_overlap
        self.min_chars = self.settings.chunk.min_chunk_chars

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _tag_keywords(text: str) -> list[str]:
        """标注 chunk 命中的领域关键词。"""
        return dedupe_keep_order([kw for kw in DOMAIN_KEYWORDS if kw in text])

    def _split_by_heading(self, text: str, base_section: str) -> list[tuple[str, str]]:
        """按标题把页面文本切成 (section, body) 列表。"""
        blocks: list[tuple[str, list[str]]] = []
        current_section = base_section
        buffer: list[str] = []

        for line in text.split("\n"):
            if looks_like_heading(line):
                if buffer:
                    blocks.append((current_section, buffer))
                    buffer = []
                current_section = line.strip()
                buffer.append(line)
            else:
                buffer.append(line)
        if buffer:
            blocks.append((current_section, buffer))
        return [(section, normalize_text("\n".join(lines))) for section, lines in blocks if normalize_text("\n".join(lines))]

    def _sliding_window(self, text: str) -> list[str]:
        """按 chunk_size / overlap 做滑窗切分；优先在句末断开。"""
        text = normalize_text(text)
        if len(text) <= self.chunk_size:
            return [text] if text else []

        pieces: list[str] = []
        start = 0
        length = len(text)
        sentence_end = re.compile(r"[。！？；\n]")
        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                # 在窗口后 30% 范围内找最近的句末，避免句子被拦腰截断
                search_from = start + int(self.chunk_size * 0.7)
                window = text[search_from:end]
                matches = list(sentence_end.finditer(window))
                if matches:
                    end = search_from + matches[-1].end()
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= length:
                break
            start = max(end - self.chunk_overlap, start + 1)
        return pieces

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    @trace
    def split(self, document: ParsedDocument) -> list[Chunk]:
        """把解析结果切成 chunk 列表（文本 + 表格）。"""
        chunks: list[Chunk] = []
        table_by_page: dict[int, list] = {}
        for table in document.tables:
            table_by_page.setdefault(table.page, []).append(table)

        counter = 0
        current_section = ""

        for page in document.pages:
            if not page.text.strip():
                continue
            for section, body in self._split_by_heading(page.text, current_section):
                if section:
                    current_section = section
                for piece in self._sliding_window(body):
                    if len(piece) < self.min_chars:
                        continue
                    counter += 1
                    chunks.append(
                        Chunk(
                            chunk_id=f"c{counter:06d}",
                            doc_id=document.doc_id,
                            page=page.page,
                            section=section or current_section,
                            type="text",
                            content=piece,
                            char_count=len(piece),
                            keywords=self._tag_keywords(piece),
                        )
                    )

            # 表格：整表成块，并额外生成一行“检索摘要”，提升表格召回
            for table in table_by_page.get(page.page, []):
                counter += 1
                content = f"[表格 {table.table_id}] 第 {table.page} 页\n{table.markdown}"
                chunks.append(
                    Chunk(
                        chunk_id=f"c{counter:06d}",
                        doc_id=document.doc_id,
                        page=table.page,
                        section=table.section or current_section,
                        type="table",
                        content=content,
                        char_count=len(content),
                        table_id=table.table_id,
                        keywords=self._tag_keywords(content),
                    )
                )

        stats = {
            "total": len(chunks),
            "text": sum(1 for c in chunks if c.type == "text"),
            "table": sum(1 for c in chunks if c.type == "table"),
            "avg_chars": round(sum(c.char_count for c in chunks) / len(chunks), 1) if chunks else 0,
        }
        logger.info("app.core.chunker", "分块完成", doc_id=document.doc_id, **stats)
        return chunks

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------
    @trace
    def save(self, chunks: list[Chunk], output_dir: Path | str | None = None, name: str = "chunks") -> Path:
        """把 chunk 列表写入 JSONL，便于人工核对与离线测试。"""
        settings = self.settings
        target_dir = Path(output_dir) if output_dir else settings.paths.data_processed
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{name}.jsonl"
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json() + "\n")
        logger.info("app.core.chunker", "分块结果已保存", path=str(target), count=len(chunks))
        return target


def get_chunker() -> Chunker:
    """工厂函数。"""
    return Chunker()
