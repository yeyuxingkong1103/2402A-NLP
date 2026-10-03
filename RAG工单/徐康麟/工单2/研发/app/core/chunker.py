"""分块：标题层级切分 + 段落滑窗（500/80）+ 表格整表成块 + 关键词标签 + 释义页标记。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 分块优化（对应 设计/接口设计.md §2.4、优化方案设计.md §2.2 M2.1~M2.5）

设计要点（依据设计取证 A.3）：
- 基线块长度中位数仅 118 字符，滑窗参数几乎不触发，因此本工单分块优化的价值在
  **切分边界与标签**，而不是"把大块切小"；
- 表格块**整表成块不拆分**，块首加 `[表格 id] 第 N 页` 与"检索摘要行"，保证表格数值可被检索；
- 每块带 `chunk_id/page/section/type/keywords/is_boilerplate` 元数据；
- 释义页（前 30 页的释义/基本用语样板）显式标记 `is_boilerplate`，供检索层审计式降权。
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.errors import RAGError, StorageError
from app.core.logging_conf import logger, trace
from app.core.retrieval_utils import AMOUNT_PATTERN, PERCENT_PATTERN, COMPANY_NAME
from app.core.text_utils import dedupe_keep_order, normalize_text
from app.models.schemas import Chunk, ParsedDocument

#: 领域关键词标签表（工单 6.2 点名词全覆盖）
KEYWORD_LABELS: tuple[str, ...] = (
    "收入",
    "主营业务收入",
    "占比",
    "比重",
    "注册资本",
    "法定代表人",
    "募集资金",
    "募资",
    "补充流动资金",
    "上下游",
    "上游",
    "下游",
    "供应商",
    "客户",
    "技术标准",
    "科技进步奖",
    "军用领域",
    "报告期内",
    "表格",
)

#: 句末标点（滑窗断开点）
SENTENCE_END = re.compile(r"(?<=[。！？；!?;])")
#: 标题行判定（行首）
HEADING_PATTERN = re.compile(
    r"^(?:第[一二三四五六七八九十百]+[章节篇]|[一二三四五六七八九十]+、|[（(][一二三四五六七八九十]+[)）]|\d+(?:\.\d+)*[\s、.]|[（(]\d+[)）])"
)
#: 标题行判定（换行之后，用于段落切分的 lookahead，不能用 ^，且必须全部非捕获组）
HEADING_INLINE = re.compile(
    r"(?=第[一二三四五六七八九十百]+[章节篇]|[一二三四五六七八九十]+、|[（(][一二三四五六七八九十]+[)）]|\d+(?:\.\d+)*[\s、.])"
)


class Chunker:
    """分块器。"""

    def __init__(self, settings=None) -> None:
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    @trace
    def split(self, document: ParsedDocument) -> list[Chunk]:
        """把解析结果切成检索块（表格整表成块；段落按 500/80 滑窗）。"""
        try:
            if not document or not document.pages:
                raise RAGError("解析结果为空，无法分块", detail={"doc_id": getattr(document, "doc_id", "")})
            settings = self._settings.chunk
            chunks: list[Chunk] = []
            counter = 0
            # 按页组织：正文块来自 blocks 的 text 项，表格块来自 tables（未被拆分的整表）
            text_blocks_by_page: dict[int, list[str]] = {}
            for block in document.blocks:
                if block.type != "text":
                    continue
                text_blocks_by_page.setdefault(block.page, []).append(block.content)
            if not text_blocks_by_page:  # 兜底：blocks 为空时用 pages
                for page in document.pages:
                    if page.text:
                        text_blocks_by_page.setdefault(page.page, []).append(page.text)

            tables_by_page: dict[int, list[Any]] = {}
            for table in document.tables:
                tables_by_page.setdefault(table.page, []).append(table)

            for page_no in sorted(set(text_blocks_by_page) | set(tables_by_page)):
                section = self._section_of(document, page_no)
                # 同一页的正文块拼成一整段后再切分：便于把相邻短段打包到接近 chunk_size
                page_text = "\n".join(text_blocks_by_page.get(page_no, []))
                for piece in self._split_paragraphs(page_text, section):
                    counter += 1
                    chunks.append(self._make_chunk(counter, document.doc_id, page_no, section, "text", piece))
                for table in tables_by_page.get(page_no, []):
                    counter += 1
                    content = self._table_block_content(table)
                    chunks.append(
                        self._make_chunk(
                            counter,
                            document.doc_id,
                            table.page,
                            table.section or section,
                            "table",
                            content,
                            table_id=table.table_id,
                        )
                    )
            chunks = self._absorb_short_chunks(chunks)
            chunks = [self._make_chunk(index + 1, c.doc_id, c.page, c.section, c.type, c.content, c.table_id)
                      for index, c in enumerate(chunks)]
            stats = self.stats(chunks)
            logger.info("app.core.chunker", "分块完成", doc_id=document.doc_id, **stats)
            return chunks
        except RAGError:
            logger.exception("app.core.chunker", "分块失败（空文档）")
            raise
        except Exception as exc:
            logger.exception("app.core.chunker", "分块异常")
            raise RAGError(f"分块失败: {exc}") from exc

    # ------------------------------------------------------------------
    @staticmethod
    def _section_of(document: ParsedDocument, page: int) -> str:
        """取该页第一个结构化条目的 section（标题路径）。"""
        for block in document.blocks:
            if block.page == page and block.section:
                return block.section
        return ""

    def _split_paragraphs(self, text: str, section: str) -> list[str]:
        """段落切分 + 滑窗（优先句末断开，目标 chunk_size，overlap）。

        健壮性：``re.split`` 的结果可能包含 ``None``（正则含捕获组或输入异常时），
        因此入口统一过滤非字符串与空白项，避免 ``AttributeError`` 中断整篇分块。
        """
        settings = self._settings.chunk
        cleaned = normalize_text(text or "")
        if not cleaned:
            return []
        paragraphs = [
            item.strip()
            for item in re.split(r"\n{2,}|\n" + HEADING_INLINE.pattern, cleaned)
            if isinstance(item, str) and item.strip()
        ]
        if not paragraphs:
            paragraphs = [cleaned]
        pieces: list[str] = []
        buffer = ""

        def flush() -> None:
            """把缓冲区落块（闭包内修改 buffer 需用 nonlocal）。"""
            nonlocal buffer
            if buffer.strip():
                pieces.append(buffer.strip())
            buffer = ""

        for paragraph in paragraphs:
            if len(paragraph) <= settings.chunk_size:
                # 打包多段：把同页相邻短段聚合到接近 chunk_size，提高单块信息密度
                if buffer and len(buffer) + len(paragraph) + 1 > settings.chunk_size:
                    flush()
                buffer = f"{buffer}\n{paragraph}" if buffer else paragraph
                continue
            flush()
            # 超长段落：按句末滑窗切分，保留 overlap
            sentences = [seg for seg in SENTENCE_END.split(paragraph) if seg.strip()]
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) <= settings.chunk_size:
                    current += sentence
                    continue
                if current:
                    pieces.append(current.strip())
                    tail = current[-settings.chunk_overlap :] if settings.chunk_overlap else ""
                    current = tail + sentence
                else:
                    pieces.append(sentence[: settings.chunk_size].strip())
                    current = sentence[settings.chunk_size - settings.chunk_overlap :]
            if current.strip():
                pieces.append(current.strip())
        flush()
        return [piece for piece in pieces if piece]

    def _absorb_short_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """短块吸附：同 section 内把 <absorb_min_chars 的块并入相邻块。"""
        settings = self._settings.chunk
        if len(chunks) <= 1:
            return chunks
        result: list[Chunk] = []
        for chunk in chunks:
            if (
                chunk.type == "text"
                and len(chunk.content) < settings.absorb_min_chars
                and result
                and result[-1].type == "text"
                and result[-1].section == chunk.section
                and len(result[-1].content) + len(chunk.content) <= settings.chunk_size + settings.chunk_overlap
            ):
                merged = result[-1].model_copy(deep=True)
                merged.content = merged.content + "\n" + chunk.content
                merged.char_count = len(merged.content)
                merged.keywords = dedupe_keep_order([*merged.keywords, *chunk.keywords])
                result[-1] = merged
                continue
            result.append(chunk.model_copy(deep=True))
        return result

    def _table_block_content(self, table: Any) -> str:
        """表格块内容：头部标识 + 检索摘要行 + Markdown 表（整表不拆）。"""
        settings = self._settings.chunk
        page_label = f"第 {table.page} 页"
        if getattr(table, "page_end", 0):
            page_label = f"第 {table.page}-{table.page_end} 页（跨页合并）"
        header = f"[表格 {table.table_id}] {page_label}"
        summary = self._table_summary(table)
        parts = [header]
        if summary:
            parts.append(f"检索摘要：{summary}")
        if table.markdown:
            parts.append(table.markdown)
        return "\n".join(parts)[: 4000]

    def _table_summary(self, table: Any) -> str:
        """表头 + 首列 + 数值/百分比单元格拼接，便于语义模型对齐。"""
        try:
            rows = getattr(table, "rows", None) or []
            if not rows:
                return ""
            tokens: list[str] = []
            header = [cell for cell in rows[0] if cell and cell.strip()]
            tokens.extend(header[:6])
            for row in rows[1:8]:
                if row and row[0] and row[0].strip():
                    tokens.append(row[0].strip())
                for cell in row[1:]:
                    if cell and (AMOUNT_PATTERN.search(cell) or PERCENT_PATTERN.search(cell)):
                        tokens.append(cell.strip())
                        break
            summary = " ".join(dedupe_keep_order(tokens))
            limit = self._settings.chunk.table_summary_chars
            return summary[:limit]
        except Exception:
            logger.exception("app.core.chunker", "表格摘要行生成失败")
            return ""

    # ------------------------------------------------------------------
    def _make_chunk(
        self,
        index: int,
        doc_id: str,
        page: int,
        section: str,
        chunk_type: str,
        content: str,
        table_id: str | None = None,
    ) -> Chunk:
        """构造 Chunk（含关键词标签与释义页标记）。"""
        settings = self._settings
        keywords = self._keywords(content, chunk_type)
        is_boilerplate = self._is_boilerplate(page, content, settings)
        return Chunk(
            chunk_id=f"c{index:06d}",
            doc_id=doc_id,
            page=int(page),
            section=section[:200],
            type=chunk_type,  # type: ignore[arg-type]
            content=content,
            char_count=len(content),
            table_id=table_id,
            keywords=keywords,
            is_boilerplate=is_boilerplate,
        )

    @staticmethod
    def _keywords(content: str, chunk_type: str) -> list[str]:
        """命中领域关键词标签。"""
        text = content or ""
        hits = [label for label in KEYWORD_LABELS if label in text]
        if chunk_type == "table" and "表格" not in hits:
            hits.append("表格")
        return dedupe_keep_order(hits)

    @staticmethod
    def _is_boilerplate(page: int, content: str, settings: Any) -> bool:
        """释义页标记：前 30 页且（含释义标记 或 公司全称反复出现）。"""
        retrieval = settings.retrieval
        if int(page) > retrieval.front_page_cutoff:
            return False
        text = content or ""
        if "释义" in text or "基本用语" in text:
            return True
        return text.count(COMPANY_NAME) >= retrieval.boilerplate_company_hits

    # ------------------------------------------------------------------
    @trace
    def save(self, chunks: list[Chunk], output_dir: Path | str | None = None, name: str = "chunks") -> Path:
        """保存为 JSONL（一行一块）。"""
        target_dir = Path(output_dir) if output_dir else self._settings.paths.data_processed
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{name}.jsonl"
        try:
            with open(target, "w", encoding="utf-8", newline="\n") as handle:
                for chunk in chunks:
                    handle.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")
            logger.info("app.core.chunker", "分块已保存", path=str(target), count=len(chunks))
            return target
        except Exception as exc:
            logger.exception("app.core.chunker", "分块保存失败", path=str(target))
            raise StorageError(f"分块保存失败: {exc}", detail={"path": str(target)}) from exc

    @staticmethod
    def load(path: Path | str) -> list[Chunk]:
        """读取 chunks.jsonl。"""
        target = Path(path)
        try:
            chunks: list[Chunk] = []
            with open(target, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        chunks.append(Chunk(**json.loads(line)))
            logger.info("app.core.chunker", "分块已加载", path=str(target), count=len(chunks))
            return chunks
        except Exception as exc:
            logger.exception("app.core.chunker", "分块加载失败", path=str(target))
            raise StorageError(f"分块加载失败: {exc}", detail={"path": str(target)}) from exc

    @staticmethod
    def stats(chunks: list[Chunk]) -> dict[str, Any]:
        """统计：总数 / 文本数 / 表格数 / 平均长度 / 长度中位数 / 释义块数。"""
        lengths = sorted(len(chunk.content) for chunk in chunks)
        total = len(chunks)
        median = lengths[total // 2] if total else 0
        return {
            "total": total,
            "text": sum(1 for chunk in chunks if chunk.type == "text"),
            "table": sum(1 for chunk in chunks if chunk.type == "table"),
            "avg_chars": round(sum(lengths) / total, 1) if total else 0.0,
            "len_p50": median,
            "len_max": lengths[-1] if lengths else 0,
            "boilerplate_count": sum(1 for chunk in chunks if chunk.is_boilerplate),
        }


_chunker: Chunker | None = None
_lock = threading.Lock()


def get_chunker() -> Chunker:
    """获取进程级分块器单例。"""
    global _chunker
    with _lock:
        if _chunker is None:
            _chunker = Chunker()
    return _chunker
