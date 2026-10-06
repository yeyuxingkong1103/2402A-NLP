"""PDF 解析：PyMuPDF 取文 + ``page.find_tables()`` 提表格 + 跨页合并 + 结构化条目。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / PDF 解析优化（对应 设计/接口设计.md §2.3、优化方案设计.md §2.1 M1.1~M1.4）

重要事实（环境事实 2.2）：工单条文写「用 pdfplumber」，但本机 **pdfplumber 不可用且断网无法安装**，
因此改用 PyMuPDF（fitz）等价的 ``page.find_tables()`` 提表格并转 Markdown；本文件在
``parser_capabilities()`` 中如实声明该替换（不伪造依赖）。

产物（``研发/data/processed/``）：
- ``<name>.parsed.json``  —— 整篇结构化结果（pages/tables/blocks/统计）
- ``<name>.pages.txt``    —— 每页正文（人工抽查用）
- ``<name>.tables.jsonl`` —— 一行一表（含 page/page_end/bbox/markdown）
- ``<name>.blocks.jsonl`` —— 一行一条结构化条目（page/section/type/content/bbox）
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.errors import PDFParseError
from app.core.logging_conf import logger, trace
from app.core.text_utils import looks_like_heading, normalize_text, safe_filename
from app.models.schemas import ParsedDocument, ParsedPage, ParsedTable, StructuredBlock

try:  # pragma: no cover - 依赖可用性分支
    import pymupdf as fitz  # PyMuPDF 1.24+ 推荐写法

    HAS_PYMUPDF = True
except Exception:  # pragma: no cover
    try:
        import fitz  # 兼容旧版导入名

        HAS_PYMUPDF = True
    except Exception:
        fitz = None  # type: ignore[assignment]
        HAS_PYMUPDF = False

try:  # pragma: no cover - 本机不可用（如实声明）
    import pdfplumber  # noqa: F401

    HAS_PDFPLUMBER = True
except Exception:  # pragma: no cover
    HAS_PDFPLUMBER = False

#: 标题层级判定（招股书结构：第X节 > 一、 > （一） > 1、）
HEADING_LEVELS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^第[一二三四五六七八九十百]+[章节篇]"),
    re.compile(r"^[一二三四五六七八九十]+、"),
    re.compile(r"^[（(][一二三四五六七八九十]+[)）]"),
    re.compile(r"^\d+(\.\d+)*[\s、.]"),
    re.compile(r"^[（(]\d+[)）]"),
)

#: 表格 Markdown 最大行数（超长表截断并留痕，避免单个块过大污染检索）
MAX_TABLE_ROWS = 120


def _heading_level(line: str) -> int:
    """返回标题层级下标（越小层级越高）；非标题返回 -1。

    额外约束（避免把"1.00 元每股发行价格""2019 年12 月24 日"这类数字行误判为标题）：
    标题必须 ≤40 字符、不含金额/日期单位、不以句号结尾。
    """
    stripped = (line or "").strip()
    if not stripped or len(stripped) > 40:
        return -1
    if stripped.endswith(("。", "；", ";", "，", ",")):
        return -1
    if re.search(r"(年|月|日|元|万元|亿元|%|％)", stripped):
        return -1
    if not looks_like_heading(stripped):
        return -1
    # 中文标题至少含 2 个汉字，且数字占比不过高（排除 "36.58"、"132.85" 这类表格数字行）
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", stripped))
    digit_ratio = len(re.findall(r"\d", stripped)) / max(1, len(stripped))
    if cjk_chars < 2 or digit_ratio > 0.3:
        return -1
    for index, pattern in enumerate(HEADING_LEVELS):
        if pattern.match(stripped):
            return index
    return len(HEADING_LEVELS) - 1


def _table_to_markdown(rows: list[list[str]], max_cell_chars: int = 200) -> str:
    """把二维单元格转成 Markdown 表格（单元格内换行与竖线做转义）。"""
    cleaned: list[list[str]] = []
    for row in rows:
        cells = []
        for cell in row:
            text = "" if cell is None else str(cell)
            text = text.replace("|", "／").replace("\n", " ").strip()
            if len(text) > max_cell_chars:
                text = text[:max_cell_chars] + "…"
            cells.append(text)
        cleaned.append(cells)
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    for row in cleaned:
        while len(row) < width:
            row.append("")
    header = cleaned[0]
    if not any(cell for cell in header):
        header = [f"列{index + 1}" for index in range(width)]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


class PDFParser:
    """PDF 解析器（PyMuPDF）。"""

    def __init__(self, settings=None) -> None:
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    def clean_page_text(self, raw: str) -> str:
        """清洗单页文本：去页眉页脚、合并断行、折叠空白。"""
        try:
            if not raw:
                return ""
            patterns = [re.compile(pattern) for pattern in self._settings.pdf.header_footer_patterns]
            kept: list[str] = []
            for line in raw.split("\n"):
                stripped = line.strip()
                if not stripped:
                    kept.append("")
                    continue
                if len(stripped) < self._settings.pdf.min_line_chars:
                    continue
                if any(pattern.match(stripped) for pattern in patterns):
                    continue
                kept.append(stripped)
            text = normalize_text("\n".join(kept))
            # 合并被 PDF 硬换行切断的句子（上一行未以句末标点结束且下一行不以标题/编号开头）
            merged: list[str] = []
            for line in text.split("\n"):
                if (
                    merged
                    and merged[-1]
                    and not re.search(r"[。！？；：）】\d]$", merged[-1])
                    and _heading_level(line) < 0
                    and not re.match(r"^[（(【\[]", line)
                ):
                    merged[-1] = merged[-1] + line
                else:
                    merged.append(line)
            return normalize_text("\n".join(merged))
        except Exception:
            logger.exception("app.core.pdf_parser", "页面文本清洗失败，返回原始文本")
            return raw or ""

    # ------------------------------------------------------------------
    @trace
    def parse(
        self,
        pdf_path: Path | str,
        doc_id: str | None = None,
        max_pages: int = 0,
        extract_tables: bool | None = None,
    ) -> ParsedDocument:
        """解析整篇 PDF（单页/单表失败只计数不中断）。"""
        path = Path(pdf_path)
        if not path.exists():
            logger.error("app.core.pdf_parser", "PDF 文件不存在", path=str(path))
            raise PDFParseError(f"PDF 文件不存在: {path}", detail={"path": str(path)})
        if not HAS_PYMUPDF:
            raise PDFParseError("PyMuPDF(pymupdf) 不可用，无法解析 PDF", detail={"pymupdf": False})
        want_tables = self._settings.pdf.extract_tables if extract_tables is None else extract_tables
        started = time.perf_counter()
        try:
            document = fitz.open(str(path))
        except Exception as exc:
            logger.exception("app.core.pdf_parser", "PDF 打开失败（文件可能损坏）", path=str(path))
            raise PDFParseError(f"PDF 打开失败: {exc}", detail={"path": str(path)}) from exc

        pages: list[ParsedPage] = []
        tables: list[ParsedTable] = []
        blocks: list[StructuredBlock] = []
        heading_stack: list[str] = []
        table_errors = 0
        text_chars = 0
        max_levels = self._settings.chunk.max_heading_levels
        try:
            total = document.page_count
            limit = total if not max_pages else min(int(max_pages), total)
            for page_index in range(limit):
                page_no = page_index + 1
                page = document.load_page(page_index)
                cleaned = self.clean_page_text(page.get_text("text"))
                text_chars += len(cleaned)
                # 1) 先更新标题栈（保证本页表格也能继承到标题）
                for line in cleaned.split("\n") if cleaned else []:
                    level = _heading_level(line)
                    if level >= 0:
                        heading_stack = heading_stack[:level]
                        heading_stack.append(line.strip())
                # 2) 正文块（保留 bbox）
                page_blocks: list[StructuredBlock] = []
                try:
                    for item in page.get_text("blocks"):
                        x0, y0, x1, y1, content = item[0], item[1], item[2], item[3], item[4]
                        body = self.clean_page_text(str(content))
                        if not body:
                            continue
                        page_blocks.append(
                            StructuredBlock(
                                page=page_no,
                                section=" > ".join(heading_stack[-max_levels:]),
                                type="text",
                                content=body,
                                bbox=[float(x0), float(y0), float(x1), float(y1)],
                            )
                        )
                except Exception:
                    logger.exception("app.core.pdf_parser", "按块取文失败，退回整页文本", page=page_no)
                # 3) 表格
                page_tables: list[ParsedTable] = []
                if want_tables:
                    page_tables, errors = self._extract_tables(page, page_no, heading_stack)
                    table_errors += errors
                section_now = " > ".join(heading_stack[-2:]) if heading_stack else ""
                pages.append(
                    ParsedPage(
                        page=page_no,
                        text=cleaned,
                        tables=[{"table_id": table.table_id, "rows": len(table.rows)} for table in page_tables],
                        char_count=len(cleaned),
                    )
                )
                for block in page_blocks:
                    block.section = block.section or section_now
                    blocks.append(block)
                for table in page_tables:
                    table.section = table.section or section_now
                    tables.append(table)
                    blocks.append(
                        StructuredBlock(
                            page=table.page,
                            section=table.section,
                            type="table",
                            content=table.markdown,
                            bbox=list(table.bbox),
                        )
                    )
            merged_tables = (
                self.merge_cross_page_tables(tables) if self._settings.pdf.merge_cross_page_tables else tables
            )
            document_obj = ParsedDocument(
                doc_id=doc_id or path.stem,
                source_path=str(path),
                title=path.stem,
                page_count=limit,
                pages=pages,
                tables=merged_tables,
                blocks=self._rebuild_blocks(merged_tables, blocks),
                parser="pymupdf+tables(fitz.find_tables；pdfplumber 本机不可用故替换)",
                parsed_at=datetime.now(timezone.utc).astimezone(),
                ocr_skipped=True,
                table_errors=table_errors,
                merged_table_count=sum(1 for table in merged_tables if table.page_end),
            )
            elapsed = (time.perf_counter() - started) * 1000
            logger.info(
                "app.core.pdf_parser",
                "文字版文档，跳过 OCR（PyMuPDF 直接取文字层）",
                page_count=limit,
                text_chars=text_chars,
            )
            logger.info(
                "app.core.pdf_parser",
                "PDF 解析完成",
                doc_id=document_obj.doc_id,
                pages=limit,
                tables=len(merged_tables),
                blocks=len(document_obj.blocks),
                merged_tables=document_obj.merged_table_count,
                table_errors=table_errors,
                elapsed_ms=round(elapsed, 2),
            )
            return document_obj
        except PDFParseError:
            raise
        except Exception as exc:
            logger.exception("app.core.pdf_parser", "PDF 解析中断", path=str(path))
            raise PDFParseError(f"PDF 解析失败: {exc}", detail={"path": str(path)}) from exc
        finally:
            try:
                document.close()
            except Exception:
                logger.exception("app.core.pdf_parser", "关闭 PDF 句柄失败")

    # ------------------------------------------------------------------
    def _extract_tables(
        self, page: Any, page_no: int, heading_stack: list[str]
    ) -> tuple[list[ParsedTable], int]:
        """提取单页表格。

        Returns:
            ``(表格列表, 失败计数)``；单页/单表失败只记日志与计数，不中断整篇解析。
        """
        results: list[ParsedTable] = []
        errors = 0
        try:
            finder = page.find_tables()
            raw_tables = list(getattr(finder, "tables", []) or [])
        except Exception:
            logger.exception("app.core.pdf_parser", "表格识别失败（该页跳过表格）", page=page_no)
            return results, 1
        for order, table in enumerate(raw_tables, start=1):
            try:
                rows_raw = table.extract() or []
                rows = [[("" if cell is None else str(cell)) for cell in row] for row in rows_raw]
                if not any(any(cell.strip() for cell in row) for row in rows):
                    logger.debug("app.core.pdf_parser", "空表格跳过", page=page_no, order=order)
                    continue
                truncated = False
                if len(rows) > MAX_TABLE_ROWS:
                    rows = rows[:MAX_TABLE_ROWS]
                    truncated = True
                markdown = _table_to_markdown(rows, self._settings.pdf.max_cell_chars)
                if truncated:
                    markdown += f"\n（表格过长，仅保留前 {MAX_TABLE_ROWS} 行）"
                bbox = [float(value) for value in (getattr(table, "bbox", None) or (0, 0, 0, 0))]
                results.append(
                    ParsedTable(
                        table_id=f"p{page_no}_t{order}",
                        page=page_no,
                        page_end=0,
                        section=" > ".join(heading_stack[-2:]) if heading_stack else "",
                        markdown=markdown,
                        rows=rows,
                        bbox=bbox,
                        merged_from=[],
                    )
                )
            except Exception:
                errors += 1
                logger.exception("app.core.pdf_parser", "单表提取失败（跳过该表）", page=page_no, order=order)
        return results, errors

    # ------------------------------------------------------------------
    @trace
    def merge_cross_page_tables(self, tables: list[ParsedTable]) -> list[ParsedTable]:
        """合并跨页续表：相邻页 + 列数一致 + 表头相似（或空表头）+ 列宽近似。"""
        try:
            if not tables:
                return []
            ordered = sorted(tables, key=lambda item: (item.page, item.table_id))
            merged: list[ParsedTable] = []
            index = 0
            while index < len(ordered):
                current = ordered[index].model_copy(deep=True)
                cursor = index + 1
                while cursor < len(ordered):
                    nxt = ordered[cursor]
                    if nxt.page != (current.page_end or current.page) + 1:
                        break
                    if not self._compatible(current, nxt):
                        logger.debug(
                            "app.core.pdf_parser",
                            "跨页表格结构不兼容，保持独立",
                            left=current.table_id,
                            right=nxt.table_id,
                        )
                        break
                    body_rows = nxt.rows[1:] if self._same_header(current, nxt) else nxt.rows
                    current.markdown = current.markdown + "\n" + _table_to_markdown(
                        body_rows or nxt.rows, self._settings.pdf.max_cell_chars
                    )
                    current.rows = current.rows + body_rows
                    current.page_end = nxt.page
                    current.merged_from = [*current.merged_from, nxt.table_id]
                    cursor += 1
                merged.append(current)
                index = cursor
            return merged
        except Exception:
            logger.exception("app.core.pdf_parser", "跨页表格合并失败，返回原始表格列表")
            return tables

    def _compatible(self, left: ParsedTable, right: ParsedTable) -> bool:
        """列数一致 + 列宽近似 + （表头相似 或 右侧首行是数据行）才视为续表。"""
        left_cols = len(left.rows[0]) if left.rows else 0
        right_cols = len(right.rows[0]) if right.rows else 0
        if left_cols == 0 or right_cols == 0 or left_cols != right_cols:
            return False
        if len(left.bbox) == 4 and len(right.bbox) == 4:
            left_width = left.bbox[2] - left.bbox[0]
            right_width = right.bbox[2] - right.bbox[0]
            if left_width > 0 and right_width > 0:
                diff = abs(left_width - right_width) / max(left_width, right_width)
                if diff > 0.15:
                    return False
        left_header = " ".join(left.rows[0]) if left.rows else ""
        right_header = " ".join(right.rows[0]) if right.rows else ""
        if not left_header.strip() or not right_header.strip():
            return True
        if self._same_header(left, right):
            return True
        # 续表特征：下一页表格的首行是数据行（多数单元格为数字）而非表头
        return self._looks_like_data_row(right.rows[0] if right.rows else [])

    @staticmethod
    def _looks_like_data_row(row: list[str]) -> bool:
        """判断一行是否更像数据行（数字/金额/百分比占比高）。"""
        cells = [cell.strip() for cell in row if cell and cell.strip()]
        if len(cells) < 2:
            return False
        numeric = sum(1 for cell in cells if re.fullmatch(r"[\d.,%％\-—\s万元亿元/年月日]+", cell))
        return numeric / len(cells) >= 0.5

    @staticmethod
    def _same_header(left: ParsedTable, right: ParsedTable) -> bool:
        """表头字符集合 Jaccard ≥ 0.5 视为同一张表。"""
        left_set = {item.strip() for item in (left.rows[0] if left.rows else []) if item and item.strip()}
        right_set = {item.strip() for item in (right.rows[0] if right.rows else []) if item and item.strip()}
        if not left_set or not right_set:
            return False
        union = left_set | right_set
        return len(left_set & right_set) / len(union) >= 0.5

    @staticmethod
    def _rebuild_blocks(merged_tables: list[ParsedTable], blocks: list[StructuredBlock]) -> list[StructuredBlock]:
        """用合并后的表格替换原表格块，并按页码+类型排序。"""
        table_blocks = [
            StructuredBlock(
                page=table.page,
                section=table.section,
                type="table",
                content=table.markdown,
                bbox=list(table.bbox),
            )
            for table in merged_tables
        ]
        text_blocks = [block for block in blocks if block.type == "text"]
        combined = [*text_blocks, *table_blocks]
        combined.sort(key=lambda item: (item.page, 0 if item.type == "text" else 1))
        return combined

    # ------------------------------------------------------------------
    @trace
    def save(self, document: ParsedDocument, output_dir: Path | str | None = None) -> dict[str, Path]:
        """保存解析产物（parsed.json / pages.txt / tables.jsonl / blocks.jsonl）。"""
        target_dir = Path(output_dir) if output_dir else self._settings.paths.data_processed
        target_dir.mkdir(parents=True, exist_ok=True)
        name = safe_filename(document.title or document.doc_id or "document")
        paths: dict[str, Path] = {}
        try:
            parsed_path = target_dir / f"{name}.parsed.json"
            parsed_path.write_text(
                json.dumps(document.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
            )
            paths["parsed_json"] = parsed_path

            pages_path = target_dir / f"{name}.pages.txt"
            with open(pages_path, "w", encoding="utf-8", newline="\n") as handle:
                for page in document.pages:
                    handle.write(f"===== 第 {page.page} 页 =====\n{page.text}\n")
            paths["pages_txt"] = pages_path

            tables_path = target_dir / f"{name}.tables.jsonl"
            with open(tables_path, "w", encoding="utf-8", newline="\n") as handle:
                for table in document.tables:
                    handle.write(json.dumps(table.model_dump(mode="json"), ensure_ascii=False) + "\n")
            paths["tables_jsonl"] = tables_path

            blocks_path = target_dir / f"{name}.blocks.jsonl"
            with open(blocks_path, "w", encoding="utf-8", newline="\n") as handle:
                for block in document.blocks:
                    handle.write(json.dumps(block.model_dump(mode="json"), ensure_ascii=False) + "\n")
            paths["blocks_jsonl"] = blocks_path

            logger.info(
                "app.core.pdf_parser",
                "解析产物已保存",
                **{key: str(value) for key, value in paths.items()},
            )
            return paths
        except Exception as exc:
            logger.exception("app.core.pdf_parser", "解析产物保存失败", output_dir=str(target_dir))
            raise PDFParseError(f"解析产物保存失败: {exc}", detail={"output_dir": str(target_dir)}) from exc

    # ------------------------------------------------------------------
    @staticmethod
    def load_parsed(path: Path | str) -> ParsedDocument:
        """读取已保存的 parsed.json（离线评估/索引重建用）。"""
        target = Path(path)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            return ParsedDocument(**payload)
        except Exception as exc:
            logger.exception("app.core.pdf_parser", "读取解析产物失败", path=str(target))
            raise PDFParseError(f"读取解析产物失败: {exc}", detail={"path": str(target)}) from exc


_parser: PDFParser | None = None
_lock = threading.Lock()


def get_pdf_parser() -> PDFParser:
    """获取进程级解析器单例。"""
    global _parser
    with _lock:
        if _parser is None:
            _parser = PDFParser()
    return _parser


def parser_capabilities() -> dict[str, bool]:
    """如实声明解析能力（pdfplumber 本机不可用，OCR 不可用）。"""
    return {"pymupdf": HAS_PYMUPDF, "pdfplumber": HAS_PDFPLUMBER, "ocr": False}
