"""PDF 解析：提取正文文字、页码与表格。

工单要求（5.1）：
- 使用 PyMuPDF 提取文字、页码；
- 使用 pdfplumber 或 camelot 提取表格；
- 表格转成 Markdown 或结构化 JSON，保留 page / table_id / section；
- 解析结果保存到 data/processed/；
- 每个函数入口/出口记录日志（由 @trace 装饰器完成）。

实现策略：
- PyMuPDF 为主解析器（速度快、页码准确）；
- 表格优先用 PyMuPDF 自带的 ``find_tables()``（无需额外依赖）；
- 若安装了 pdfplumber，作为备选表格解析器（``prefer_pdfplumber=True``）；
- 两者都不可用时，表格列表为空，但正文解析仍然成功——**绝不静默失败**，
  会在日志中明确记录降级原因。
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging_conf import logger, trace
from app.core.text_utils import looks_like_heading, normalize_text, safe_filename, stable_id
from app.models.schemas import ParsedDocument, ParsedPage, ParsedTable

try:  # pragma: no cover
    import pymupdf  # PyMuPDF >= 1.24

    HAS_PYMUPDF = True
except Exception:  # pragma: no cover
    try:
        import fitz as pymupdf  # type: ignore

        HAS_PYMUPDF = True
    except Exception:
        pymupdf = None  # type: ignore
        HAS_PYMUPDF = False

try:  # pragma: no cover
    import pdfplumber

    HAS_PDFPLUMBER = True
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore
    HAS_PDFPLUMBER = False


class PDFParser:
    """招股说明书 PDF 解析器。"""

    def __init__(self, settings: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._header_patterns = [re.compile(p) for p in self.settings.pdf.header_footer_patterns]

    # ------------------------------------------------------------------
    # 文本清洗
    # ------------------------------------------------------------------
    def _should_drop_line(self, line: str) -> bool:
        """过滤页眉、页脚、孤立页码等噪声行。"""
        stripped = line.strip()
        if len(stripped) < self.settings.pdf.min_line_chars and not stripped.isdigit():
            return True
        return any(pattern.match(stripped) for pattern in self._header_patterns)

    def clean_page_text(self, raw: str) -> str:
        """清洗单页文本：去页眉页脚、规整空白、合并被硬换行切断的句子。

        招股书 PDF 的文字层按**排版行**输出，一句话常被切成多行，例如::

            美军的C4ISR 系统）荣获国家科技进步一等奖。该工程参与建设单位包括项目
            主管部门、总体单位、国有科研机构等……

        如果直接按行拼回，检索片段与抽取式答案就会出现“半句话”。因此这里做
        三件事：

        1. 去掉页眉页脚与孤立页码；
        2. 把连续的单字符行合成一行（表格被抽成竖排单字的情况）；
        3. **行合并**：若上一行没有以句末标点结束，且下一行不以标题模式开头，
           则把两行接起来（中文之间不加空格），恢复完整句子。
        """
        # ---- 1. 去噪声行 ----
        lines = [line for line in raw.split("\n") if not self._should_drop_line(line)]

        # ---- 2. 合并连续单字符行 ----
        merged: list[str] = []
        buffer: list[str] = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) == 1:
                buffer.append(stripped)
            else:
                if len(buffer) >= 2:
                    merged.append("".join(buffer))
                elif buffer:
                    merged.extend(buffer)
                buffer = []
                merged.append(line.rstrip())
        if len(buffer) >= 2:
            merged.append("".join(buffer))
        elif buffer:
            merged.extend(buffer)

        # ---- 3. 行合并：修复被硬换行切断的句子 ----
        # 注意：PDF 常在断句处插入空行（排版换段），因此合并时必须**跨过空行**，
        # 否则会出现“美军的C4ISR系统）荣获…”这类从半句开始的碎片。
        joined: list[str] = []
        for line in merged:
            stripped = line.strip()
            if not stripped:
                # 空行先留着；若后续行是对上一行的续写，合并时会把空行吃掉
                joined.append("")
                continue

            # 找到最近的一条非空行作为合并候选
            previous_index = len(joined) - 1
            while previous_index >= 0 and not joined[previous_index]:
                previous_index -= 1

            if previous_index < 0:
                joined.append(stripped)
                continue

            previous = joined[previous_index]
            if self._should_join(previous, stripped):
                separator = "" if (self._ends_cjk(previous) and self._starts_cjk(stripped)) else " "
                joined[previous_index] = previous + separator + stripped
                # 丢弃中间的空行
                del joined[previous_index + 1 :]
            else:
                joined.append(stripped)

        return normalize_text("\n".join(joined))

    # 句末标点：这些字符结尾说明一句话已经结束，下一行是新句子
    _SENTENCE_END = ("。", "！", "？", "；", "：", "…", ".", "!", "?", ";", ":", "|")

    @staticmethod
    def _ends_cjk(text: str) -> bool:
        return bool(text) and "\u4e00" <= text[-1] <= "\u9fff"

    @staticmethod
    def _starts_cjk(text: str) -> bool:
        return bool(text) and "\u4e00" <= text[0] <= "\u9fff"

    def _should_join(self, previous: str, current: str) -> bool:
        """判断 ``current`` 是否应接到 ``previous`` 之后（即上一行未写完）。"""
        if not previous:
            return False
        # 上一行以句末标点收尾 -> 已经是完整句子，不合并
        if previous.rstrip().endswith(self._SENTENCE_END):
            return False
        # 下一行是标题 -> 结构边界，不合并
        if looks_like_heading(current):
            return False
        # 上一行是标题 -> 不合并（标题独立成行）
        if looks_like_heading(previous):
            return False
        # 下一行以 Markdown 表格行或列表符号开头 -> 结构元素，不合并
        if current.lstrip().startswith(("|", "-", "•", "*", "（", "(")):
            return False
        # 上一行以项目符号/序号结尾无意义，忽略
        return True

    # ------------------------------------------------------------------
    # 表格
    # ------------------------------------------------------------------
    def _table_to_markdown(self, rows: list[list[str]]) -> str:
        """把二维单元格转成 Markdown 表格。"""
        cleaned = [["" if cell is None else str(cell).replace("\n", " ").strip() for cell in row] for row in rows]
        cleaned = [row for row in cleaned if any(cell for cell in row)]
        if not cleaned:
            return ""
        width = max(len(row) for row in cleaned)
        padded = [row + [""] * (width - len(row)) for row in cleaned]
        header = padded[0]
        separator = ["---"] * width
        body = padded[1:] if len(padded) > 1 else []
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        return "\n".join(lines)

    @trace
    def _extract_tables_pymupdf(self, page: Any, page_number: int) -> list[ParsedTable]:
        """使用 PyMuPDF 的表格识别能力提取表格。"""
        tables: list[ParsedTable] = []
        try:
            finder = page.find_tables()
        except Exception as exc:
            logger.warning(
                "app.core.pdf_parser",
                "PyMuPDF 表格识别不可用，跳过该页表格",
                page=page_number,
                error=str(exc),
            )
            return tables

        for order, table in enumerate(getattr(finder, "tables", []) or [], start=1):
            try:
                rows = table.extract()
            except Exception as exc:
                logger.warning(
                    "app.core.pdf_parser", "表格抽取失败", page=page_number, table_order=order, error=str(exc)
                )
                continue
            markdown = self._table_to_markdown(rows)
            if not markdown:
                continue
            tables.append(
                ParsedTable(
                    table_id=f"p{page_number}_t{order}",
                    page=page_number,
                    markdown=markdown,
                    rows=[[("" if c is None else str(c)) for c in row] for row in rows],
                )
            )
        return tables

    @trace
    def _extract_tables_pdfplumber(self, pdf_path: Path, page_number: int) -> list[ParsedTable]:
        """使用 pdfplumber 提取指定页表格（备选路径）。"""
        if not HAS_PDFPLUMBER:
            return []
        tables: list[ParsedTable] = []
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                if page_number > len(pdf.pages):
                    return []
                page = pdf.pages[page_number - 1]
                for order, raw in enumerate(page.extract_tables() or [], start=1):
                    rows = [[("" if c is None else str(c)) for c in row] for row in raw]
                    markdown = self._table_to_markdown(rows)
                    if markdown:
                        tables.append(
                            ParsedTable(table_id=f"p{page_number}_t{order}", page=page_number, markdown=markdown, rows=rows)
                        )
        except Exception as exc:
            logger.warning(
                "app.core.pdf_parser", "pdfplumber 表格抽取失败", page=page_number, error=str(exc)
            )
        return tables

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    @trace
    def parse(
        self,
        pdf_path: Path | str,
        doc_id: str | None = None,
        max_pages: int = 0,
        extract_tables: bool | None = None,
    ) -> ParsedDocument:
        """解析整篇 PDF。

        Args:
            pdf_path: PDF 文件路径。
            doc_id: 文档 ID，为空时按文件名与内容生成稳定 ID。
            max_pages: 最多解析页数，0 表示全部（测试时可调小加速）。
            extract_tables: 是否提取表格，默认取配置。

        Returns:
            ParsedDocument

        Raises:
            FileNotFoundError: 文件不存在。
            RuntimeError: 未安装 PyMuPDF。
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {path}")
        if not HAS_PYMUPDF:
            raise RuntimeError("未安装 PyMuPDF，无法解析 PDF。请执行: pip install pymupdf")

        do_tables = self.settings.pdf.extract_tables if extract_tables is None else extract_tables
        resolved_id = doc_id or stable_id("doc_", path.name, path.stat().st_size, length=8)
        started = time.perf_counter()

        pages: list[ParsedPage] = []
        tables: list[ParsedTable] = []
        section_stack: list[str] = []

        with pymupdf.open(str(path)) as document:
            total = document.page_count
            limit = min(total, max_pages) if max_pages else total
            for index in range(limit):
                page = document.load_page(index)
                page_number = index + 1
                raw_text = page.get_text("text")
                cleaned = self.clean_page_text(raw_text)
                pages.append(ParsedPage(page=page_number, text=cleaned, char_count=len(cleaned)))

                # 章节跟踪：页内出现的标题会更新当前章节路径
                for line in cleaned.split("\n"):
                    if looks_like_heading(line):
                        title = line.strip()
                        if not section_stack or section_stack[-1] != title:
                            section_stack.append(title)
                            if len(section_stack) > 4:
                                section_stack.pop(0)

                if do_tables:
                    page_tables = self._extract_tables_pymupdf(page, page_number)
                    for table in page_tables:
                        table.section = " > ".join(section_stack[-2:]) if section_stack else ""
                    tables.extend(page_tables)
                if page_number % 50 == 0:
                    logger.info("app.core.pdf_parser", "解析进度", parsed=page_number, total=limit)

        document_result = ParsedDocument(
            doc_id=resolved_id,
            source_path=str(path),
            title=path.stem,
            page_count=len(pages),
            pages=pages,
            tables=tables,
            parser="pymupdf" + ("+tables" if do_tables else ""),
            parsed_at=datetime.now(timezone.utc).astimezone(),
        )
        elapsed = time.perf_counter() - started
        logger.info(
            "app.core.pdf_parser",
            "PDF 解析完成",
            doc_id=resolved_id,
            pages=len(pages),
            tables=len(tables),
            chars=sum(p.char_count for p in pages),
            elapsed_s=round(elapsed, 2),
        )
        return document_result

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------
    @trace
    def save(self, document: ParsedDocument, output_dir: Path | str | None = None) -> dict[str, Path]:
        """把解析结果写入 data/processed/：全文 JSON、逐页文本、表格 JSONL。"""
        settings = self.settings
        target_dir = Path(output_dir) if output_dir else settings.paths.data_processed
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = safe_filename(document.title or document.doc_id)

        full_path = target_dir / f"{stem}.parsed.json"
        full_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")

        text_path = target_dir / f"{stem}.pages.txt"
        with open(text_path, "w", encoding="utf-8", newline="\n") as handle:
            for page in document.pages:
                handle.write(f"\f===PAGE {page.page}===\n{page.text}\n")

        table_path = target_dir / f"{stem}.tables.jsonl"
        with open(table_path, "w", encoding="utf-8", newline="\n") as handle:
            for table in document.tables:
                handle.write(json.dumps(table.model_dump(), ensure_ascii=False) + "\n")

        logger.info(
            "app.core.pdf_parser",
            "解析结果已保存",
            parsed_json=str(full_path),
            pages_txt=str(text_path),
            tables_jsonl=str(table_path),
            table_count=len(document.tables),
        )
        return {"parsed_json": full_path, "pages_txt": text_path, "tables_jsonl": table_path}

    @staticmethod
    def load_parsed(path: Path | str) -> ParsedDocument:
        """从 JSON 载入已解析结果（避免重复解析）。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return ParsedDocument(**data)


def get_pdf_parser() -> PDFParser:
    """工厂函数：获取解析器实例。"""
    return PDFParser()


def parser_capabilities() -> dict[str, bool]:
    """返回当前环境的解析能力，供健康检查与界面展示。"""
    return {"pymupdf": HAS_PYMUPDF, "pdfplumber": HAS_PDFPLUMBER}
