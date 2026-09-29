# app/core/table_service.py
"""pdfplumber 表格提取：把表格还原成 Markdown。

## 它在链路里的位置

    MinerU 可用  →  MinerU 一次解析出版面 / OCR / 公式 / 表格（主路径）
    MinerU 不可用 →  PyMuPDF 取文本层 + ocr_service 补图表文字 + 本模块补表格（兜底）

## 为什么需要单独提表格

PyMuPDF 取到的是**按阅读顺序拍平的纯文本**。一个三行四列的表格，
提出来会变成十二个格子里的字连成的一串，行列对应关系完全丢失 ——
对"某年某产品的收益率是多少"这类问题，这串文字是没法用的。
pdfplumber 能识别表格线并还原成行列结构。

## 代价

pdfplumber 逐页解析较慢，因此只在兜底路径下调用。
"""
import re
from typing import Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

MAX_CELL_CHARS = 120    # 单元格超长则截断：合并单元格被误判时会长得离谱，会撑爆上下文
MIN_ROWS = 2            # 少于两行不成表（一行多半是把带框的标题误认成表格）
MIN_COLS = 2            # 少于两列不成表（单列多半是把带框的标题/图注误认成表格）

# Markdown 表格里的竖线与换行必须转义，否则表格结构会被破坏
_CELL_CLEAN = re.compile(r"\s*\n\s*")


class TableService:
    """pdfplumber 表格提取，带不可用降级。"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._failed = False

    def available(self) -> bool:
        """pdfplumber 是否可用。"""
        if not self.enabled:
            return False
        if self._failed:
            return False
        try:
            import pdfplumber  # noqa: F401
            return True
        except Exception as e:
            self._failed = True
            logger.warning("pdfplumber 不可用，表格将退化为纯文本: %s", e)
            return False

    def extract_document_tables(self, pdf_path: str,
                                page_count: int) -> Dict[int, List[str]]:
        """一次打开、逐页提取，返回 `{页码: [Markdown 表格]}`。

        页码从 **1** 开始 —— 与 PyMuPDF 的 0-based 页码不同。

        刻意做成「一次打开全篇」：`pdfplumber.open()` 每次都要解析整份 PDF 的
        交叉引用表，按页调用等于把同一份文件重复打开 N 次（28 页文档就是 28 次）。
        兜底路径本已最慢，不该再叠这个开销。
        """
        if not self.available():
            return {}

        import pdfplumber
        out: Dict[int, List[str]] = {}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_no in range(1, min(page_count, len(pdf.pages)) + 1):
                    found = self._tables_of_page(pdf.pages[page_no - 1])
                    if found:
                        out[page_no] = found
        except Exception as e:                              # pragma: no cover
            logger.warning("提取表格失败 %s: %s", pdf_path, e)
        return out

    def _tables_of_page(self, page) -> List[str]:
        """提取单页表格并转 Markdown；不合格的表格在这里被挡掉。"""
        try:
            tables = page.extract_tables() or []
        except Exception as e:                              # pragma: no cover
            logger.warning("提取表格失败: %s", e)
            return []

        out = []
        for table in tables:
            if not table or len(table) < MIN_ROWS:
                continue
            # 单列表格几乎都是带框的标题或图注 —— 那段文字正文里本来就有，
            # 收进来只会在【表格数据】里制造重复噪声
            if max(len(r or []) for r in table) < MIN_COLS:
                continue
            markdown = self._to_markdown(table)
            if markdown:
                out.append(markdown)
        return out

    @staticmethod
    def _to_markdown(table: List[List[Optional[str]]]) -> str:
        """把 pdfplumber 的二维列表转成 Markdown 表格。"""
        rows = []
        for row in table:
            cells = []
            for cell in (row or []):
                # pdfplumber 用 None 表示空格子；单元格内的换行会让 Markdown 表格断行
                text = _CELL_CLEAN.sub(" ", (cell or "").strip())
                if len(text) > MAX_CELL_CHARS:
                    text = text[:MAX_CELL_CHARS] + "…"
                # 竖线会与 Markdown 的列分隔符冲突，替换成全角
                cells.append(text.replace("|", "｜"))
            if cells:
                rows.append(cells)

        if len(rows) < MIN_ROWS:
            return ""

        width = max(len(r) for r in rows)
        # 补齐每行列数，否则 Markdown 渲染时列会错位
        rows = [r + [""] * (width - len(r)) for r in rows]

        lines = ["| " + " | ".join(rows[0]) + " |",
                 "| " + " | ".join(["---"] * width) + " |"]
        for r in rows[1:]:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)

    @staticmethod
    def merge_into_text(text: str, tables: List[str]) -> tuple:
        """把表格并进正文，返回 (合并后的正文, 新增表格数)。"""
        if not tables:
            return text, 0
        merged = text.rstrip() + "\n\n【表格数据】\n" + "\n\n".join(tables)
        return merged, len(tables)


_table: Optional[TableService] = None


def get_table_service() -> TableService:
    global _table
    if _table is None:
        _table = TableService()
    return _table
