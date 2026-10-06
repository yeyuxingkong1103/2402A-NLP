# -*- coding: utf-8 -*-
"""PDF 文档解析模块
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

功能：
- 提取 PDF 页面文字内容；
- 提取页内表格（金融招股说明书的核心财务数据多以表格呈现）；
- 按“页”组织文档，供知识库切片使用；
- 具备文档解析失败等场景的容错处理。

实现说明：
本项目依赖本地 langchain2 环境中的 PyMuPDF (fitz)，可同时解析文字与表格，
避免把财务表格数据遗漏在向量化的内容之外。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PageDoc:
    """一页 PDF 的解析结果。"""

    page_no: int                      # 页码（从 1 开始）
    text: str = ""                    # 该页纯文本（含表格渲染文本）
    tables: List[str] = field(default_factory=list)  # 该页表格（Markdown 风格文本）

    @property
    def content(self) -> str:
        """用于切片的完整内容 = 正文文字 + 表格文本。"""
        parts = [self.text.strip()]
        for t in self.tables:
            parts.append(f"[表格]\n{t}")
        return "\n".join(p for p in parts if p)


def _render_table(table) -> str:
    """把 pymupdf 返回的表格对象渲染为 Markdown 表格文本。"""
    try:
        if not table:
            return ""
        rows = table.extract()
        if not rows:
            return ""
        lines = []
        for i, row in enumerate(rows):
            cells = [str(c).replace("\n", " ").strip() if c else "" for c in row]
            lines.append("| " + " | ".join(cells) + " |")
        if lines:
            # 表头后插入对齐分隔行
            n = len(rows[0])
            lines.insert(1, "|" + "---|" * n)
        return "\n".join(lines)
    except Exception as exc:  # 表格渲染失败不影响整体解析
        logger.debug("table render failed: %s", exc)
        return ""


def parse_pdf(
    pdf_path: str | Path, with_tables: bool = True
) -> List[PageDoc]:
    """解析 PDF 文件，返回按页组织的文档列表。

    Args:
        pdf_path: PDF 文件路径。
        with_tables: 是否提取页内表格。

    Returns:
        页文档列表；解析失败时抛出异常（由调用方容错处理）。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {path}")

    try:
        import pymupdf as fitz  # PyMuPDF 新包名
    except ImportError:
        import fitz  # 兼容旧包名

    pages: List[PageDoc] = []
    with fitz.open(str(path)) as doc:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            page_no = i + 1
            try:
                text = page.get_text("text", sort=True) or ""
            except Exception:  # 单页文字解析失败，保留空内容继续
                text = ""

            tables: List[str] = []
            if with_tables:
                try:
                    found = page.find_tables() or []
                    for tb in found:
                        rendered = _render_table(tb)
                        if rendered:
                            tables.append(rendered)
                except Exception as exc:
                    logger.debug("page %s table parse failed: %s", page_no, exc)

            pages.append(PageDoc(page_no=page_no, text=text, tables=tables))
    return pages


def parse_to_text(pdf_path: str | Path) -> str:
    """解析 PDF 并返回全部纯文本（便于人工检索与人工核对答案）。"""
    pages = parse_pdf(pdf_path, with_tables=True)
    return "\n\n".join(f"—— 第 {p.page_no} 页 ——\n{p.content}" for p in pages)


if __file__ and "__main__" in __name__:
    logging.basicConfig(level=logging.INFO)
    import sys

    from src.config import PDF_PATH

    pages = parse_pdf(PDF_PATH)
    print(f"总页数: {len(pages)}")
    print(f"总字符: {sum(len(p.content) for p in pages)}")
    print("首两页预览:")
    for p in pages[:2]:
        print(p.content[:600])
        print("=" * 40)