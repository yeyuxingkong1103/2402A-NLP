# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
PDF 解析模块：
  1. 提取每页文字；
  2. 使用表格解析技术（PyMuPDF find_tables）识别页面中的表格，
     并把表格转换为 Markdown 文本，作为可检索的结构化内容；
  3. 将“段落文本”与“表格文本”统一分块，供检索使用。
"""
import re
import pymupdf


def table_to_markdown(rows):
    """把二维表格（list[list[str]]）转为 Markdown 文本。"""
    if not rows:
        return ""
    rows = [[(c or "").strip().replace("\n", " ") for c in r] for r in rows]
    ncol = max(len(r) for r in rows)
    lines = []
    for i, r in enumerate(rows):
        r = r + [""] * (ncol - len(r))
        lines.append("| " + " | ".join(r) + " |")
        if i == 0:
            lines.append("|" + "---|" * ncol)
    return "\n".join(lines)


def extract_pdf_content(pdf_path: str):
    """
    解析 PDF，返回 (text_blocks, table_blocks)。
    - text_blocks：每页纯文字内容
    - table_blocks：解析出的表格 Markdown 文本
    """
    doc = pymupdf.open(pdf_path)
    text_blocks = []
    table_blocks = []
    for pno, page in enumerate(doc):
        # 纯文字
        txt = page.get_text().strip()
        if txt:
            text_blocks.append(f"【第{pno + 1}页】\n{txt}")
        # 表格解析
        try:
            tabs = page.find_tables()
            if tabs.tables:
                for tb in tabs.tables:
                    md = table_to_markdown(tb.extract())
                    if md:
                        table_blocks.append(f"【第{pno + 1}页·表格】\n{md}")
        except Exception:
            # 某些页面不支持表格检测，忽略
            pass
    doc.close()
    return text_blocks, table_blocks


def chunk_text(text: str, chunk_size: int, overlap: int):
    text = text.replace("\x00", " ").strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = end - overlap
    return [c for c in chunks if c]
