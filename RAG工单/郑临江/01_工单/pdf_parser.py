# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
PDF 解析模块：读取 PDF 文本，并按指定大小切分为文本块。
"""
import pymupdf


def load_pdf_text(pdf_path: str) -> str:
    """解析 PDF，返回全文（按页拼接）。"""
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int, overlap: int):
    """
    将长文本按固定窗口切块，返回文本块列表。
    相邻块保留 overlap 字符的重叠，避免关键信息被截断。
    """
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
