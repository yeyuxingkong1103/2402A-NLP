# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
PDF 解析与分块。
"""
import pymupdf


def load_pdf_text(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def chunk_text(text, chunk_size, overlap):
    text = text.replace("\x00", " ").strip()
    chunks = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = end - overlap
    return [c for c in chunks if c]
