# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-LightRAG优化
PDF 解析与文本切块。
"""
import pymupdf


def extract_pdf_text(path):
    doc = pymupdf.open(path)
    text = "\n".join(p.get_text() for p in doc)
    doc.close()
    return text


def chunk_text(text, size=500, overlap=80):
    text = text.replace("\x00", " ").strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c]


def load_documents(paths):
    docs = []
    for p in paths:
        text = extract_pdf_text(p)
        for i, c in enumerate(chunk_text(text)):
            docs.append({"id": f"{p}-{i}", "text": c, "source": p})
    return docs
