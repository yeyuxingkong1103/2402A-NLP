# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
PDF 解析与分块优化：
  - chunk_fixed  ：基线方案（固定窗口，用于对比）
  - chunk_sentence：优化方案（按句号/换行切分后合并，保留语义完整段落）
"""
import re
import pymupdf


def load_pdf_text(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def chunk_fixed(text: str, chunk_size: int, overlap: int):
    """基线：固定窗口滑动切块。"""
    text = text.replace("\x00", " ").strip()
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


def _split_sentences(text: str):
    """按句末标点与换行切分为短句。"""
    return [s.strip() for s in re.split(r"[。！？；\n]+", text) if s.strip()]


def chunk_sentence(text: str, chunk_size: int, overlap: int):
    """
    优化：先切句，再按 chunk_size 合并成块；
    新块开头回退 overlap 字符，减少语义断裂，提升检索命中。
    """
    text = text.replace("\x00", " ").strip()
    sentences = _split_sentences(text)
    chunks = []
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) + 1 <= chunk_size:
            buf = (buf + "。" + sent).strip("。") if buf else sent
        else:
            if buf:
                chunks.append(buf)
            # 回退 overlap，保留上一块末尾上下文
            buf = buf[-overlap:] + "。" + sent if buf else sent
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c]
