# -*- coding: utf-8 -*-
"""
文本分块模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
说明：按段落边界 + 滑动窗口切块，保证语义完整性，相邻块带重叠。
"""
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP


def _split_paragraphs(text):
    """按换行/句号切出自然段落"""
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    return paras


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """把单页文本切成多个语义块
    返回: [块文本, ...]
    """
    paras = _split_paragraphs(text)
    chunks, buf = [], ""

    for p in paras:
        # 单段超长时按句号再切
        if len(p) > chunk_size:
            sentences = re.split(r"(?<=[。；;！!？?])", p)
            for s in sentences:
                if len(buf) + len(s) > chunk_size and buf:
                    chunks.append(buf)
                    buf = buf[-overlap:] if overlap else ""  # 保留尾部重叠
                buf += s
        else:
            if len(buf) + len(p) + 1 > chunk_size and buf:
                chunks.append(buf)
                buf = buf[-overlap:] if overlap else ""
            buf += ("\n" if buf else "") + p

    if buf.strip():
        chunks.append(buf)
    # 过滤过短块（多为页眉页脚残留）
    return [c.strip() for c in chunks if len(c.strip()) >= 30]


def chunk_pages(pages, **kw):
    """把逐页文本列表分块，并带上页码元数据
    参数 pages: [{"page": int, "text": str}, ...]
    返回: [{"text": 块文本, "page": 页码}, ...]
    """
    out = []
    for p in pages:
        for c in chunk_text(p["text"], **kw):
            out.append({"text": c, "page": p["page"]})
    return out
