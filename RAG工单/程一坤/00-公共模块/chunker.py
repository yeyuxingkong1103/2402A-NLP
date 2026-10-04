# -*- coding: utf-8 -*-
"""
文本分块模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
说明：按段落边界 + 滑动窗口切块，保证语义完整性，相邻块带重叠。
"""
import re  # 正则切分段落与句子
from config import CHUNK_SIZE, CHUNK_OVERLAP  # 默认块长 500 / 重叠 80，见 config.py


def _split_paragraphs(text):
    """按换行切出自然段落"""
    # 按 1 个及以上换行符切分；strip() 去首尾空白，滤掉空串（连续换行产生的空段）
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    return paras


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """把单页文本切成多个语义块
    返回: [块文本, ...]
    """
    paras = _split_paragraphs(text)
    chunks, buf = [], ""  # buf 是当前累积中的块缓冲区

    for p in paras:
        # 单段超长时按句号再切
        if len(p) > chunk_size:
            # 用零宽断言 (?<=...) 在中英文句末标点之后切分，标点保留在句尾不丢
            sentences = re.split(r"(?<=[。；;！!？?])", p)
            for s in sentences:
                # 加上新句会超长且 buf 非空时：先落盘当前块
                if len(buf) + len(s) > chunk_size and buf:
                    chunks.append(buf)
                    buf = buf[-overlap:] if overlap else ""  # 保留尾部重叠
                buf += s
        else:
            # 普通段：+1 是段间换行符的长度预算
            if len(buf) + len(p) + 1 > chunk_size and buf:
                chunks.append(buf)
                buf = buf[-overlap:] if overlap else ""
            # 非首段前补换行，保持段落在块内的边界清晰
            buf += ("\n" if buf else "") + p

    # 收尾：循环结束后 buf 里剩的最后一块（可能不满 chunk_size）也要入库
    if buf.strip():
        chunks.append(buf)
    # 过滤过短块（多为页眉页脚残留）
    # 阈值 30 字符：太短的块无检索价值，反而拉低向量检索信噪比
    return [c.strip() for c in chunks if len(c.strip()) >= 30]


def chunk_pages(pages, **kw):
    """把逐页文本列表分块，并带上页码元数据
    参数 pages: [{"page": int, "text": str}, ...]
    返回: [{"text": 块文本, "page": 页码}, ...]
    """
    out = []
    for p in pages:
        # **kw 透传 chunk_size/overlap 等参数给 chunk_text
        for c in chunk_text(p["text"], **kw):
            # 每个块记录来源页码，供回答时标注"（来源：第X页）"
            out.append({"text": c, "page": p["page"]})
    return out
