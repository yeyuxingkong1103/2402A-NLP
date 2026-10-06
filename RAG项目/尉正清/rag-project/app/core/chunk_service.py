# app/core/chunk_service.py
"""语义分块：按话题转折切分长文本。

**为什么本项目选这一种**：知识库的绝大部分记录本身就是天然语义单元——
法条按「条」、问答按「一对」——不需要也不应该再切。唯独 PDF 导入的正文
是连续散文，无论按固定字数还是按段落切，都可能把一段完整论述拦腰截断。
语义分块在句子粒度上判断相邻句的语义相似度，在话题转折处下刀，
切出来的块内部更连贯。

做法（只借 numpy 做向量运算，不引入额外的分块库）：
    1. 按句号/分号等切成句子
    2. 逐句向量化，算相邻句余弦相似度
    3. 相似度低于阈值的相邻位置作为切点
    4. 再按最大长度约束合并，避免切得过碎

成本：每个句子一次 embedding。只在长文本上启用，短文本直接返回。
"""
import re
from typing import Callable, List

import numpy as np
import logging

logger = logging.getLogger(__name__)

# 句子边界：中文标点 + 换行
SENT_RE = re.compile(r"[^。！？；\n]+[。！？；]?")

# 低于该分位数（相似度最低的那部分）的位置才作为切点
BREAK_PERCENTILE = 25
# 单块最大字符数，超过就强制切
MAX_CHUNK_CHARS = 900
# 单块最小字符数，碎块并入前一块
MIN_CHUNK_CHARS = 150
# 句子太少的文本不值得做语义分析
MIN_SENTENCES = 6


def split_sentences(text: str) -> List[str]:
    """切成句子，保留标点。"""
    out = []
    for m in SENT_RE.finditer(text or ""):
        s = m.group(0).strip()
        if s:
            out.append(s)
    return out


def _cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def semantic_split(text: str, embed_fn: Callable[[List[str]], List[List[float]]],
                   max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """把长文本按语义转折切成若干块。

    embed_fn 接收句子列表，返回对应的向量列表（批量调用，避免逐句开销）。
    """
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = split_sentences(text)
    if len(sentences) < MIN_SENTENCES:
        return _pack(sentences, max_chars)

    try:
        vectors = embed_fn(sentences)
    except Exception as e:                                  # pragma: no cover
        logger.warning("语义分块向量化失败，退回按长度切分: %s", e)
        return _pack(sentences, max_chars)

    # 相邻句相似度
    sims = [_cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    if not sims:
        return _pack(sentences, max_chars)

    threshold = float(np.percentile(sims, BREAK_PERCENTILE))
    # 相似度低 = 话题转折 = 切点
    breaks = {i + 1 for i, s in enumerate(sims) if s <= threshold}
    breaks.add(0)

    chunks, cur = [], []
    for i, sent in enumerate(sentences):
        if i in breaks and cur:
            chunks.append("".join(cur))
            cur = []
        cur.append(sent)
    if cur:
        chunks.append("".join(cur))

    return _enforce(chunks, max_chars)


def _pack(sentences: List[str], max_chars: int) -> List[str]:
    """退化路径：只按长度合并，不判语义。"""
    chunks, cur, size = [], [], 0
    for s in sentences:
        if size + len(s) > max_chars and cur:
            chunks.append("".join(cur))
            cur, size = [], 0
        cur.append(s)
        size += len(s)
    if cur:
        chunks.append("".join(cur))
    return chunks


def _enforce(chunks: List[str], max_chars: int) -> List[str]:
    """收尾：过大的块再切，过小的块并入前一块。"""
    out = []
    for c in chunks:
        if len(c) <= max_chars:
            # 太小则并入前一块（前提是合并后不超限）
            if (out and len(c) < MIN_CHUNK_CHARS
                    and len(out[-1]) + len(c) <= max_chars):
                out[-1] += c
            else:
                out.append(c)
        else:
            # 切点仍不够细，按句子再兜底切一次
            out.extend(_pack(split_sentences(c), max_chars))
    return [c for c in out if c.strip()]
