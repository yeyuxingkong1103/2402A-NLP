# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失工单
检索器：向量相似度 + 关键词混合打分 + ReRank，支持权重可调。
"""
import hashlib
import math

import config


def _qvec(text):
    grams = [text[i:i + 2] for i in range(len(text) - 1)]
    vec = {}
    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16) % 4096
        vec[h] = vec.get(h, 0) + 1
    return vec


def _cos(a, b):
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0) * v for k, v in b.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb + 1e-9)


def _keyword_score(text, query):
    kws = [k for k in query if len(k) > 1]
    return sum(1 for k in kws if k in text) / max(len(kws), 1)


def _char_grams(text):
    return [text[i:i + 2] for i in range(len(text) - 1)]


def retrieve(kb, query, top_k=None):
    top_k = top_k or config.TOP_K
    qv = _qvec(query)
    scores = []
    for cid, item in kb.items():
        v = _cos(qv, item["vector"])
        k = _keyword_score(item["text"], _char_grams(query) + query.split())
        # 向量相似度权重 + 关键词权重（可调）
        s = config.VECTOR_WEIGHT * v + config.TEXT_WEIGHT * k
        scores.append((s, cid, item))
    scores.sort(key=lambda x: -x[0])
    return scores[:top_k]


def rerank(results, query):
    """ReRank：对初排结果按关键词命中密度重排。"""
    if not config.RERANK_ENABLE:
        return results
    grams = _char_grams(query)
    ranked = []
    for s, cid, item in results:
        hits = sum(item["text"].count(g) for g in grams)
        density = hits / max(len(item["text"]), 1)
        ranked.append((s * 0.5 + density * 50, cid, item))
    ranked.sort(key=lambda x: -x[0])
    return ranked
