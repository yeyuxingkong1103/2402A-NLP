# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程工单
混合检索器：多路召回（文本查询 + 图像增强查询）+ 融合（加权 / RRF）。
"""
import hashlib
import math
import re

import config


def _grams(text):
    text = re.sub(r"\s+", "", text)
    return [text[i:i + 2] for i in range(len(text) - 1)]


def _vec(text):
    vec = {}
    for g in _grams(text):
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


def _text_score(block, query):
    qv = _vec(query)
    tv = _vec(block["text"] + " " + block.get("image_desc", ""))
    return _cos(qv, tv)


def vector_search(blocks, query, top_k=None):
    """单路召回：按文本/图像描述联合相似度。"""
    top_k = top_k or config.TOP_K * 2
    scored = [(_text_score(b, query), i, b) for i, b in enumerate(blocks)]
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


def reciprocal_rank_fusion(lists, k=60):
    """RRF 融合：多路召回结果按倒数排名融合。"""
    scores = {}
    for lst in lists:
        for rank, (_, i, b) in enumerate(lst):
            scores[i] = scores.get(i, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def weighted_fusion(list_text, list_image, weight):
    scores = {}
    for _, i, b in list_text:
        scores[i] = scores.get(i, 0) + (1 - weight)
    for _, i, b in list_image:
        scores[i] = scores.get(i, 0) + weight
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_retrieve(blocks, query, enhanced_query=None):
    """多路召回融合。"""
    list_text = vector_search(blocks, query)
    if enhanced_query and enhanced_query != query:
        list_image = vector_search(blocks, enhanced_query)
        if config.FUSION_METHOD == "rrf":
            fused = reciprocal_rank_fusion([list_text, list_image])
        else:
            fused = weighted_fusion(list_text, list_image, config.IMAGE_QUERY_WEIGHT)
        return [blocks[i] for i, _ in fused[:config.TOP_K]]
    return [b for _, i, b in list_text[:config.TOP_K]]
