# -*- coding: utf-8 -*-
"""retrieval/recall.py —— 两路召回。

在链路中的位置：
    retrieve_with_trace 的第二步：分别从向量路和关键词路召回候选。

两路互补之处：
    dense_hits   向量路管语义相近 —— "回充设备" 与 "气体重新充入装置" 能对上
    keyword_hits BM25 路管字面精确 —— "GB/T 44653-2024"、"净化处理装置" 靠它命中
两路结果交给 retrieval/fusion.py 做加权 RRF 融合。
"""
from __future__ import annotations

from typing import Any

try:
    from ..pipeline import COLLECTION, embed
    from ..vector_store import search_vectors
except ImportError:
    from pipeline import COLLECTION, embed
    from vector_store import search_vectors

from .config import KEYWORD_K, VECTOR_K, VECTOR_THRESHOLD
from .corpus import bm25_index, load_corpus

def dense_hits(query: str) -> list[dict[str, Any]]:
    """向量路召回：语义相似度召回 top20。

    参数：
        query: 改写后的检索串
    返回：
        候选片段列表，每项附带 vec_score（余弦相似度，保留 4 位）。

    两个关键点：
        1. 用改写后的查询而不是原问题 —— 把口语词换成文档术语，向量也会更靠近目标区块
        2. score_threshold=VECTOR_THRESHOLD 在检索层就过滤低分候选，
           让噪声不进入后续的融合与精排（省时间，也避免它们干扰精排判断）
    """
    corpus, index_map = load_corpus()
    hits = search_vectors(
        COLLECTION,
        embed([query])[0],  # 查询也要用同一个 bge-m3 编码，两侧向量空间才一致
        limit=VECTOR_K,
        score_threshold=VECTOR_THRESHOLD,
    )
    results = []
    for hit in hits:
        payload = hit["payload"]
        # 把 Milvus 返回的记录映射回语料下标，这样两路结果才能融合
        key = (str(payload.get("source", "")), int(payload.get("page", -1)), payload.get("section", ""), payload.get("text", ""))
        index = index_map.get(key)
        if index is not None:
            results.append({**corpus[index], "vec_score": round(float(hit["score"]), 4)})
    return results

def keyword_hits(query: str) -> tuple[list[dict[str, Any]], list[tuple[int, float]]]:
    """BM25 路召回：对全语料打分取 top20。

    参数：
        query: 改写后的检索串
    返回：
        (前 KEYWORD_K 条候选, 全部文档的 (下标, 原始分) 降序列表)
        第二个返回值用于统计"有多少条分数 >= 1"，写进 trace 让链路可观测。
    """
    corpus = load_corpus()[0]
    scores = sorted(((index, bm25_index().score(query, index)) for index in range(len(corpus))), key=lambda pair: -pair[1])
    return [{**corpus[index], "bm25_score": round(score, 3)} for index, score in scores[:KEYWORD_K]], scores
