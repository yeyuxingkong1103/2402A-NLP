# -*- coding: utf-8 -*-
"""retrieval/fusion.py —— 加权 RRF 融合。

在链路中的位置：
    retrieve_with_trace 的第三步：把两路召回结果合并成一份统一排序的候选。

为什么必须用 RRF 而不是加权求和：
    向量路给的是余弦相似度（0~1），BM25 给的是无上界的对数分数，
    两者量纲完全不同，直接相加等于让 BM25 的数值范围单方面主导排序。
    RRF 只看排名不看分数，天然规避量纲问题。
"""
from __future__ import annotations

from typing import Any

from .config import KEYWORD_WEIGHT, RRF_K
from .corpus import load_corpus

def rrf_merge(vector_rank: list[int], keyword_rank: list[int]) -> list[int]:
    """加权 RRF 融合两路召回的排名。

    参数：
        vector_rank: 向量路结果的文档下标，按相关性从高到低
        keyword_rank: BM25 路结果的文档下标，按相关性从高到低
    返回：
        融合后按分数从高到低排序的文档下标列表。

    为什么必须用 RRF 而不是加权求和：
        向量路给的是余弦相似度（0~1），BM25 给的是无上界的对数分数，
        两者量纲完全不同，直接相加等于让 BM25 的数值范围单方面主导排序。
        RRF 只看排名不看分数，天然规避了量纲问题：
            RRF 分数 = 1 / (RRF_K + 排名)
        关键词路额外乘 KEYWORD_WEIGHT(2.5) 来体现"更信任字面匹配"。

    排序的次键取 pair[0]（下标）而不是任其随机：
        分数相同时按文档下标升序，保证同一查询的排序结果稳定、可复现。
    """
    scores: dict[int, float] = {}
    for rank, index in enumerate(vector_rank, 1):
        scores[index] = scores.get(index, 0) + 1 / (RRF_K + rank)
    for rank, index in enumerate(keyword_rank, 1):
        scores[index] = scores.get(index, 0) + KEYWORD_WEIGHT / (RRF_K + rank)
    return [index for index, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]

def fuse_hits(vector_hits: list[dict[str, Any]], keyword_hits_: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """融合两路召回结果，并补齐每条候选的分数明细。

    参数：
        vector_hits: dense_hits 的输出
        keyword_hits_: keyword_hits 的前 20 条（这里只是要它们的下标排名）
        limit: 融合后保留多少条（默认 15，即进入精排的候选数）
    返回：
        融合后的候选列表，每条含 rrf_rank / rrf_score / vec_score / bm25_score。

    分数明细是特意算出来给前端看的：
        "这一条为什么排前面"在答辩时要能指着界面说清楚 ——
        是两路都命中（rrf_score 高），还是向量路命中但 BM25 没中（只有 vec_score）。
        注意 vec_score / bm25_score 对某条候选可能是 None，表示该路没有召回它。
    """
    vector_rank = [hit["idx"] for hit in vector_hits]
    keyword_rank = [hit["idx"] for hit in keyword_hits_]
    merged = rrf_merge(vector_rank, keyword_rank)[:limit]
    vector_scores = {hit["idx"]: hit.get("vec_score") for hit in vector_hits}
    keyword_scores = {hit["idx"]: hit.get("bm25_score") for hit in keyword_hits_}
    corpus = load_corpus()[0]
    result = []
    for rank, index in enumerate(merged, 1):
        # 重新按名次算两路的贡献分量：rrf_merge 只返回了总分，这里需要拆开展示。
        # next(...) 在找不到时返回 0，表示该路未召回这条候选。
        vector_rank_score = next((1 / (RRF_K + position) for position, item in enumerate(vector_rank, 1) if item == index), 0)
        keyword_rank_score = next((KEYWORD_WEIGHT / (RRF_K + position) for position, item in enumerate(keyword_rank, 1) if item == index), 0)
        result.append({
            **corpus[index],
            "rrf_rank": rank,
            "rrf_score": round(vector_rank_score + keyword_rank_score, 6),
            "vec_score": vector_scores.get(index),
            "bm25_score": keyword_scores.get(index),
        })
    return result
