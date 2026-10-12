# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
混合检索模块：融合向量检索与全文检索结果。
提供融合算法：加权平均、倒数排名融合（RRF）、投票机制。
"""
import numpy as np


def normalize(scores):
    """把得分归一化到 [0,1]。"""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def weighted_fusion(results, weights):
    """
    加权平均融合。
    results: [(doc, score), ...] 列表；weights: 与 results 等长。
    返回 [(doc, fused_score)]。
    """
    agg = {}
    for (docs_scores, w) in zip(results, weights):
        if not docs_scores:
            continue
        docs = [d for d, _ in docs_scores]
        scores = normalize([s for _, s in docs_scores])
        for d, s in zip(docs, scores):
            agg[d] = agg.get(d, 0.0) + w * s
    ranked = sorted(agg.items(), key=lambda x: -x[1])
    return ranked


def reciprocal_rank_fusion(results, k=60):
    """倒数排名融合（RRF）。"""
    agg = {}
    for docs_scores in results:
        for rank, (d, _) in enumerate(docs_scores):
            agg[d] = agg.get(d, 0.0) + 1.0 / (k + rank + 1)
    return sorted(agg.items(), key=lambda x: -x[1])


def voting_fusion(results):
    """投票机制：按各列表出现名次累积票数。"""
    agg = {}
    for docs_scores in results:
        for rank, (d, _) in enumerate(docs_scores):
            agg[d] = agg.get(d, 0.0) + (len(docs_scores) - rank)
    return sorted(agg.items(), key=lambda x: -x[1])


class HybridRetriever:
    """混合检索编排：向量 + 全文，权重可配置。"""

    def __init__(self, vector_retriever, fulltext_retriever,
                 vector_weight=0.6, fulltext_weight=0.4, fusion="weighted"):
        self.vec = vector_retriever
        self.ft = fulltext_retriever
        self.vector_weight = vector_weight
        self.fulltext_weight = fulltext_weight
        self.fusion = fusion

    def search(self, query, recall_top_k=10, top_k=3):
        vec_results = self.vec.recall(query, top_k=recall_top_k)
        ft_results = self.ft.search(query, top_k=recall_top_k)
        if self.fusion == "rrf":
            merged = reciprocal_rank_fusion([vec_results, ft_results])
        elif self.fusion == "voting":
            merged = voting_fusion([vec_results, ft_results])
        else:
            merged = weighted_fusion([vec_results, ft_results],
                                     [self.vector_weight, self.fulltext_weight])
        return [(doc, float(s)) for doc, s in merged[:top_k]]
