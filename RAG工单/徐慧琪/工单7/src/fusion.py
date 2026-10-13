# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）

模块：混合检索结果融合（工单技术要求：加权平均、投票机制、RRF 等）

三路融合算法
------------
1. RRF（Reciprocal Rank Fusion，默认）
     score(d) = Σ_r  w_r / (k + rank_r(d))      k=60
   只用**名次**不用分数，天然规避"余弦相似度分数"与"BM25 分数"量纲不可比的问题；
   上游 01~05 的实测首选（工单05 事实命中率 0.9062 即建立在此融合之上）。

2. 加权平均（weighted）
     score(d) = Σ_r  w_r · norm_r(score_r(d))
   先把每路分数归一化（min-max 或 z-score）再加权求和。
   优点：分数差距被保留（RRF 会抹平"遥遥领先"与"险胜"的差别）；
   缺点：对归一化方式敏感，离群分数会压缩其余文档的动态范围。

3. 投票（vote / Borda count）
     每个通路给命中文档投票：第 i 名得 (N - i + 1) 分（N=该路候选数），
     再按权重加总。
   优点：抗单路噪声（一路极端高分不会左右结局），可解释性强；
   缺点：票的分辨率粗，名次相近的文档容易并列。

三者的选择由 config.FUSION_METHOD_DEFAULT / 界面「融合算法」下拉决定，
用户可实测对比（本工单报告给出三算法在 16 题上的指标对比）。
"""
from __future__ import annotations

from typing import Sequence


def doc_key(hit: dict) -> str:
    """融合键：必须带 source。

    两个文档的 chunk_id 各自从 c000001 开始（上游工单3 多文档踩坑修正），
    只用 chunk_id 会把"招股1 的第 N 块"和"招股2 的第 N 块"错误合并成一条。
    """
    cid = hit.get("chunk_id")
    if cid:
        return f"{hit.get('source', '')}#{cid}"
    return (f"{hit.get('source', '')}#p{hit.get('page_idx')}"
            f"-{hit.get('seq', 0)}")


def _ensure_container(merged: dict, key: str, hit: dict) -> dict:
    item = merged.get(key)
    if item is None:
        item = {"payload": dict(hit), "score": 0.0, "ranks": {}, "arms": []}
        merged[key] = item
    return item


def _stash_scores(item: dict, arm: str, hit: dict) -> None:
    """把各通路原始分数挂到 payload 上（供展示与调试）。"""
    payload = item["payload"]
    if arm == "dense" and "score" in hit:
        payload["vector_score"] = float(hit["score"])
    elif arm in ("fulltext", "sparse"):
        if "fulltext_score" in hit:
            payload["fulltext_score"] = float(hit["fulltext_score"])
        if "bm25_score" in hit:
            payload["bm25_score"] = float(hit["bm25_score"])
    for k, v in hit.items():
        payload.setdefault(k, v)


def rrf_fuse(ranked_lists: Sequence[tuple[str, Sequence[dict]]],
             k: int | None = None,
             weights: dict[str, float] | None = None) -> list[dict]:
    """RRF 融合：score(d) = Σ w_r / (k + rank_r(d))。"""
    from src import config

    k = k or config.RRF_K
    weights = weights or {}
    merged: dict[str, dict] = {}
    for arm, hits in ranked_lists:
        w = float(weights.get(arm, 1.0))
        if w <= 0:                 # 权重为 0 的通路整体不参与：不贡献分数，
            continue               # 也不该把只在它里面出现的文档带进结果集
        for rank, hit in enumerate(hits, start=1):
            item = _ensure_container(merged, doc_key(hit), hit)
            item["score"] += w / (k + rank)
            item["ranks"][arm] = rank
            if arm not in item["arms"]:
                item["arms"].append(arm)
            _stash_scores(item, arm, hit)

    out: list[dict] = []
    for item in merged.values():
        p = item["payload"]
        p["rrf_score"] = round(item["score"], 6)
        p["fused_score"] = p["rrf_score"]
        p["ranks"] = item["ranks"]
        p["arms"] = item["arms"]
        out.append(p)
    out.sort(key=lambda p: p["fused_score"], reverse=True)
    return out


def _normalize(values: list[float], method: str = "minmax") -> list[float]:
    if not values:
        return []
    if method == "zscore":
        import statistics
        mu = statistics.fmean(values)
        sd = statistics.pstdev(values) or 1.0
        return [(v - mu) / sd for v in values]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def weighted_fuse(ranked_lists: Sequence[tuple[str, Sequence[dict]]],
                  weights: dict[str, float] | None = None,
                  normalize: str = "minmax") -> list[dict]:
    """加权平均融合：各路分数先归一化，再按权重求和。"""
    weights = weights or {}
    merged: dict[str, dict] = {}
    for arm, hits in ranked_lists:
        if not hits:
            continue
        w = float(weights.get(arm, 1.0))
        if w <= 0:
            continue
        raw = [float(h.get("score", h.get("fulltext_score", 0.0)) or 0.0)
               for h in hits]
        norm = _normalize(raw, normalize)
        for hit, s in zip(hits, norm):
            item = _ensure_container(merged, doc_key(hit), hit)
            item["score"] += w * s
            item["ranks"].setdefault(arm, len(item["ranks"]) + 1)
            if arm not in item["arms"]:
                item["arms"].append(arm)
            _stash_scores(item, arm, hit)

    # 权重归一化：多路都命中时不应因"路数多"而分数更大
    total_w = 0.0
    for arm, _ in ranked_lists:
        total_w += float(weights.get(arm, 1.0))
    total_w = total_w or 1.0

    out: list[dict] = []
    for item in merged.values():
        p = item["payload"]
        p["fused_score"] = round(item["score"] / total_w, 6)
        p["ranks"] = item["ranks"]
        p["arms"] = item["arms"]
        out.append(p)
    out.sort(key=lambda p: p["fused_score"], reverse=True)
    return out


def vote_fuse(ranked_lists: Sequence[tuple[str, Sequence[dict]]],
              weights: dict[str, float] | None = None) -> list[dict]:
    """投票融合（Borda count）：第 i 名得 (N-i+1) 票，按路权重加总。"""
    weights = weights or {}
    merged: dict[str, dict] = {}
    for arm, hits in ranked_lists:
        n = len(hits)
        w = float(weights.get(arm, 1.0))
        if w <= 0:
            continue
        for rank, hit in enumerate(hits, start=1):
            item = _ensure_container(merged, doc_key(hit), hit)
            item["score"] += w * (n - rank + 1)
            item["ranks"][arm] = rank
            if arm not in item["arms"]:
                item["arms"].append(arm)
            _stash_scores(item, arm, hit)

    out: list[dict] = []
    for item in merged.values():
        p = item["payload"]
        p["votes"] = round(item["score"], 2)
        p["fused_score"] = p["votes"]
        p["ranks"] = item["ranks"]
        p["arms"] = item["arms"]
        out.append(p)
    out.sort(key=lambda p: (p["fused_score"], len(p["arms"])), reverse=True)
    return out


def fuse(method: str,
         ranked_lists: Sequence[tuple[str, Sequence[dict]]],
         weights: dict[str, float] | None = None,
         rrf_k: int | None = None) -> list[dict]:
    """融合统一入口：method ∈ {rrf, weighted, vote}。"""
    method = (method or "rrf").lower()
    if method == "weighted":
        return weighted_fuse(ranked_lists, weights)
    if method == "vote":
        return vote_fuse(ranked_lists, weights)
    return rrf_fuse(ranked_lists, k=rrf_k, weights=weights)
