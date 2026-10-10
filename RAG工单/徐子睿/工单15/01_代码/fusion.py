# -*- coding: utf-8 -*-
"""
工单 15 · 混合检索融合：多路召回（原问题 / 图像增强查询）融合
提供两种融合算法：
  - rrf      : Reciprocal Rank Fusion，无分数尺度问题，推荐
  - weighted : 加权分数融合（与 RAGFlow 内部 hybrid_similarity 同思路）
"""
from __future__ import annotations

from typing import Dict, List, Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]],
                           k: int = 60,
                           weights: Sequence[float] | None = None) -> List[str]:
    """
    rankings: 多路召回的「有序 id 列表」（每路按相关性降序）
    k       : RRF 平滑常数（默认 60）
    weights : 每路的权重（默认等权）
    返回: 融合后的 id 列表（按融合分降序）
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    assert len(weights) == len(rankings)
    score: Dict[str, float] = {}
    for ranks, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranks):
            score[doc_id] = score.get(doc_id, 0.0) + w / (k + rank + 1)
    return [d for d, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)]


def weighted_fusion(scores: Sequence[Dict[str, float]],
                    weights: Sequence[float] | None = None,
                    normalize: bool = True) -> List[str]:
    """
    scores : 每路的 {doc_id: score}
    weights: 每路权重
    normalize: 各来源 min-max 归一化（应对 ES 分与向量分不同尺度）
    """
    if weights is None:
        weights = [1.0] * len(scores)
    fused: Dict[str, float] = {}
    for sc, w in zip(scores, weights):
        if not sc:
            continue
        if normalize:
            lo, hi = min(sc.values()), max(sc.values())
            rng = (hi - lo) or 1.0
        else:
            lo, rng = 0.0, 1.0
        for doc_id, s in sc.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + w * ((s - lo) / rng)
    return [d for d, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]


if __name__ == "__main__":
    route_text = ["a", "b", "c", "d"]
    route_image = ["c", "a", "e", "f"]
    print("RRF     :", reciprocal_rank_fusion([route_text, route_image]))
    print("weighted:", weighted_fusion(
        [{"a": 0.9, "b": 0.5, "c": 0.4, "d": 0.1},
         {"c": 0.95, "a": 0.7, "e": 0.5, "f": 0.2}]))
