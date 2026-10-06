# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
src/retrieval/fusion.py —— 工单六 混合检索结果融合算法

任务要求"提供混合检索结果的融合算法，如加权平均、投票机制等"：
  1. rrf_fuse：投票机制（Reciprocal Rank Fusion）——
     每路按排名贡献 1/(k+rank)，不依赖分数尺度，对向量/全文异质分数鲁棒
  2. weighted_fuse：加权平均 —— 两路分数各自 min-max 归一化后按权重线性组合
"""
from typing import Any, Dict, List, Tuple


def _key(h: Dict[str, Any]) -> str:
    return f"{h.get('doc_id','')}|{h.get('chunk_id','')}"


def rrf_fuse(vec_hits: List[Dict[str, Any]],
             ft_hits: List[Dict[str, Any]],
             rrf_k: int = 60,
             top_k: int = 28) -> List[Dict[str, Any]]:
    """工单六：RRF 投票融合（向量路 + 全文路）"""
    pool: Dict[str, Dict[str, Any]] = {}

    def _add(hits: List[Dict[str, Any]], tag: str):
        for rank, h in enumerate(hits):
            k = _key(h)
            contribution = 1.0 / (rrf_k + rank + 1)
            if k in pool:
                pool[k]["rrf_score"] += contribution
                pool[k]["search_path"] = "vector+fulltext"
            else:
                pool[k] = dict(h, rrf_score=contribution,
                               score=h.get("score", 0.0),
                               vec_score=h.get("score", 0.0) if tag == "vec" else 0.0,
                               ft_score=h.get("score", 0.0) if tag == "ft" else 0.0,
                               search_path=("vector" if tag == "vec" else "fulltext"))

    _add(vec_hits, "vec")
    _add(ft_hits, "ft")
    fused = sorted(pool.values(), key=lambda x: x["rrf_score"], reverse=True)
    return fused[:top_k]


def _norm(hits: List[Dict[str, Any]], score_key: str) -> Dict[str, float]:
    """工单六：单路 min-max 归一化（key → [0,1]）"""
    vals = [float(h.get(score_key, 0.0)) for h in hits]
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    out = {}
    for h in hits:
        v = float(h.get(score_key, 0.0))
        out[_key(h)] = (v - lo) / (hi - lo) if hi - lo > 1e-9 else 0.5
    return out


def weighted_fuse(vec_hits: List[Dict[str, Any]],
                  ft_hits: List[Dict[str, Any]],
                  vector_weight: float = 0.6,
                  fulltext_weight: float = 0.4,
                  top_k: int = 28) -> List[Dict[str, Any]]:
    """工单六：加权平均融合（两路归一化分数线性组合）"""
    vec_n = _norm(vec_hits, "score")
    ft_n = _norm(ft_hits, "score")
    pool: Dict[str, Dict[str, Any]] = {}
    for h in vec_hits:
        k = _key(h)
        pool[k] = dict(h, vec_norm=vec_n.get(k, 0.0), ft_norm=0.0,
                       weighted_score=vector_weight * vec_n.get(k, 0.0),
                       search_path="vector")
    for h in ft_hits:
        k = _key(h)
        if k in pool:
            pool[k]["ft_norm"] = ft_n.get(k, 0.0)
            pool[k]["weighted_score"] = (
                vector_weight * pool[k]["vec_norm"]
                + fulltext_weight * ft_n.get(k, 0.0))
            pool[k]["search_path"] = "vector+fulltext"
        else:
            pool[k] = dict(h, vec_norm=0.0, ft_norm=ft_n.get(k, 0.0),
                           weighted_score=fulltext_weight * ft_n.get(k, 0.0),
                           search_path="fulltext")
    fused = sorted(pool.values(), key=lambda x: x["weighted_score"],
                   reverse=True)
    return fused[:top_k]
