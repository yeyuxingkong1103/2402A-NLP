# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""评估器：检索指标（Recall@K / MRR / nDCG@K）与相似度指标（正负例间隔）。

设计原则：
  - 指标实现全部纯 numpy，不使用 sklearn，便于单元测试逐值校验；
  - 相似度一律使用 **已 L2 归一化** 向量的内积（等价余弦相似度）；
  - 排名为 1-based；未命中记为 ``inf``（Recall/MRR/nDCG 均为 0）。
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


# ------------------------------------------------------------------ 基础指标
def ranks_from_scores(query_vecs: np.ndarray, doc_vecs: np.ndarray,
                      gold_indices: Sequence[int]) -> list[float]:
    """返回每个查询「首个正例文档」的 1-based 排名（未命中 = inf）。

    并列分数按文档下标升序（``argsort`` 默认稳定行为由实现显式保证）。
    """
    if len(query_vecs) != len(gold_indices):
        raise ValueError("查询数与金标准数不一致")
    out: list[float] = []
    for i, gi in enumerate(gold_indices):
        if gi is None or gi < 0:
            out.append(float("inf"))
            continue
        sims = doc_vecs @ query_vecs[i]
        order = np.argsort(-sims, kind="stable")
        pos = np.where(order == int(gi))[0]
        out.append(float(pos[0] + 1) if len(pos) else float("inf"))
    return out


def ranks_from_gold_sets(query_vecs: np.ndarray, doc_vecs: np.ndarray,
                         gold_index_sets: Sequence[Sequence[int]]) -> tuple[list[float], list[int]]:
    """多正例（多金标准）检索：返回每个查询的首个命中排名与其命中数。"""
    out: list[float] = []
    hit_counts: list[int] = []
    for i, gs in enumerate(gold_index_sets):
        gs = [int(g) for g in gs if g is not None and int(g) >= 0]
        if not gs:
            out.append(float("inf"))
            hit_counts.append(0)
            continue
        sims = doc_vecs @ query_vecs[i]
        order = np.argsort(-sims, kind="stable")
        rank_of = {int(d): r + 1 for r, d in enumerate(order)}
        best = min(rank_of.get(g, float("inf")) for g in gs)
        out.append(float(best))
        hit_counts.append(len(gs))
    return out, hit_counts


def recall_at_k(ranks: Iterable[float], k: int) -> float:
    rs = list(ranks)
    if not rs:
        return 0.0
    return round(sum(1 for r in rs if r <= k) / len(rs), 6)


def mrr(ranks: Iterable[float]) -> float:
    rs = list(ranks)
    if not rs:
        return 0.0
    return round(sum((1.0 / r if math.isfinite(r) else 0.0) for r in rs) / len(rs), 6)


def ndcg_at_k(ranks: Iterable[float], k: int) -> float:
    """单正例二值相关性下的 nDCG@k：DCG = 1/log2(rank+1)，IDCG = 1/log2(2) = 1。"""
    rs = list(ranks)
    if not rs:
        return 0.0
    total = 0.0
    for r in rs:
        if math.isfinite(r) and r <= k:
            total += 1.0 / math.log2(r + 1)
    return round(total / len(rs), 6)


def hit_rate_at_k(ranks: Iterable[float], k: int) -> float:
    return recall_at_k(ranks, k)


def retrieval_metrics(ranks: Sequence[float], ks: Sequence[int] = (1, 5, 10),
                      ndcg_k: int = 10) -> dict:
    m = {f"Recall@{k}": recall_at_k(ranks, k) for k in ks}
    m["MRR"] = mrr(ranks)
    m[f"nDCG@{ndcg_k}"] = ndcg_at_k(ranks, ndcg_k)
    finite = [r for r in ranks if math.isfinite(r)]
    m["命中数"] = len(finite)
    m["查询数"] = len(ranks)
    m["平均命中排名"] = round(sum(finite) / len(finite), 4) if finite else None
    m["中位命中排名"] = float(np.median(finite)) if finite else None
    return m


# ------------------------------------------------------------- 相似度指标
def similarity_stats(pos_sims: Sequence[float], neg_sims: Sequence[float]) -> dict:
    """正例对平均余弦相似度 vs 负例对平均余弦相似度（间隔越大越好）。"""
    p = np.asarray(list(pos_sims), dtype="float64")
    n = np.asarray(list(neg_sims), dtype="float64")
    out = {
        "正例对平均相似度": round(float(p.mean()), 6) if p.size else None,
        "负例对平均相似度": round(float(n.mean()), 6) if n.size else None,
        "相似度间隔": round(float(p.mean() - n.mean()), 6) if (p.size and n.size) else None,
        "正例对数": int(p.size),
        "负例对数": int(n.size),
    }
    if p.size:
        out["正例相似度标准差"] = round(float(p.std()), 6)
        out["正例相似度最小"] = round(float(p.min()), 6)
    if n.size:
        out["负例相似度标准差"] = round(float(n.std()), 6)
        out["负例相似度最大"] = round(float(n.max()), 6)
    if p.size and n.size:
        # 排序判别力：随机取一对正例与负例，正例分数更高的概率（AUC 形式，可解释为判别准确率）
        # 用采样避免 O(N*M) 爆炸
        rng = np.random.default_rng(20260826)
        a = rng.choice(p, size=min(2000, p.size), replace=True)
        b = rng.choice(n, size=min(2000, n.size), replace=True)
        out["正负例判别准确率"] = round(float((a[:, None] > b[None, :]).mean()), 6)
    return out


def bootstrap_ci(values: Sequence[float], *, n: int = 2000, alpha: float = 0.05,
                 seed: int = 20260826) -> tuple[float, float]:
    """自助法置信区间（用于说明小样本下的提升是否稳健）。"""
    v = np.asarray(list(values), dtype="float64")
    if v.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (round(float(lo), 6), round(float(hi), 6))


def summarize_binary_diff(before: Sequence[float], after: Sequence[float]) -> dict:
    """配对比较：微调前后「命中/未命中」的变化计数（用于显著性说明）。"""
    b = list(before)
    a = list(after)
    n = min(len(a), len(b))
    improved = sum(1 for i in range(n) if a[i] > b[i])
    worsened = sum(1 for i in range(n) if a[i] < b[i])
    same = n - improved - worsened
    return {"提升条数": improved, "下降条数": worsened, "不变条数": same, "可比条数": n}
