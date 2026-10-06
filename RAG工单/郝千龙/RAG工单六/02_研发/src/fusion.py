# -*- coding: utf-8 -*-
# 【融合算法模块 · fusion.py】实现混合检索三种融合方式：加权平均、投票融合（Borda计数）、RRF倒数排名融合
# 工单编号：人工智能NLP-RAG-混合检索任务

"""融合层：对向量（稠密）召回列表与全文（多字段 BM25）召回列表做结果融合。

三种融合方式（工单技术要求“加权平均、投票机制等”，本系统实现三种可切换可对比）：
1. ``weighted_avg`` 加权平均：两路分数各自 min-max 归一化到 [0,1] 后，
   按热配权重 ``w_vec · s_vec + w_full · s_full`` 线性融合；
2. ``vote`` 投票融合（Borda 计数）：每路榜单按名次给出选票分
   ``(榜单长度 - 名次)``，两路累加，仅依赖名次、对分数量纲完全不敏感；
3. ``rrf`` 倒数排名融合（Reciprocal Rank Fusion）：
   ``score(d) = w_vec/(k+rank_vec(d)) + w_full/(k+rank_full(d))``，
   未上榜通道贡献 0。工业界最常用的无参（仅一个常数 k）稳健融合。
"""
from typing import Dict, List, Tuple

from config import CONFIG

HitList = List[Tuple[int, float]]  # [(chunk_index, score), ...]


def _minmax_normalize(hits: HitList) -> Dict[int, float]:
    """单路分数 min-max 归一化到 [0,1]（空表/同分退化处理）。

    :param hits: 单路召回 [(chunk_idx, raw_score)]
    :return: {chunk_idx: 归一化分}
    """
    if not hits:
        return {}
    scores = [s for _, s in hits]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span <= 1e-9:  # 全部同分时给统一中性分，避免一路垄断
        return {idx: 0.5 for idx, _ in hits}
    return {idx: (s - lo) / span for idx, s in hits}


def weighted_average(dense_hits: HitList, sparse_hits: HitList,
                     w_vec: float = 0.5, w_full: float = 0.5) -> Dict[int, float]:
    """加权平均融合。

    公式：``fuse(d) = w_vec·norm(s_vec(d)) + w_full·norm(s_full(d))``

    :param dense_hits: 向量路召回
    :param sparse_hits: 全文路召回
    :param w_vec: 向量路权重
    :param w_full: 全文路权重
    :return: {chunk_idx: 融合分}
    """
    norm_dense = _minmax_normalize(dense_hits)
    norm_sparse = _minmax_normalize(sparse_hits)
    fused: Dict[int, float] = {}
    for idx, score in norm_dense.items():
        fused[idx] = fused.get(idx, 0.0) + w_vec * score
    for idx, score in norm_sparse.items():
        fused[idx] = fused.get(idx, 0.0) + w_full * score
    return fused


def borda_vote(dense_hits: HitList, sparse_hits: HitList) -> Dict[int, float]:
    """投票融合（Borda 计数）：每张榜单按名次投票，榜单末尾得 1 分、榜首得 N 分。

    公式：``fuse(d) = Σ_list (N_list - rank_list(d) + 1)``

    :param dense_hits: 向量路召回（一张“选票”）
    :param sparse_hits: 全文路召回（一张“选票”）
    :return: {chunk_idx: 累计选票分}
    """
    fused: Dict[int, float] = {}
    for hits in (dense_hits, sparse_hits):
        ballot_size = len(hits)
        for rank, (idx, _) in enumerate(hits, start=1):
            fused[idx] = fused.get(idx, 0.0) + (ballot_size - rank + 1)
    return fused


def reciprocal_rank_fusion(dense_hits: HitList, sparse_hits: HitList,
                           k: int = None, w_vec: float = 0.5,
                           w_full: float = 0.5) -> Dict[int, float]:
    """RRF 倒数排名融合（支持双路权重热配）。

    公式：``fuse(d) = w_vec/(k+rank_vec(d)) + w_full/(k+rank_full(d))``

    :param dense_hits: 向量路召回
    :param sparse_hits: 全文路召回
    :param k: RRF 平滑常数（缺省取 CONFIG.rrf_k=60）
    :param w_vec: 向量路权重
    :param w_full: 全文路权重
    :return: {chunk_idx: RRF 融合分}
    """
    k = k if k is not None else CONFIG.rrf_k
    fused: Dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense_hits, start=1):
        fused[idx] = fused.get(idx, 0.0) + w_vec / (k + rank)
    for rank, (idx, _) in enumerate(sparse_hits, start=1):
        fused[idx] = fused.get(idx, 0.0) + w_full / (k + rank)
    return fused


# 融合方式名 -> 实现函数（retriever 统一调度）
FUSION_REGISTRY = {
    "weighted_avg": weighted_average,
    "vote": borda_vote,
    "rrf": reciprocal_rank_fusion,
}


def fuse(method: str, dense_hits: HitList, sparse_hits: HitList,
         w_vec: float = 0.5, w_full: float = 0.5) -> Dict[int, float]:
    """融合统一入口。

    :param method: weighted_avg / vote / rrf
    :param dense_hits: 向量路召回
    :param sparse_hits: 全文路召回
    :param w_vec: 向量路权重（vote 不使用）
    :param w_full: 全文路权重（vote 不使用）
    :return: {chunk_idx: 融合分}
    :raises ValueError: 未知融合方式时抛出
    """
    if method not in FUSION_REGISTRY:
        raise ValueError(f"未知融合方式: {method}，可选 {list(FUSION_REGISTRY)}")
    if method == "vote":
        return borda_vote(dense_hits, sparse_hits)
    if method == "weighted_avg":
        return weighted_average(dense_hits, sparse_hits, w_vec, w_full)
    return reciprocal_rank_fusion(dense_hits, sparse_hits,
                                  CONFIG.rrf_k, w_vec, w_full)
