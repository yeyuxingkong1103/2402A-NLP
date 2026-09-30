# -*- coding: utf-8 -*-
"""多路检索结果融合模块。

实现 RRF（Reciprocal Rank Fusion，倒数排名融合）与加权融合两种策略，
按内容主键去重后输出 Top-K。融合不依赖各引擎的绝对分数，
只用排名，因此对不同量纲的引擎结果天然鲁棒。

设计：``BaseFusion`` 抽取公共骨架（遍历引擎 → 汇总得分 → 去重 → 排序），
子类只需实现 ``_score_item`` 定义单条结果的计分方式，消除重复。 
"""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Optional

from src import config
from src.core.retrieval.base.retriever_base import RetrievalResult


class BaseFusion(ABC):
    """融合策略公共骨架。"""

    name: str = "base"

    @abstractmethod
    def _score_item(self, source: str, rank: int, result: RetrievalResult) -> float:
        """计算单条结果对本路引擎的得分贡献。"""
        raise NotImplementedError

    def fuse(self, engine_results: dict[str, list[RetrievalResult]]) -> list[RetrievalResult]:
        """遍历各引擎结果，汇总得分、去重、排序、赋排名。

        - 先**引擎内去重**（同一 key 只保留首条），避免同名多规格 / 重复 chunk
          在同一引擎内自累加、重复贡献 RRF 分；
        - 再跨引擎聚合：同一 key 在多路出现时得分累加、保留首个结果对象；
        - 结果按总分降序排列，重新赋全局 rank。
        """
        scores: dict[str, float] = {}
        best: dict[str, RetrievalResult] = {}

        for source, results in engine_results.items():
            seen_in_engine: set[str] = set()
            for rank, result in enumerate(results, start=1):
                if result.key in seen_in_engine:
                    continue
                seen_in_engine.add(result.key)
                scores[result.key] = scores.get(result.key, 0.0) + \
                    self._score_item(source, rank, result)
                if result.key not in best:
                    best[result.key] = result

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        fused: list[RetrievalResult] = []
        for rank, (key, score) in enumerate(ranked, start=1):
            # 非破坏性：不修改入参 engine_results 里的原对象，避免污染下游复用。
            fused.append(dataclasses.replace(
                best[key], score=round(score, 6), rank=rank))
        return fused


class RRFFusion(BaseFusion):
    """倒数排名融合。

    公式：score(item) = Σ_i w_i / (k + rank_i(item))，其中 rank_i 为
    item 在第 i 路引擎结果中的排名（1 起），k 为平滑常数（默认 60），
    w_i 为该路引擎的来源权重（``source_weights``，默认 1.0）。

    ``source_weights`` 由检索路由（``RouteDecision.weights``）按信息需求类型提供，
    用于让主检索引擎占主导；不传时对所有来源等权，完全向后兼容。
    """

    name = "rrf"

    def __init__(self, k: int = 60, source_weights: Optional[dict[str, float]] = None):
        self.k = max(1, int(k))
        self.source_weights = source_weights or {}

    def _score_item(self, source: str, rank: int, result: RetrievalResult) -> float:
        weight = self.source_weights.get(source, 1.0)
        return weight / (self.k + rank)


class WeightedFusion(BaseFusion):
    """加权融合：先按各路引擎的最大分归一化到 [0,1]，再乘引擎权重。

    各引擎分数量纲不同（vector 常为余弦相似度，graph 为独立评分），
    直接 ``weight * score`` 可能让某一路高量纲引擎主导结果。
    因此默认对每路分数做 max 归一化，使各路贡献可比。
    """

    name = "weighted"

    def __init__(self, weights: Optional[dict[str, float]] = None,
                 normalize: bool = True):
        self.weights = dict(weights or config.FUSION_WEIGHTS)
        self.normalize = normalize
        self._max_by_source: dict[str, float] = {}

    def fuse(self, engine_results: dict[str, list[RetrievalResult]]) -> list[RetrievalResult]:
        # 预计算每路引擎的最大分，用于把跨引擎分数拉到可比量纲（[0,1]）。
        # max <= 0 时置 0，避免除零并防止负分放大。
        if self.normalize:
            self._max_by_source = {}
            for source, results in engine_results.items():
                source_max = 0.0
                for result in results:
                    score = float(result.score)
                    if score > source_max:
                        source_max = score
                self._max_by_source[source] = source_max
        return super().fuse(engine_results)

    def _score_item(self, source: str, rank: int, result: RetrievalResult) -> float:
        weight = self.weights.get(source, 1.0)
        score = float(result.score)
        if self.normalize:
            source_max = self._max_by_source.get(source, 0.0)
            score = score / source_max if source_max > 0.0 else 0.0
        return weight * score


def fusion(engine_results: dict[str, list[RetrievalResult]],
           strategy: str = "rrf",
           top_k: Optional[int] = None,
           rrf_k: Optional[int] = None,
           weights: Optional[dict[str, float]] = None,
           source_weights: Optional[dict[str, float]] = None) -> list[RetrievalResult]:
    """融合统一入口。

    Parameters
    ----------
    engine_results : dict[str, list[RetrievalResult]]
        引擎名 -> 检索结果列表。
    strategy : str
        rrf（默认）或 weighted。
    top_k : int | None
        输出条数上限；None 用 config.FUSION_TOP_K。
    rrf_k : int | None
        RRF 平滑常数；仅 rrf 策略生效。
    weights : dict[str, float] | None
        加权融合的引擎权重；仅 weighted 策略生效。
    source_weights : dict[str, float] | None
        RRF 的来源权重（每路引擎贡献乘该系数）；None 时全部为 1.0（原行为）。

    Returns
    -------
    list[RetrievalResult]
        按融合分降序、已去重的结果。
    """
    top = top_k or config.FUSION_TOP_K
    if strategy == "weighted":
        fus: BaseFusion = WeightedFusion(weights)
    else:
        fus = RRFFusion(
            rrf_k if rrf_k is not None else config.RRF_K,
            source_weights=source_weights,
        )

    return fus.fuse(engine_results)[:top]
