"""融合排序：加权 RRF（Reciprocal Rank Fusion）。

RRF 只看排名不看分数，天然规避稠密余弦、稀疏内积、BM25 三者量纲不一致的问题：

    score(d) = Σ_r  weight_r / (k + rank_r(d))

其中 ``rank_r(d)`` 从 1 开始；k 默认 60（与 Milvus 原生 RRFRanker 保持一致）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass(slots=True)
class FusedItem:
    """融合结果：总分 + 每一路的排名与原始分。"""

    key: str
    score: float
    routes: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "score": round(self.score, 6), "routes": self.routes}


def weighted_rrf(
    rankings: Mapping[str, Sequence[str]],
    weights: Mapping[str, float] | None = None,
    k: int = 60,
    scores: Mapping[str, Mapping[str, float]] | None = None,
) -> list[FusedItem]:
    """加权 RRF 融合。

    参数
    ----
    rankings : {路名: [chunk_id, ...]}，按各路相关性从高到低排序
    weights  : {路名: 权重}，缺省为 1.0；权重 <= 0 的路直接忽略
    k        : RRF 平滑常数
    scores   : {路名: {chunk_id: 原始分}}，仅用于回填调试信息
    """

    if k < 1:
        raise ValueError("RRF 的 k 必须 >= 1")
    weights = weights or {}
    scores = scores or {}
    fused: dict[str, FusedItem] = {}

    for route, ordered in rankings.items():
        weight = float(weights.get(route, 1.0))
        if weight <= 0:
            continue
        route_scores = scores.get(route, {})
        for position, key in enumerate(ordered, start=1):
            item = fused.get(key)
            if item is None:
                item = FusedItem(key=key, score=0.0)
                fused[key] = item
            item.score += weight / (k + position)
            item.routes[route] = {
                "rank": float(position),
                "raw": round(float(route_scores.get(key, 0.0)), 6),
                "weight": weight,
            }
    return sorted(fused.values(), key=lambda item: (-item.score, item.key))


def normalize_scores(values: Iterable[float]) -> list[float]:
    """Min-Max 归一化（用于分数融合模式）。"""

    items = list(values)
    if not items:
        return []
    low, high = min(items), max(items)
    if high - low < 1e-12:
        return [1.0 for _ in items]
    return [(value - low) / (high - low) for value in items]


def dedupe(keys: Sequence[str]) -> list[str]:
    """保持首次出现顺序去重。"""

    seen: set[str] = set()
    result: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result
