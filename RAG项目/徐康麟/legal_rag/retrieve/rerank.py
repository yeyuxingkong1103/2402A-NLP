# -*- coding: utf-8 -*-
"""重排：Reranker 接口 + BGE-rerank 真实后端 + 余弦相似度兜底。

另外实现「时间衰减」：按 chunk 的 created_at 给较新的知识加权，
对应设计文档里的「Rerank 阶段引入时间衰减权重，确保最新法律/医疗知识优先召回」。
"""
from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod

from ..config import RetrievalConfig
from ..schemas import SearchHit

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker(ABC):
    """重排接口：给候选打**更懂语义**的相关性分，再按分数截断。

    三个实现：``BgeReranker``（CrossEncoder 真实重排）、``CosineReranker``（零依赖兜底，
    直接用检索分数）、以及测试里的替身。子类只需要实现 :meth:`score`；
    :meth:`rerank` 统一负责"时间衰减 → 阈值过滤 → 排序 → 取 top_k"。

    ⚠️ 重排**失败不阻断主链路**：``score`` 抛异常时退回原始检索分数并打日志
    （宁可用次优顺序，也不要让用户拿不到答案）—— 这一点与"绝不静默失败"并不冲突：
    降级会明确写进日志。
    """

    #: 后端名（用于日志与 `/health`）。
    name: str = "base"

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()

    @abstractmethod
    def score(self, query: str, hits: list[SearchHit]) -> list[float]:
        """给每个 hit 打一个相关性分数（顺序与 ``hits`` 一一对应）。"""

    def rerank(self, query: str, hits: list[SearchHit], top_k: int | None = None) -> list[SearchHit]:
        """时间衰减 + 阈值过滤 + 降序取 ``top_k``（``top_k`` 缺省用 ``config.final_top_k``）。

        * ``rerank_score`` 保留**原始重排分**（可追溯），``score`` 才是加过衰减、
          用于排序与过滤的那个值；
        * 低于 ``config.score_threshold`` 的一律丢掉 —— 这是"没依据就不硬答"的第一道闸门。
        """
        if not hits:
            return []
        try:
            scores = self.score(query, hits)
        except Exception:  # noqa: BLE001 - 重排失败时降级为原始分数，不阻断主链路
            logger.exception("重排失败，降级为原始检索分数")
            scores = [hit.score for hit in hits]

        now = time.time()
        cfg = self.config
        for hit, raw in zip(hits, scores):
            value = float(raw)
            hit.rerank_score = value
            if cfg.time_decay > 0.0 and hit.chunk.created_at:
                age_days = max((now - hit.chunk.created_at) / 86400.0, 0.0)
                decay = math.pow(0.5, age_days / max(cfg.time_decay_half_life_days, 1e-6))
                value = value * ((1.0 - cfg.time_decay) + cfg.time_decay * decay)
            hit.score = value

        filtered = [h for h in hits if h.score >= cfg.score_threshold]
        filtered.sort(key=lambda h: h.score, reverse=True)
        limit = top_k if top_k is not None else cfg.final_top_k
        return filtered[:limit]


class CosineReranker(Reranker):
    """零依赖兜底：直接用检索阶段的余弦/融合分数。"""

    name = "cosine"

    def score(self, query: str, hits: list[SearchHit]) -> list[float]:
        """原样返回检索分数（**不做任何重排**，只为了让接口可用）。"""
        return [hit.score for hit in hits]


class BgeReranker(Reranker):
    """BGE-rerank 真实后端（CrossEncoder），权重走 ModelScope。"""

    name = "bge_rerank"

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL,
                 config: RetrievalConfig | None = None, batch_size: int = 8) -> None:
        super().__init__(config)
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "未安装 sentence-transformers，无法使用 BGE-rerank。\n"
                "请先安装重依赖：\n"
                "    .venv\\Scripts\\python.exe -m pip install -r requirements-full.txt\n"
                "或改用兜底重排：RERANK_PROVIDER=cosine"
            ) from exc

        from ..embedding.bge_m3 import resolve_model_path

        path = resolve_model_path(self.model_name)
        self._model = CrossEncoder(path)
        return self._model

    def score(self, query: str, hits: list[SearchHit]) -> list[float]:
        """用 CrossEncoder 逐对（query, 条文）打分；首次调用时懒加载权重。"""
        model = self._load()
        pairs = [(query, hit.chunk.text) for hit in hits]
        raw = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(x) for x in raw]


def build_reranker(provider: str, model: str = "",
                   config: RetrievalConfig | None = None) -> Reranker:
    """按名字构建重排后端：``cosine``（兜底，零依赖）/ ``bge_rerank``（CrossEncoder）。

    ⚠️ ⚠️ **口径警告**：``cosine`` 只是"用检索分数排序"，它**不是重排**；
    真实链路（生产）用 ``bge_rerank``。两者的读数不可互比（见 `docs/RERANK-EXPERIMENT.md`）。
    """
    key = (provider or "cosine").strip().lower()
    if key in ("cosine", "none", "fallback", "offline"):
        return CosineReranker(config)
    if key in ("bge_rerank", "bge-rerank", "bgererank", "cross_encoder"):
        return BgeReranker(model or DEFAULT_RERANK_MODEL, config)
    raise ValueError(f"未知的 rerank_provider: {provider!r}（可选 cosine | bge_rerank）")
