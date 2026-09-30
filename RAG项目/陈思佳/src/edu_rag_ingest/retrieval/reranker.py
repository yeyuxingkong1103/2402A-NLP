from __future__ import annotations

"""可选的 CrossEncoder 二阶段重排序模块。"""

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class PairScorer(Protocol):
    def predict(self, pairs: list[list[str]], batch_size: int = 16) -> Any:
        ...


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool
    model_name: str
    device: str
    batch_size: int
    max_candidates: int


class CrossEncoderReranker:
    def __init__(self, config: RerankConfig) -> None:
        self.config = config
        self.model: PairScorer | None = None
        if not config.enabled or not config.model_name.strip():
            return
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(config.model_name, device=config.device)
        except Exception as error:  # pragma: no cover - 依赖和模型由运行环境提供
            logger.warning("无法加载 rerank 模型，将使用 RRF 排序：%s", error)

    def rerank(self, query: str, candidates: list[Any], top_k: int) -> list[Any]:
        if not candidates or not self.model or not query.strip():
            return candidates[:top_k]
        limited_candidates = candidates[: self.config.max_candidates]
        pairs = [[query, candidate.content] for candidate in limited_candidates]
        try:
            scores = self.model.predict(pairs, batch_size=self.config.batch_size)
            ranked = sorted(
                zip(scores, limited_candidates, strict=True),
                key=lambda item: float(item[0]),
                reverse=True,
            )
            return [candidate for _, candidate in ranked[:top_k]]
        except Exception as error:  # pragma: no cover - 模型推理由运行环境提供
            logger.warning("rerank 推理失败，将使用 RRF 排序：%s", error)
            return candidates[:top_k]
