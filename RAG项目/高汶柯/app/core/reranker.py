"""BGE-reranker 精排封装（懒加载）。"""
from __future__ import annotations

from app.config import settings
from app.logging_conf import log


class Reranker:
    """封装 BGE-reranker 交叉编码器打分。"""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or settings.bge_reranker_path
        self._model = None
        self._available: bool | None = None

    def _load(self):
        if self._model is not None:
            return self._model
        if self._available is False:
            raise RuntimeError("BGE-reranker 模型不可用，请检查 BGE_RERANKER_PATH 与 FlagEmbedding 安装")
        try:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(self.model_path, use_fp16=True)
            self._available = True
            log.info("BGE-reranker 加载完成: %s", self.model_path)
        except Exception as exc:  # noqa: BLE001
            self._available = False
            log.error("BGE-reranker 加载失败: %s", exc)
            raise
        return self._model

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                self._load()
            except Exception:  # noqa: BLE001
                return False
        return bool(self._available)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        model = self._load()
        pairs = [[query, p] for p in passages]
        scores = model.compute_score(pairs)
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(s) for s in scores]
