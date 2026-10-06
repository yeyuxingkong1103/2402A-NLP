"""BGE-m3 向量化封装（懒加载）。"""
from __future__ import annotations

from app.config import settings
from app.logging_conf import log


class Embedder:
    """封装 BGE-m3 dense 向量。模型首次使用时加载，避免拖慢启动。"""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or settings.bge_m3_path
        self._model = None
        self._available: bool | None = None

    def _load(self):
        if self._model is not None:
            return self._model
        if self._available is False:
            raise RuntimeError("BGE-m3 模型不可用，请检查 BGE_M3_PATH 与 FlagEmbedding 安装")
        try:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(self.model_path, use_fp16=True)
            self._available = True
            log.info("BGE-m3 加载完成: %s", self.model_path)
        except Exception as exc:  # noqa: BLE001
            self._available = False
            log.error("BGE-m3 加载失败: %s", exc)
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

    def encode(self, texts: list[str], batch_size: int = 8, max_length: int = 1024) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(texts, batch_size=batch_size, max_length=max_length)["dense_vecs"]
        return [v.tolist() for v in vectors]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]
