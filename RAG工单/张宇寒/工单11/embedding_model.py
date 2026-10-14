from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class DomainEmbeddingModel:
    """轻量领域Embedding：冻结TF-IDF/SVD基础编码，只训练线性投影适配器。"""

    def __init__(self, dimension: int = 96, seed: int = 42):
        if dimension < 2:
            raise ValueError("dimension至少为2")
        self.requested_dimension = dimension
        self.seed = seed
        self.vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), min_df=1, max_features=2048, sublinear_tf=True
        )
        self.reducer: TruncatedSVD | None = None
        self.adapter: np.ndarray | None = None
        self.training_history: list[float] = []

    @property
    def dimension(self) -> int:
        if self.adapter is not None:
            return int(self.adapter.shape[0])
        return self.requested_dimension

    def fit_base(self, texts: list[str]) -> "DomainEmbeddingModel":
        cleaned = [str(text).strip() for text in texts if str(text).strip()]
        if len(cleaned) < 2:
            raise ValueError("基础模型至少需要两条非空文本")
        matrix = self.vectorizer.fit_transform(cleaned)
        components = min(self.requested_dimension, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if components < 2:
            raise ValueError("语料特征不足，无法建立Embedding模型")
        self.reducer = TruncatedSVD(n_components=components, random_state=self.seed)
        self.reducer.fit(matrix)
        self.adapter = np.eye(components, dtype=np.float64)
        return self

    def _base_encode(self, texts: list[str]) -> np.ndarray:
        if self.reducer is None or self.adapter is None:
            raise RuntimeError("模型尚未加载或训练")
        matrix = self.vectorizer.transform([str(text) for text in texts])
        return _normalize(self.reducer.transform(matrix).astype(np.float64))

    def encode(self, texts: list[str], adapted: bool = False) -> np.ndarray:
        base = self._base_encode(texts)
        if not adapted:
            return base
        return _normalize(base @ self.adapter)

    def fit_adapter(
        self,
        triplets: list[dict],
        epochs: int = 40,
        learning_rate: float = 0.03,
        margin: float = 0.5,
        regularization: float = 1e-4,
    ) -> list[float]:
        if self.adapter is None:
            raise RuntimeError("请先调用fit_base")
        if not triplets:
            raise ValueError("训练三元组为空")
        if epochs < 1 or learning_rate <= 0 or margin <= 0:
            raise ValueError("训练参数必须为正数")

        queries = self._base_encode([row["query"] for row in triplets])
        positives = self._base_encode([row["positive"] for row in triplets])
        negatives = self._base_encode([row["negative"] for row in triplets])
        identity = np.eye(self.adapter.shape[0], dtype=np.float64)
        history: list[float] = []

        for _ in range(epochs):
            positive_delta = queries - positives
            negative_delta = queries - negatives
            positive_projected = positive_delta @ self.adapter
            negative_projected = negative_delta @ self.adapter
            raw_losses = margin + np.sum(positive_projected**2, axis=1) - np.sum(negative_projected**2, axis=1)
            active = raw_losses > 0
            loss = float(np.maximum(raw_losses, 0).mean())
            history.append(round(loss, 8))
            if not np.any(active):
                continue
            pos = positive_delta[active]
            neg = negative_delta[active]
            gradient = 2 * (pos.T @ (pos @ self.adapter) - neg.T @ (neg @ self.adapter)) / len(triplets)
            gradient += regularization * (self.adapter - identity)
            norm = float(np.linalg.norm(gradient))
            if norm > 5:
                gradient *= 5 / norm
            self.adapter -= learning_rate * gradient

        self.training_history = history
        return history

    def save(self, path: Path) -> None:
        if self.reducer is None or self.adapter is None:
            raise RuntimeError("不能保存尚未训练的模型")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "DomainEmbeddingModel":
        if not path.is_file():
            raise FileNotFoundError(f"模型文件不存在：{path}")
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise ValueError("模型文件格式不正确")
        return model
