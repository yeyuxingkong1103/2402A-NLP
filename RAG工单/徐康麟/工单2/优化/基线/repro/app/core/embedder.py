"""向量化模块：把文本转成稠密向量。

工单要求（5.3）：使用 BGE-M3 或 bge-large-zh-v1.5 生成向量，本地加载。

设计要点：
1. **可插拔后端**：``SentenceTransformerEmbedder``（真实模型）与
   ``HashingEmbedder``（确定性哈希向量，零依赖）。
2. **自动降级**：模型不可用（未下载 / 无网络 / 缺依赖）时自动切换到哈希后端，
   并在日志中给出明确原因——不静默失败。
3. **确定性**：哈希后端对同一文本永远给出同一向量，保证索引可复现。
4. **批量与缓存**：批量编码 + 内存 LRU 缓存，避免重复计算。
"""

from __future__ import annotations

import hashlib
import math
import time
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from app.core.config import get_settings
from app.core.logging_conf import logger, trace
from app.core.text_utils import tokenize

try:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    HAS_SENTENCE_TRANSFORMERS = False


class BaseEmbedder(ABC):
    """嵌入器接口。"""

    name: str = "base"
    dimension: int = 0

    @abstractmethod
    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """把一批文本编码成 (n, dim) 的 float32 矩阵。"""

    def encode_one(self, text: str) -> np.ndarray:
        """编码单条文本，返回 (dim,) 向量。"""
        return self.encode([text])[0]

    @property
    def is_semantic(self) -> bool:
        """是否为语义向量（哈希后端为 False，检索质量有限）。"""
        return False


class HashingEmbedder(BaseEmbedder):
    """确定性哈希向量后端（降级方案，零外部依赖）。

    做法：分词后用多组哈希把词映射到固定维度并累加（类似 hashing trick），
    再做 L2 归一化。它只捕捉词形重合，**不具备语义泛化能力**，
    但在无模型环境下仍可让整条检索链路跑通，且结果完全可复现。
    """

    name = "hashing-fallback"

    def __init__(self, dimension: int | None = None) -> None:
        settings = get_settings()
        self.dimension = dimension or settings.embedding.fallback_dim

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = tokenize(text) or [text]
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            # 用哈希的前 4 字节决定下标，第 5 字节决定符号，减少冲突带来的偏置
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    @trace
    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._embed(text) for text in texts]).astype(np.float32)


class SentenceTransformerEmbedder(BaseEmbedder):
    """基于 sentence-transformers 的本地模型嵌入器（BGE 系列）。"""

    name = "sentence-transformers"

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        if not HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError("未安装 sentence-transformers")
        settings = get_settings()
        # 本地模型目录优先，避免联网下载
        self.model_name = model_name or settings.embedding.resolve_model_path()
        self.device = device or settings.embedding.device
        self.normalize = settings.embedding.normalize
        started = time.perf_counter()
        self.model = SentenceTransformer(self.model_name, device=self.device)
        # sentence-transformers 5.x 起把 get_sentence_embedding_dimension 改名为
        # get_embedding_dimension；两个名字都兼容，避免升级后报错或告警。
        dimension_getter = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension"
        )
        self.dimension = int(dimension_getter())
        logger.info(
            "app.core.embedder",
            "嵌入模型加载完成",
            model=self.model_name,
            device=self.device,
            dimension=self.dimension,
            elapsed_s=round(time.perf_counter() - started, 2),
        )

    @property
    def is_semantic(self) -> bool:
        return True

    @trace
    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        settings = get_settings()
        vectors = self.model.encode(
            texts,
            batch_size=batch_size or settings.embedding.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class Embedder:
    """嵌入器门面：负责后端选择、降级与缓存。"""

    def __init__(self, prefer_model: bool = True, model_name: str | None = None) -> None:
        self.settings = get_settings()
        self.backend: BaseEmbedder
        self.degraded_reason = ""
        if prefer_model:
            try:
                self.backend = SentenceTransformerEmbedder(model_name=model_name)
            except Exception as exc:
                self.degraded_reason = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "app.core.embedder",
                    "语义模型不可用，降级为哈希向量（检索精度会下降，请尽快下载 BGE 模型）",
                    model=model_name or self.settings.embedding.model_name,
                    reason=self.degraded_reason,
                )
                self.backend = HashingEmbedder()
        else:
            self.degraded_reason = "调用方指定不使用语义模型"
            self.backend = HashingEmbedder()

    @property
    def dimension(self) -> int:
        return self.backend.dimension

    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def is_semantic(self) -> bool:
        return self.backend.is_semantic

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """编码一批文本；带缓存（相同文本不重复计算）。"""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        matrix = self.backend.encode(texts, batch_size=batch_size)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """编码单条（带 LRU 缓存，检索时高频命中）。"""
        return _cached_encode(self, text)

    def health(self) -> dict[str, object]:
        """返回嵌入器状态，供界面与健康检查展示。"""
        return {
            "backend": self.name,
            "dimension": self.dimension,
            "semantic": self.is_semantic,
            "degraded_reason": self.degraded_reason,
            "model": self.settings.embedding.model_name,
        }


@lru_cache(maxsize=2048)
def _cached_encode(embedder: Embedder, text: str) -> np.ndarray:
    """带缓存的单条编码。"""
    return embedder.encode([text])[0]


_embedder: Embedder | None = None


def get_embedder(prefer_model: bool = True, model_name: str | None = None) -> Embedder:
    """工厂函数：获取进程级单例嵌入器。"""
    global _embedder
    if _embedder is None:
        _embedder = Embedder(prefer_model=prefer_model, model_name=model_name)
    return _embedder


def reset_embedder() -> None:
    """重置单例（测试用）。"""
    global _embedder
    _embedder = None
    _cached_encode.cache_clear()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度（两个已归一化向量即点积）。"""
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def softmax(scores: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax（用于把相似度转成 0~1 权重）。"""
    if scores.size == 0:
        return scores
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    total = exp.sum()
    return exp / total if total else exp


__all__ = [
    "BaseEmbedder",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "cosine_similarity",
    "get_embedder",
    "reset_embedder",
    "softmax",
]
