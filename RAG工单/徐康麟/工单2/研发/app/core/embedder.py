"""嵌入层：主路径 Ollama ``bge-m3``（1024 维、多语言），降级 sentence-transformers。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 嵌入优化（对应 设计/接口设计.md §2.5、环境事实 3.2）

环境事实：
- 本机 Ollama 0.35.0 在 ``127.0.0.1:11434`` 提供 ``bge-m3:latest``（**1024 维**，多语言）；
- 本机 torch 为 **CPU 版**（无 CUDA），因此本地降级模型 ``bge-small-zh-v1.5``（512 维）
  只在 Ollama 不可用时启用；
- **索引按嵌入模型分目录存放**（``<index>/<slug>-<dim>/``），不同维度绝不可混用。

纪律：两后端都不可用时**明确报错**（抛 ``RAGError``），绝不返回随机向量假装成功。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

import numpy as np

from app.core.config import get_settings
from app.core.errors import RAGError
from app.core.logging_conf import logger, trace

#: 进程内单条文本向量缓存上限（首字延迟治理：同一提问反复编码无意义）
_CACHE_MAX = 512


class BaseEmbedder(ABC):
    """嵌入后端抽象基类。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """模型名（用于索引目录 slug 与健康检查）。"""

    def is_semantic(self) -> bool:
        """是否为语义向量（False 表示哈希降级向量，仅用于极端离线场景）。"""
        return True

    @abstractmethod
    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """批量编码，返回 ``(n, dim)`` 的 float32 数组。"""

    def encode_one(self, text: str) -> np.ndarray:
        """编码单条文本，返回 ``(dim,)`` float32 数组。"""
        matrix = self.encode([text])
        return matrix[0]


class OllamaEmbedder(BaseEmbedder):
    """通过 Ollama ``/api/embed`` 调用 ``bge-m3:latest``（1024 维，多语言）。"""

    def __init__(self, model: str = "", base_url: str = "", timeout: float = 0.0) -> None:
        settings = get_settings()
        self._model = model or settings.embedding.ollama_model
        self._base_url = (base_url or settings.embedding.ollama_base_url).rstrip("/")
        self._timeout = timeout or settings.embedding.timeout
        self._dimension = settings.embedding.dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return self._model

    @trace
    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """批量编码（按批请求，单条失败必须抛错而不是静默返回零向量）。"""
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        settings = get_settings()
        size = batch_size or settings.embedding.ollama_batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), size):
            batch = [t if t.strip() else " " for t in texts[start : start + size]]
            payload = json.dumps({"model": self._model, "input": batch}).encode("utf-8")
            request = urllib.request.Request(
                f"{self._base_url}/api/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                logger.exception(
                    "app.core.embedder",
                    "Ollama 嵌入请求失败",
                    model=self._model,
                    base_url=self._base_url,
                    batch=len(batch),
                )
                raise RAGError(f"Ollama 嵌入失败: {exc}", detail={"model": self._model}) from exc
            embeddings = data.get("embeddings") or []
            if len(embeddings) != len(batch):
                raise RAGError(
                    "Ollama 嵌入返回条数与请求不一致",
                    detail={"requested": len(batch), "returned": len(embeddings)},
                )
            vectors.extend(embeddings)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self._dimension:
            raise RAGError(
                "嵌入维度与配置不一致",
                detail={"expected": self._dimension, "got": list(matrix.shape)},
            )
        if settings.embedding.normalize:
            matrix = _l2_normalize(matrix)
        return matrix

    @trace
    def health(self) -> dict[str, object]:
        """探测 Ollama 嵌入服务是否可用（超时 0.5 s，不重试）。"""
        settings = get_settings()
        started = time.perf_counter()
        try:
            request = urllib.request.Request(f"{self._base_url}/api/tags", method="GET")
            with urllib.request.urlopen(request, timeout=settings.llm.probe_timeout) as response:
                response.read()
            elapsed = (time.perf_counter() - started) * 1000
            return {"available": True, "probe_ms": round(elapsed, 2), "detail": ""}
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            logger.warning(
                "app.core.embedder",
                "Ollama 嵌入服务探测失败",
                base_url=self._base_url,
                elapsed_ms=round(elapsed, 2),
                error=str(exc),
            )
            return {"available": False, "probe_ms": round(elapsed, 2), "detail": str(exc)}


class SentenceTransformerEmbedder(BaseEmbedder):
    """降级路径：本地 sentence-transformers（``bge-small-zh-v1.5``，512 维）。"""

    def __init__(self, model_name: str = "") -> None:
        settings = get_settings()
        self._model_name = model_name or settings.embedding.resolve_st_model_path()
        self._dimension = settings.embedding.fallback_dimension
        self._model = None
        self._lock = threading.Lock()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return self._model_name

    def _ensure_model(self):
        """延迟加载模型（加载失败抛 RAGError，不静默）。"""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer  # 局部导入，避免启动即加载

                settings = get_settings()
                self._model = SentenceTransformer(self._model_name, device=settings.embedding.device)
                real_dim = int(self._model.get_sentence_embedding_dimension())
                if real_dim != self._dimension:
                    logger.warning(
                        "app.core.embedder",
                        "降级嵌入模型实际维度与配置不符，按实际维度使用",
                        configured=self._dimension,
                        actual=real_dim,
                    )
                    self._dimension = real_dim
                logger.info(
                    "app.core.embedder",
                    "降级嵌入模型加载完成",
                    model=self._model_name,
                    dimension=self._dimension,
                )
                return self._model
            except Exception as exc:
                logger.exception("app.core.embedder", "加载 sentence-transformers 模型失败", model=self._model_name)
                raise RAGError(f"加载降级嵌入模型失败: {exc}") from exc

    @trace
    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """批量编码（normalize_embeddings=True，与索引期一致）。"""
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        settings = get_settings()
        model = self._ensure_model()
        matrix = model.encode(
            [t if t.strip() else " " for t in texts],
            batch_size=batch_size or settings.embedding.batch_size,
            normalize_embeddings=settings.embedding.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(matrix, dtype=np.float32)


class Embedder:
    """嵌入门面：自动选择后端 + 单条缓存 + 健康信息 + 索引目录 slug。"""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._backend: BaseEmbedder | None = None
        self._degraded = False
        self._degrade_reason = ""
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _build_primary(self) -> BaseEmbedder:
        mode = self._settings.embedding.backend
        if mode in {"ollama", "auto"}:
            return OllamaEmbedder()
        return SentenceTransformerEmbedder()

    def _get_backend(self) -> BaseEmbedder:
        """获取后端；主后端不可用时降级到 sentence-transformers 并记 WARNING。"""
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is not None:
                return self._backend
            primary = self._build_primary()
            self._backend = primary
            return primary

    @property
    def degraded(self) -> bool:
        """当前是否处于降级状态（Ollama 不可用，改用本地 ST 模型）。"""
        return self._degraded

    @property
    def dimension(self) -> int:
        return self._get_backend().dimension

    @property
    def name(self) -> str:
        return self._get_backend().name

    @property
    def slug(self) -> str:
        """索引目录 slug，形如 ``bge-m3``、``bge-small-zh-v1-5``。"""
        raw = self.name.split("/")[-1].split(":")[0].lower()
        keep = [ch if (ch.isalnum() or ch == "-") else "-" for ch in raw]
        slug = "".join(keep).strip("-")
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug or "embedder"

    def index_dir_name(self) -> str:
        """索引子目录名：``<slug>-<dim>``（维度不可混用）。"""
        return f"{self.slug}-{self.dimension}"

    @trace
    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """批量编码；Ollama 失败时降级到 sentence-transformers（记录 degraded）。"""
        backend = self._get_backend()
        try:
            return backend.encode(texts, batch_size=batch_size)
        except Exception as primary_error:
            if isinstance(backend, SentenceTransformerEmbedder):
                logger.exception("app.core.embedder", "降级嵌入后端编码失败")
                raise
            logger.exception(
                "app.core.embedder",
                "主嵌入后端失败，降级到 sentence-transformers",
                backend=backend.name,
                fallback=self._settings.embedding.model_name,
            )
            fallback = SentenceTransformerEmbedder()
            self._backend = fallback
            self._degraded = True
            self._degrade_reason = str(primary_error)
            self._cache.clear()
            return fallback.encode(texts, batch_size=batch_size)

    def encode_one(self, text: str) -> np.ndarray:
        """编码单条文本（进程内缓存，重复提问零成本）。"""
        key = (text or "").strip()
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        vector = self.encode([key or " "])[0]
        with self._lock:
            if len(self._cache) >= _CACHE_MAX:
                self._cache.clear()
            self._cache[key] = vector
        return vector

    def warmup(self) -> dict[str, object]:
        """预热：编码一次固定文本，把模型加载耗时移出首问。"""
        started = time.perf_counter()
        try:
            self.encode(["预热"])
            elapsed = (time.perf_counter() - started) * 1000
            logger.info(
                "app.core.embedder",
                "嵌入预热完成",
                model=self.name,
                dimension=self.dimension,
                elapsed_ms=round(elapsed, 2),
            )
            return {"ok": True, "model": self.name, "dimension": self.dimension, "elapsed_ms": round(elapsed, 2)}
        except Exception as exc:
            logger.exception("app.core.embedder", "嵌入预热失败")
            return {"ok": False, "model": self.name, "dimension": 0, "elapsed_ms": 0.0, "error": str(exc)}

    def health(self) -> dict[str, object]:
        """健康信息：后端名 / 维度 / 是否降级 / 探测结果。"""
        info: dict[str, object] = {
            "backend": "ollama" if isinstance(self._get_backend(), OllamaEmbedder) else "sentence_transformers",
            "model": self.name,
            "dimension": self.dimension,
            "degraded": self._degraded,
            "degrade_reason": self._degrade_reason,
            "index_dir": self.index_dir_name(),
        }
        backend = self._get_backend()
        if isinstance(backend, OllamaEmbedder):
            info["probe"] = backend.health()
        return info


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化（零向量保持零，避免除零）。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度（对未归一化向量亦安全）。"""
    try:
        left = np.asarray(a, dtype=np.float32).ravel()
        right = np.asarray(b, dtype=np.float32).ravel()
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0:
            return 0.0
        return float(np.dot(left, right) / denominator)
    except Exception:
        logger.exception("app.core.embedder", "余弦相似度计算失败")
        return 0.0


_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    """获取进程级嵌入门面单例。"""
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = Embedder()
    return _embedder


def reset_embedder() -> None:
    """重置单例（测试用）。"""
    global _embedder
    with _embedder_lock:
        _embedder = None
