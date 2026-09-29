"""BGE-M3 向量化封装（本地模型，单例懒加载，线程安全）。"""
import threading
import time
from typing import List, Optional

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger("rag.embedder")


class Embedder:
    """BGE-M3 dense embedding，输出 1024 维归一化向量（COSINE 检索）。

    归一化后向量长度为 1，两向量点积即等于余弦相似度，因此 Milvus 端
    用 COSINE 距离即可高效检索，无需在召回后再手动算余弦。
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or settings.embedding_model_path
        self.device = device or settings.embedding_device
        self._model = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return settings.embedding_dim

    def load(self):
        """懒加载模型，采用「双检锁」(double-checked locking)。

        为什么双检锁？模型加载很重（数 GB），必须全局只加载一次（单例）；
        但多线程可能同时触发 load。若直接加锁会每次都排队，性能差；
        双检锁先无锁判断一次（绝大多数请求直接命中已加载的模型，零开销），
        只有"尚未加载"时才抢锁，抢到锁后再查一次防止并发重复加载。
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    start = time.time()
                    logger.info("加载 BGE-M3：%s device=%s", self.model_path, self.device)
                    try:
                        self._model = SentenceTransformer(self.model_path, device=self.device)
                    except Exception as exc:  # GPU 不可用时降级到 CPU
                        # GPU 显存不足/无 CUDA 时加载会抛异常，此时降级 CPU 兜底，
                        # 宁可慢一点也要保证功能可用，避免整个服务起不来。
                        logger.error("BGE-M3 在 %s 加载失败（%s），降级到 CPU", self.device, exc)
                        self._model = SentenceTransformer(self.model_path, device="cpu")
                    logger.info("BGE-M3 加载完成，耗时 %.1fs，device=%s",
                                time.time() - start, self._model.device)
        return self._model

    def encode(self, texts: List[str], batch_size: Optional[int] = None,
               normalize: bool = True) -> List[List[float]]:
        """批量向量化，默认做 L2 归一化（配合 Milvus COSINE 检索）。"""
        if not texts:
            return []
        model = self.load()
        batch_size = batch_size or settings.embedding_batch_size
        start = time.time()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        logger.debug("向量化 %d 条文本，耗时 %.0fms", len(texts), (time.time() - start) * 1000)
        return vectors.tolist()

    def encode_query(self, text: str) -> List[float]:
        """查询向量（BGE-M3 无需额外指令前缀）。

        BGE-M3 针对检索做了训练，query 不需要像老式 BGE 那样加"为这个句子生成表示"前缀。
        """
        return self.encode([text])[0]

    @property
    def loaded(self) -> bool:
        return self._model is not None


# 模块级单例缓存：全局只保留一个 Embedder 实例，避免重复加载模型。
_embedder: Optional[Embedder] = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    """获取全局单例 Embedder（双检锁，线程安全）。

    与 load() 的双检锁思路一致：先无锁判断，再抢锁二次判断，
    保证并发环境下也只创建一个实例、只加载一次模型。
    """
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = Embedder()
    return _embedder


def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embedder().encode(texts)


def embed_query(text: str) -> List[float]:
    return get_embedder().encode_query(text)