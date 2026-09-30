"""BGE-Reranker-v2-M3 重排序封装（本地模型，单例懒加载）。"""
import math
import threading
import time
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger("rag.reranker")


class Reranker:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or settings.reranker_model_path
        self.device = device or settings.reranker_device
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        """双检锁懒加载 CrossEncoder，加载失败自动降级 CPU。

        CrossEncoder（交叉编码器）把 query 与文档拼成一对输入模型，
        逐对打分，精度远高于向量内积，但无法离线预计算，只能在线跑，
        因此同样要单例 + 懒加载，并做 GPU→CPU 降级兜底。
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder
                    start = time.time()
                    logger.info("加载 BGE-Reranker-v2-M3：%s device=%s", self.model_path, self.device)
                    try:
                        self._model = CrossEncoder(self.model_path, device=self.device, max_length=512)
                    except Exception as exc:
                        logger.error("Reranker 在 %s 加载失败（%s），降级到 CPU", self.device, exc)
                        self._model = CrossEncoder(self.model_path, device="cpu", max_length=512)
                    logger.info("Reranker 加载完成，耗时 %.1fs", time.time() - start)
        return self._model

    def rerank(self, query: str, candidates: List[Dict[str, Any]],
               top_n: Optional[int] = None, text_field: str = "text") -> List[Dict[str, Any]]:
        """对候选片段精排，score 归一化到 (0,1)，返回按分数降序的列表。

        为什么做 sigmoid 归一化（1/(1+e^-x)）？交叉编码器的原始 logit 是无界的
        （可能负几到正几），无法直接跟阈值比较。sigmoid 把它压到 (0,1)，
        与检索阈值统一量纲，后续 retriever 才能用同一个 threshold 过滤。
        """
        top_n = top_n or settings.rerank_top_n
        if not candidates:
            return []
        model = self.load()
        pairs = [(query, c.get(text_field, "") or "") for c in candidates]
        start = time.time()
        scores = model.predict(pairs, batch_size=settings.reranker_batch_size, show_progress_bar=False)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if not isinstance(scores, list):
            scores = [scores]

        results = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            # sigmoid 归一化：把任意实数 logit 映射到 (0,1) 区间的"相关概率"
            item["rerank_score"] = round(1.0 / (1.0 + math.exp(-float(score))), 6)
            results.append(item)
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.info("重排序 %d 条候选，耗时 %.0fms", len(results), (time.time() - start) * 1000)
        return results[:top_n]

    @property
    def loaded(self) -> bool:
        return self._model is not None


# 模块级单例缓存 + 锁：与 embedder 同理，避免重复加载重排模型。
_reranker: Optional[Reranker] = None
_lock = threading.Lock()


def get_reranker() -> Reranker:
    """获取全局单例 Reranker（双检锁，线程安全）。"""
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = Reranker()
    return _reranker


def rerank(query: str, candidates: List[Dict[str, Any]], top_n: Optional[int] = None):
    """模块级便捷函数：直接对候选做精排，屏蔽单例获取细节。"""
    return get_reranker().rerank(query, candidates, top_n=top_n)