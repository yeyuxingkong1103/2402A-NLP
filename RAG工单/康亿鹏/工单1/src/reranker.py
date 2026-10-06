# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：重排模块。使用 BGE-Reranker 对向量召回的候选片段进行精排，
          提升送入大模型的上下文相关性（即“检索与生成”中的检索质量优化）。
"""
import numpy as np

import config
from src.utils import logger


def _resolve_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Reranker:
    """BGE-Reranker 封装（优先 FlagEmbedding，缺失时回退 sentence-transformers）。"""

    def __init__(self, model_path: str = None, batch_size: int = None, max_length: int = None):
        self.model_path = model_path or config.RERANKER_MODEL_PATH
        self.batch_size = batch_size or config.RERANKER_BATCH_SIZE
        self.max_length = max_length or config.RERANKER_MAX_LENGTH
        self.device = _resolve_device()
        self.backend = ""
        self._model = self._load_model()

    def _load_model(self):
        try:
            from FlagEmbedding import FlagReranker

            model = FlagReranker(
                self.model_path,
                use_fp16=self.device == "cuda" and config.RERANKER_USE_FP16,
                device=self.device,
            )
            self.backend = "FlagEmbedding"
            logger.info("BGE-Reranker 加载完成（FlagEmbedding），device=%s", self.device)
            return model
        except Exception as exc:
            logger.warning("FlagEmbedding 重排加载失败（%s），回退 sentence-transformers", exc)

        from sentence_transformers import CrossEncoder

        model = CrossEncoder(self.model_path, max_length=self.max_length, device=self.device)
        self.backend = "sentence-transformers"
        logger.info("BGE-Reranker 加载完成（CrossEncoder），device=%s", self.device)
        return model

    def _score(self, pairs: list) -> np.ndarray:
        if not pairs:
            return np.zeros((0,), dtype=np.float32)
        if self.backend == "FlagEmbedding":
            scores = self._model.compute_score(pairs, normalize=True, batch_size=self.batch_size)
        else:
            logits = self._model.predict(pairs, batch_size=self.batch_size)
            scores = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))  # sigmoid 归一化
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def rerank(self, query: str, hits: list, top_n: int = None) -> list:
        """对候选片段重排，返回按相关度降序的 top_n 条。

        Args:
            query: 用户问题（或消歧后的改写问题）
            hits : MilvusStore.search 返回的候选列表
            top_n: 保留条数
        """
        top_n = top_n or config.RERANK_TOP_N
        if not hits:
            return []

        pairs = [[query, hit["content"]] for hit in hits]
        scores = self._score(pairs)

        ranked = []
        for hit, score in zip(hits, scores):
            item = dict(hit)
            item["metadata"] = dict(hit.get("metadata", {}))
            item["metadata"]["rerank_score"] = float(score)
            ranked.append(item)
        ranked.sort(key=lambda x: x["metadata"]["rerank_score"], reverse=True)
        return ranked[:top_n]


_reranker_cache = {}


def get_reranker() -> Reranker:
    """获取重排模型单例。"""
    if "reranker" not in _reranker_cache:
        _reranker_cache["reranker"] = Reranker()
    return _reranker_cache["reranker"]
