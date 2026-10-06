# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
src/reranker.py —— 工单二重排序器

基于 BAAI/bge-reranker-v2-m3（本地 /home/dabaie/models/bge-reranker-v2-m3）对混合检索
候选做交叉编码精排；模型加载失败时自动降级为"保序透传"，保证检索链路永不中断。
"""
import os
import threading
from typing import Any, Dict, List, Optional

from loguru import logger

DEFAULT_RERANKER_PATH = os.getenv("RERANKER_PATH", "/home/dabaie/models/bge-reranker-v2-m3")

# 工单四（人工智能NLP-RAG-图像内容解析及检索优化）：进程级单例。
# TextRetrieverV3 与 TableRetriever 共用同一 FlagReranker，避免重复加载
# 模型（省一份显存，且消除"Reranker 加载完成"重复日志）。
_GLOBAL_RERANKER: Optional["Reranker"] = None
_RERANKER_LOCK = threading.Lock()


def get_reranker() -> "Reranker":
    """获取进程内唯一的 Reranker 单例（懒加载，加锁防并行检索线程重复加载）"""
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is None:
        with _RERANKER_LOCK:
            if _GLOBAL_RERANKER is None:
                _GLOBAL_RERANKER = Reranker()
    return _GLOBAL_RERANKER


class Reranker:
    """工单二重排序器（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""

    def __init__(self, model_path: Optional[str] = None, use_fp16: bool = True):
        self.model_path = model_path or DEFAULT_RERANKER_PATH
        self._model = None
        self._failed = False
        self._use_fp16 = use_fp16

    def _load(self):
        """懒加载 FlagReranker（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
        if self._model is not None or self._failed:
            return
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(self.model_path, use_fp16=self._use_fp16)
            logger.info(f"✅ Reranker 加载完成: {self.model_path}")
        except Exception as e:
            self._failed = True
            logger.warning(f"⚠️ Reranker 加载失败（降级为保序透传）: {e}")

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5,
               content_key: str = "content",
               max_chars: int = 1500) -> List[Dict[str, Any]]:
        """交叉编码重排：按 (query, content) 相关性降序，返回 top_k（含 rerank_score）

        工单四（人工智能NLP-RAG-图像内容解析及检索优化）：max_chars 控制送入
        cross-encoder 的文本长度（默认 512 token），文本检索链传 600 可将
        batch 序列压到 512 token 以内，热路径精排省约 30% 且 top8 排序稳定。
        """
        if not candidates:
            return []
        self._load()
        if self._model is None:  # 降级：保序返回前 top_k
            out = [dict(c, rerank_score=c.get("score", 0.0)) for c in candidates]
            return out[:top_k]
        pairs = [[query, c.get(content_key, "")[:max_chars]] for c in candidates]
        try:
            scores = self._model.compute_score(pairs, normalize=True)
            if isinstance(scores, (int, float)):  # 单条时 FlagEmbedding 返回标量
                scores = [scores]
        except Exception as e:
            logger.warning(f"Rerank 失败（降级保序）: {e}")
            out = [dict(c, rerank_score=c.get("score", 0.0)) for c in candidates]
            return out[:top_k]
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
