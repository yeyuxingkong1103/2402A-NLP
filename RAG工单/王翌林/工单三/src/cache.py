# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
src/cache.py —— 工单二语义缓存（Redis 不可用时的本地双层缓存）

双层结构：
  L1 进程内 dict（热查询零开销）
  L2 diskcache 磁盘持久（进程重启后仍命中）
命中策略：查询向量与缓存向量余弦相似度 ≥ SEMANTIC_THRESHOLD(0.92) 即命中，
同义不同措辞的问题可复用答案，大幅降低重复问题延迟。
"""
import hashlib
import os
import threading
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

import diskcache

# ---------- 工单二语义缓存参数（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
SEMANTIC_THRESHOLD = 0.92   # 余弦相似度命中阈值
CACHE_TTL = 7 * 24 * 3600   # 缓存 7 天
CACHE_DIR = "data/retriever_cache/semantic_v2"
MAX_ENTRIES = 512           # 语义缓存最大条目数（防膨胀）

_instance: Optional["SemanticCache"] = None
_lock = threading.Lock()


def get_semantic_cache() -> "SemanticCache":
    """全局单例（线程安全，人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SemanticCache()
    return _instance


def _hash_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class SemanticCache:
    """工单二语义缓存（人工智能NLP-RAG-基于PDF文档的问答系统优化）：
    L1 内存 + L2 diskcache，向量相似度命中；Redis 可通过 REDIS_URL 环境变量切换（预留）"""

    def __init__(self, cache_dir: str = CACHE_DIR, threshold: float = SEMANTIC_THRESHOLD):
        self.threshold = threshold
        os.makedirs(cache_dir, exist_ok=True)
        self._store = diskcache.Cache(cache_dir, size_limit=int(2e9))
        self._hot: Dict[str, Dict[str, Any]] = {}  # L1: key -> {vec, payload}
        self._embedder = None
        self._lock = threading.Lock()
        logger.info(f"✅ 语义缓存就绪: {cache_dir} (threshold={threshold})")

    def _get_embedder(self):
        if self._embedder is None:
            from src.embedding import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _embed_query(self, query: str) -> np.ndarray:
        v = np.asarray(self._get_embedder().encode([query], show_progress_bar=False)[0],
                       dtype="float32")
        return v / (np.linalg.norm(v) + 1e-9)

    def _iter_entries(self):
        """遍历 L2 全部条目（key 由 _hash_key 生成）"""
        for key in self._store.iterkeys():
            entry = self._store.get(key)
            if entry:
                yield key, entry

    def lookup(self, query: str, lang: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """语义查找：相似度 ≥ 阈值即返回 {payload, cache_key, similarity, cache_hit}
        工单二：lang 隔离——不同语言空间分开，避免英文问题命中中文答案（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
        if not query:
            return None
        # L1 精确命中（零编码开销）
        hk = _hash_key(query.strip())
        with self._lock:
            if hk in self._hot:
                e = self._hot[hk]
                if lang is None or e.get("lang") == lang:
                    return dict(e["payload"], cache_hit=True, similarity=1.0, exact=True)
        qv = self._embed_query(query)
        best_key, best_sim = None, -1.0
        for key, entry in self._iter_entries():
            if lang is not None and entry.get("lang") != lang:
                continue  # 语言隔离过滤
            vec = np.asarray(entry.get("vec"), dtype="float32")
            if vec.shape != qv.shape:
                continue
            sim = float(vec @ qv)
            if sim > best_sim:
                best_key, best_sim = key, sim
        if best_key is not None and best_sim >= self.threshold:
            entry = self._store.get(best_key)
            with self._lock:  # 提升到 L1
                self._hot[best_key] = entry
                self._hot[hk] = entry
            logger.info(f"🟡 语义缓存命中 sim={best_sim:.3f}")
            return dict(entry["payload"], cache_hit=True, similarity=round(best_sim, 4))
        return None

    def put(self, query: str, payload: Dict[str, Any], lang: Optional[str] = None) -> str:
        """写入缓存（L1 + L2，带语言标记），payload 内含 references/answer 等"""
        hk = _hash_key(query.strip())
        entry = {"vec": self._embed_query(query).tolist(), "query": query,
                 "lang": lang, "payload": payload}
        with self._lock:
            self._hot[hk] = entry
        self._store.set(hk, entry, expire=CACHE_TTL)
        # 简易容量控制：超过上限清 L1 与最旧 L2（diskcache LRU 由 size_limit 兜底）
        if len(self._hot) > MAX_ENTRIES:
            self._hot.clear()
        return hk

    def clear(self):
        with self._lock:
            self._hot.clear()
        self._store.clear()

    def stats(self) -> Dict[str, Any]:
        return {"l1_entries": len(self._hot), "l2_entries": len(self._store),
                "threshold": self.threshold}
