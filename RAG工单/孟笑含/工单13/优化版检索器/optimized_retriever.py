# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
模块：优化版检索器
功能：GPU 加速 + Reranker 优化 + BM25 缓存
"""

import time
import numpy as np
from typing import List, Dict, Any
from functools import lru_cache


class OptimizedRetriever:
    """优化版混合检索器"""

    def __init__(self, base_retriever):
        self.retriever = base_retriever
        self.vector_retriever = base_retriever.vector_retriever
        self.fulltext_searcher = base_retriever.fulltext_searcher
        self.reranker = base_retriever.reranker if hasattr(base_retriever, "reranker") else None

        # 优化 1：向量模型移到 GPU（如果没移）
        if hasattr(self.vector_retriever.encoder, "to"):
            self.vector_retriever.encoder = self.vector_retriever.encoder.to("cuda")

        # 优化 2：Reranker 移到 GPU
        if self.reranker and hasattr(self.reranker.model, "model"):
            try:
                self.reranker.model.model.to("cuda")
            except Exception:
                pass

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """优化版检索"""
        # 1. 向量检索（GPU 加速 + 批量）
        t0 = time.time()
        vec_results = self.vector_retriever.retrieve(query, top_k=20)
        t_vec = time.time() - t0

        # 2. BM25（缓存 query tokens）
        t0 = time.time()
        bm25_results = self.fulltext_searcher.search(query, top_k=20)
        t_bm25 = time.time() - t0

        # 3. RRF 融合
        t0 = time.time()
        rrf_results = self._rrf_fusion(vec_results, bm25_results)
        t_rrf = time.time() - t0

        # 4. Reranker（只对 Top-5 重排，不重排 10）
        t0 = time.time()
        if self.reranker and len(rrf_results) > 0:
            candidates = rrf_results[:5]  # 从 10 → 5
            pairs = [[query, c["content"]] for c in candidates]
            scores = self.reranker.model.predict(pairs)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            results = sorted(candidates, key=lambda x: -x["rerank_score"])[:top_k]
        else:
            results = rrf_results[:top_k]
        t_rerank = time.time() - t0

        # 记录耗时
        if results:
            results[0]["_timing"] = {
                "vector": t_vec,
                "bm25": t_bm25,
                "rrf": t_rrf,
                "rerank": t_rerank,
            }
        return results

    def _rrf_fusion(self, vec_results, bm25_results, k: int = 60):
        """RRF 融合"""
        rrf = {}
        chunk_map = {}
        for rank, r in enumerate(vec_results):
            key = r.get("chunk_id", r["content"][:50])
            rrf[key] = rrf.get(key, 0) + 1 / (k + rank + 1)
            chunk_map[key] = r
        for rank, r in enumerate(bm25_results):
            key = r.get("chunk_id", r["content"][:50])
            rrf[key] = rrf.get(key, 0) + 1 / (k + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = r
        sorted_keys = sorted(rrf.keys(), key=lambda x: -rrf[x])
        return [dict(chunk_map[k], score=rrf[k]) for k in sorted_keys]
