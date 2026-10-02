"""检索层门面：混合检索（稠密向量 + BM25）、RRF 融合、query 改写。

路由/流水线只会调 `HybridRetriever.retrieve(query, role_id, top_k)`；两路召回怎么分工、
分数怎么合并、索引缓存在哪，全部封装在这一层里，调用方不感知。
"""
from .hybrid_retriever import BM25Index, HybridRetriever, rrf_fusion, tokenize

__all__ = ["BM25Index", "HybridRetriever", "rrf_fusion", "tokenize"]
