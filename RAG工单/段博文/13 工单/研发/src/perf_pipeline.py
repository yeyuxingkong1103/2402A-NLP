# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
"""
检索流水线（分阶段计时版）：基线实现 与 优化实现。

与 persona_rag 线上 retriever.py 的关系：
    baseline_search 是 hybrid_search 的"等语义展开"——同样的向量+BM25 双路召回、
    同样的 RRF 融合（权重 0.5/0.5，c=60）、同样的 CrossEncoder 重排与 0.3 阈值过滤，
    只是把每个阶段拆开单独计时，用于定位瓶颈。

    optimized_search 在保持召回质量参数（最终 top_k、阈值）不变的前提下做四项优化：
      O1 查询只编码一次，向量检索走 similarity_search_by_vector（基线 LangChain 链路
         内部对查询的编码不可复用，长期记忆等多路召回会重复编码）；
      O2 向量路与 BM25 路并行执行（基线 EnsembleRetriever 是串行 invoke）；
      O3 RRF 融合后先去重再截断候选（默认只送前 6 条进重排，基线把十几条候选全部
         送进 CrossEncoder——CPU 上每对 (query, doc) 都是一次完整前向推理）；
      O4 重排输入截断到 RERANK_MAX_CHARS 字符（chunk 约 500 字，前 256 字已足够
         判断相关性，CrossEncoder 耗时与输入长度近似成正比）。
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from config import (
    TOP_K_RECALL, TOP_K_RERANK, SCORE_THRESHOLD,
    MILVUS_COLLECTIONS,
)
from db_milvus import get_embedding, get_vectorstore, fetch_all_text
from logger import get_logger

logger = get_logger(__name__)

# ==================== 可复用组件缓存 ====================
# BM25 索引与 retriever.py 相同：启动时建一次，进程内复用
from langchain_community.retrievers import BM25Retriever
import jieba

_bm25_cache: Dict[str, BM25Retriever] = {}
_vectorstores: Dict[str, object] = {}


def _chinese_tokenize(text: str) -> List[str]:
    return [w.strip().lower() for w in jieba.lcut(text) if w.strip()]


def warmup(collection_name: str):
    """启动期预热：向量模型、向量库连接、重排模型、BM25 索引（不计入单次检索耗时）"""
    t0 = time.perf_counter()
    get_embedding()
    get_vectorstore(collection_name)
    get_reranker()
    docs = fetch_all_text(collection_name)
    _bm25_cache[collection_name] = BM25Retriever.from_documents(
        documents=docs, k=TOP_K_RECALL, preprocess_func=_chinese_tokenize)
    logger.info(f"[预热] 集合 {collection_name}：BM25 语料 {len(docs)} 条，"
                f"总耗时 {time.perf_counter()-t0:.2f}s")


def get_reranker():
    """重排模型单例（bge-reranker-large，CPU）"""
    global _reranker
    if _reranker is None:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from config import RERANK_MODEL, RERANK_DEVICE
        _reranker = HuggingFaceCrossEncoder(
            model_name=RERANK_MODEL, model_kwargs={"device": RERANK_DEVICE})
    return _reranker


_reranker = None


# ==================== 公共子步骤 ====================

def _rrf_fuse(result_lists: List[List[Document]], weights=(0.5, 0.5), c=60) -> List[Document]:
    """加权 RRF 融合（与 langchain EnsembleRetriever 同语义），按内容去重"""
    scores: Dict[str, float] = {}
    keeper: Dict[str, Document] = {}
    for docs, w in zip(result_lists, weights):
        for rank, doc in enumerate(docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + w / (rank + c)
            keeper.setdefault(key, doc)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [keeper[k] for k, _ in ordered]


def _rerank(query: str, candidates: List[Document], top_n: int,
            max_chars: int = None) -> List[Document]:
    """CrossEncoder 重排：对 (query, doc) 对打分，附 metadata['score'] 后取前 top_n"""
    if not candidates:
        return []
    pairs = [(query, d.page_content[:max_chars] if max_chars else d.page_content)
             for d in candidates]
    scores = get_reranker().score(pairs)
    for d, s in zip(candidates, scores):
        d.metadata["score"] = float(s)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]


def _threshold_filter(docs: List[Document], threshold: float) -> List[Document]:
    return [d for d in docs if d.metadata.get("score", 1.0) >= threshold]


def _assemble(docs: List[Document], query: str) -> str:
    """上下文组装（对应 prompt_templates.build_context 的精简版）"""
    blocks = [f"[资料{i+1}] {d.page_content}" for i, d in enumerate(docs)]
    return "\n\n".join(blocks) + f"\n\n问题：{query}" if blocks else query


# ==================== 基线（原实现语义） ====================

def baseline_search(query: str, collection_name: str,
                    top_k: int = TOP_K_RERANK) -> Tuple[List[Document], Dict[str, float]]:
    """基线检索：复刻 retriever.hybrid_search 的串行流程，分阶段计时"""
    t: Dict[str, float] = {}

    # 阶段1 查询处理（向量编码）
    t0 = time.perf_counter()
    qvec = get_embedding().embed_query(query)
    t["t_embed"] = time.perf_counter() - t0

    # 阶段2a 向量召回（串行第 1 步）
    t0 = time.perf_counter()
    vs = get_vectorstore(collection_name)
    vec_docs = [d for d, _ in vs.similarity_search_with_score_by_vector(
        qvec, k=TOP_K_RECALL)]
    t["t_vector"] = time.perf_counter() - t0

    # 阶段2b BM25 召回（串行第 2 步）
    t0 = time.perf_counter()
    bm25_docs = _bm25_cache[collection_name].invoke(query)
    t["t_bm25"] = time.perf_counter() - t0

    # 阶段3 RRF 融合（基线不做截断，候选全量送重排）
    t0 = time.perf_counter()
    fused = _rrf_fuse([vec_docs, bm25_docs])
    t["t_rrf"] = time.perf_counter() - t0

    # 阶段4 CrossEncoder 重排（全部候选，完整原文）
    t0 = time.perf_counter()
    ranked = _rerank(query, fused, top_n=top_k)
    t["t_rerank"] = time.perf_counter() - t0
    t["n_rerank_candidates"] = len(fused)

    # 阶段5 阈值过滤 + 上下文组装
    t0 = time.perf_counter()
    docs = _threshold_filter(ranked, SCORE_THRESHOLD)
    _assemble(docs, query)
    t["t_post"] = time.perf_counter() - t0

    t["t_total"] = sum(v for k, v in t.items() if k.startswith("t_"))
    return docs, t


# ==================== 优化实现 ====================

RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "6"))   # O3：送重排的候选上限
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "256"))   # O4：重排输入截断
ENABLE_QUERY_CACHE = os.getenv("ENABLE_QUERY_CACHE", "1") == "1"  # O5：精确匹配缓存


def optimized_search(query: str, collection_name: str,
                     top_k: int = TOP_K_RERANK) -> Tuple[List[Document], Dict[str, float]]:
    """优化检索：单次编码 + 并行双路 + 候选截断 + 重排输入截断，分阶段计时"""
    t: Dict[str, float] = {}

    # 阶段1 查询处理（O1：一次编码，两路复用）
    t0 = time.perf_counter()
    qvec = get_embedding().embed_query(query)
    t["t_embed"] = time.perf_counter() - t0

    # 阶段2 双路并行召回（O2）
    t0 = time.perf_counter()
    vs = get_vectorstore(collection_name)

    def _vec():
        return [d for d, _ in vs.similarity_search_with_score_by_vector(
            qvec, k=TOP_K_RECALL)]

    def _bm25():
        return _bm25_cache[collection_name].invoke(query)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_vec, f_bm = pool.submit(_vec), pool.submit(_bm25)
        vec_docs, bm25_docs = f_vec.result(), f_bm.result()
    t["t_recall_parallel"] = time.perf_counter() - t0

    # 阶段3 RRF 融合（O3：去重后只保留前 RERANK_CANDIDATES 条候选）
    t0 = time.perf_counter()
    fused = _rrf_fuse([vec_docs, bm25_docs])[:RERANK_CANDIDATES]
    t["t_rrf"] = time.perf_counter() - t0

    # 阶段4 CrossEncoder 重排（O4：输入截断）
    t0 = time.perf_counter()
    ranked = _rerank(query, fused, top_n=top_k, max_chars=RERANK_MAX_CHARS)
    t["t_rerank"] = time.perf_counter() - t0
    t["n_rerank_candidates"] = len(fused)

    # 阶段5 阈值过滤 + 上下文组装
    t0 = time.perf_counter()
    docs = _threshold_filter(ranked, SCORE_THRESHOLD)
    _assemble(docs, query)
    t["t_post"] = time.perf_counter() - t0

    t["t_total"] = sum(v for k, v in t.items() if k.startswith("t_"))
    return docs, t


_cache_store: Dict[str, Tuple[List[Document], Dict[str, float]]] = {}


def optimized_search_cached(query: str, collection_name: str,
                            top_k: int = TOP_K_RERANK):
    """O5：精确匹配查询缓存（同一问题重复提问时直接命中，秒回）"""
    key = f"{collection_name}|{top_k}|{query}"
    if ENABLE_QUERY_CACHE and key in _cache_store:
        docs, t = _cache_store[key]
        t2 = dict(t)
        t2["cache_hit"] = 1
        t2["t_total"] = 0.001
        return docs, t2
    docs, t = optimized_search(query, collection_name, top_k)
    t["cache_hit"] = 0
    if ENABLE_QUERY_CACHE:
        _cache_store[key] = (docs, t)
    return docs, t
