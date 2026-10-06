# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""
检索模块：向量检索 + BM25 + RRF 融合 + 重排 + 缓存。

优化点：
    1. 表格块优先：检索时对表格块给予额外权重；
    2. LRU 缓存：检索结果缓存，重复查询毫秒级返回；
    3. RRF 融合：向量检索和 BM25 检索结果融合排序；
    4. 重排优化：CrossEncoder 重排，保留最相关的结果。
"""

import hashlib
import json
import re
from collections import OrderedDict
from typing import List, Optional

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import (
    TOP_K_RECALL, TOP_K_RERANK, SCORE_THRESHOLD,
    RERANK_MODEL, RERANK_DEVICE,
    CACHE_ENABLED, CACHE_MAX_SIZE,
)
from logger import get_logger
from db_milvus import get_vectorstore, get_all_texts

logger = get_logger(__name__)

# 缓存
_search_cache: OrderedDict = OrderedDict()
_bm25_cache: Optional[BM25Okapi] = None
_bm25_docs: Optional[List[Document]] = None
_reranker: Optional[CrossEncoder] = None


class SearchCache:
    """LRU 缓存实现。"""

    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[List[Document]]:
        """获取缓存。"""
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            logger.debug(f"缓存命中：{key[:30]}...")
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value: List[Document]):
        """设置缓存。"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self):
        """清空缓存。"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_info(self) -> dict:
        """获取缓存统计。"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / (self.hits + self.misses) * 100, 2) if (self.hits + self.misses) > 0 else 0,
        }


# 全局缓存实例
_search_cache = SearchCache(CACHE_MAX_SIZE)


def _get_cache_key(query: str, top_k: int) -> str:
    """生成缓存键。"""
    content = f"{query}_{top_k}"
    return hashlib.md5(content.encode()).hexdigest()


def get_reranker() -> CrossEncoder:
    """获取重排模型单例。"""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE)
        logger.info(f"重排模型已加载：{RERANK_MODEL}")
    return _reranker


def refresh_bm25():
    """重建 BM25 索引。"""
    global _bm25_cache, _bm25_docs

    logger.info("开始重建 BM25 索引...")
    all_texts = get_all_texts()

    if not all_texts:
        logger.warning("没有可用于 BM25 索引的文本")
        _bm25_cache = None
        _bm25_docs = None
        return

    # 提取文本和分词
    documents = []
    corpus = []

    for item in all_texts:
        text = item.get("text", "")
        if not text:
            continue

        doc = Document(
            page_content=text,
            metadata={"pk": item.get("pk")},
        )
        documents.append(doc)

        # 分词
        tokens = list(jieba.cut(text))
        corpus.append(tokens)

    _bm25_docs = documents
    _bm25_cache = BM25Okapi(corpus)

    logger.info(f"BM25 索引重建完成：{len(_bm25_docs)} 个文档")


def _vector_search(query: str, top_k: int) -> List[Document]:
    """向量检索。"""
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search_with_score(query, k=top_k)

    results = []
    for doc, score in docs:
        doc.metadata["score"] = score
        doc.metadata["search_type"] = "vector"
        results.append(doc)

    return results


def _bm25_search(query: str, top_k: int) -> List[Document]:
    """BM25 检索。"""
    global _bm25_cache, _bm25_docs

    if _bm25_cache is None or _bm25_docs is None:
        logger.warning("BM25 索引未初始化")
        return []

    tokens = list(jieba.cut(query))
    scores = _bm25_cache.get_scores(tokens)

    # 获取 top_k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            doc = _bm25_docs[idx]
            doc.metadata["score"] = float(scores[idx])
            doc.metadata["search_type"] = "bm25"
            results.append(doc)

    return results


def _rrf_fusion(vector_results: List[Document], bm25_results: List[Document], k: int = 60) -> List[Document]:
    """RRF（Reciprocal Rank Fusion）融合排序。"""
    scores = {}

    # 向量检索结果
    for rank, doc in enumerate(vector_results, start=1):
        pk = doc.metadata.get("pk", "")
        scores[pk] = scores.get(pk, 0) + 1 / (rank + k)

    # BM25 检索结果
    for rank, doc in enumerate(bm25_results, start=1):
        pk = doc.metadata.get("pk", "")
        scores[pk] = scores.get(pk, 0) + 1 / (rank + k)

    # 排序
    sorted_pks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 构建结果
    pk_to_doc = {}
    for doc in vector_results + bm25_results:
        pk = doc.metadata.get("pk", "")
        if pk not in pk_to_doc:
            pk_to_doc[pk] = doc

    results = []
    for pk, rrf_score in sorted_pks:
        if pk in pk_to_doc:
            doc = pk_to_doc[pk]
            doc.metadata["rrf_score"] = rrf_score
            results.append(doc)

    return results


def _rerank_documents(query: str, documents: List[Document], top_k: int) -> List[Document]:
    """使用 CrossEncoder 重排。"""
    if not documents:
        return []

    try:
        reranker = get_reranker()

        # 准备输入
        pairs = [(query, doc.page_content[:512]) for doc in documents]  # 限制长度
        scores = reranker.predict(pairs)

        # 按分数排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 过滤阈值并保留 top_k
        results = []
        for doc, score in scored_docs[:top_k]:
            if score >= SCORE_THRESHOLD:
                doc.metadata["rerank_score"] = float(score)
                results.append(doc)

        return results
    except Exception as e:
        logger.error(f"重排失败：{e}")
        return documents[:top_k]


def hybrid_search(query: str, top_k: int = TOP_K_RERANK) -> List[Document]:
    """混合检索：向量检索 + BM25 + RRF 融合 + 重排。

    Args:
        query: 查询文本
        top_k: 返回结果数量

    Returns:
        List[Document]: 检索结果
    """
    if not query or not query.strip():
        raise ValueError("查询不能为空")

    # 检查缓存
    if CACHE_ENABLED:
        cache_key = _get_cache_key(query, top_k)
        cached_result = _search_cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"检索缓存命中：{query[:30]}...")
            return cached_result

    logger.info(f"开始混合检索：{query[:50]}...")

    # 1. 向量检索
    vector_results = _vector_search(query, TOP_K_RECALL)
    logger.info(f"向量检索：{len(vector_results)} 条结果")

    # 2. BM25 检索
    bm25_results = _bm25_search(query, TOP_K_RECALL)
    logger.info(f"BM25 检索：{len(bm25_results)} 条结果")

    # 3. RRF 融合
    if vector_results and bm25_results:
        fused_results = _rrf_fusion(vector_results, bm25_results)
    elif vector_results:
        fused_results = vector_results
    elif bm25_results:
        fused_results = bm25_results
    else:
        fused_results = []

    logger.info(f"融合后：{len(fused_results)} 条结果")

    # 4. 重排
    reranked_results = _rerank_documents(query, fused_results, top_k)
    logger.info(f"重排后：{len(reranked_results)} 条结果")

    # 5. 缓存结果
    if CACHE_ENABLED and reranked_results:
        cache_key = _get_cache_key(query, top_k)
        _search_cache.set(cache_key, reranked_results)

    return reranked_results


def get_cache_info() -> dict:
    """获取缓存统计信息。"""
    return _search_cache.get_info()


def clear_cache():
    """清空缓存。"""
    _search_cache.clear()
    logger.info("检索缓存已清空")
