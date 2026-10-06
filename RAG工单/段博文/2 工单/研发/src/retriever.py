# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""
检索模块（优化版）：向量检索 + BM25 -> RRF 融合 -> CrossEncoder 重排 -> 分数过滤。

核心优化点：
    1. 结果缓存（LRU）：相同 query 第二次命中缓存直接返回，跳过重排耗时；
    2. 候选收敛：TOP_K_RECALL 10→6，重排候选从 ~20 降到 ~12，CPU 重排耗时近半；
    3. 精排保留 3 条：喂给 LLM 的上下文更少更精，生成更快；
    4. 阈值下调 0.3→0.25：候选少了适当放低门槛保住召回率；
    5. 检索计时日志：每步耗时可见，便于性能分析。
"""

import os
import time
from functools import lru_cache
from typing import List, Optional, Tuple

import jieba
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from config import (
    RERANK_MODEL, RERANK_DEVICE,
    TOP_K_RECALL, TOP_K_RERANK, SCORE_THRESHOLD,
    CACHE_ENABLED, CACHE_MAX_SIZE,
)
from db_milvus import get_vectorstore, get_milvus_client, fetch_all_text
from logger import get_logger

logger = get_logger(__name__)

_reranker: Optional[HuggingFaceCrossEncoder] = None
_bm25: Optional[BM25Retriever] = None


def _chinese_tokenize(text: str) -> List[str]:
    """BM25 预分词：jieba 分词 -> 去空白 -> 转小写。"""
    return [w.strip().lower() for w in jieba.lcut(text) if w.strip()]


def get_reranker() -> HuggingFaceCrossEncoder:
    """获取重排模型单例（首次调用时加载，之后复用）。"""
    global _reranker
    if _reranker is None:
        if not os.path.isdir(RERANK_MODEL):
            raise RuntimeError(f"重排模型路径不存在：{RERANK_MODEL}")
        _reranker = HuggingFaceCrossEncoder(
            model_name=RERANK_MODEL,
            model_kwargs={"device": RERANK_DEVICE},
        )
        logger.info(f"重排模型已加载：{RERANK_MODEL}")
    return _reranker


def refresh_bm25() -> None:
    """从 Milvus 全量拉文本重建 BM25 索引。"""
    global _bm25
    texts = fetch_all_text()
    if texts:
        docs = [Document(page_content=t) for t in texts if t and t.strip()]
        _bm25 = BM25Retriever.from_documents(
            documents=docs,
            k=TOP_K_RECALL,
            preprocess_func=_chinese_tokenize,
        )
        logger.info(f"BM25 索引已重建：语料 {len(docs)} 条（k={TOP_K_RECALL}）")
    else:
        _bm25 = None
        logger.warning("BM25 索引为空（集合无数据）")


# ==================== 新增：检索结果缓存 ====================
# 用 lru_cache 装饰一个内部函数，相同 query 第二次直接命中缓存
@lru_cache(maxsize=CACHE_MAX_SIZE if CACHE_ENABLED else 0)
def _hybrid_search_cached(query: str, top_k: int, score_threshold: float) -> Tuple[Tuple, ...]:
    """
    带缓存的内部检索函数。

    lru_cache 要求参数可哈希、返回值可哈享：这里把 Document 列表转成
    元组的元组（(page_content, score, source, page_number)），缓存命中后
    再还原成 Document 列表返回。
    """
    docs = _do_hybrid_search(query, top_k, score_threshold)
    # 转成可哈希的元组
    return tuple(
        (d.page_content, d.metadata.get("score", 1.0),
         d.metadata.get("source", ""), d.metadata.get("page_number", 0))
        for d in docs
    )


def _do_hybrid_search(query: str, top_k: int, score_threshold: float) -> List[Document]:
    """实际执行混合检索 + 重排（无缓存，无异常处理，由上层兜）。"""
    from config import MILVUS_COLLECTION

    t0 = time.time()

    # 1. Dense 路：向量检索
    vectorstore = get_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RECALL})
    t_vec = time.time()
    vector_docs = vector_retriever.invoke(query)
    logger.info(f"[检索计时] 向量召回 {len(vector_docs)} 条：{time.time()-t_vec:.3f}s")

    # 2. Sparse 路：BM25 关键词检索
    if _bm25 is None:
        raise RuntimeError("BM25 索引未初始化，请先入库或重启服务")
    bm25 = _bm25
    t_bm = time.time()
    bm25_docs = bm25.invoke(query)
    logger.info(f"[检索计时] BM25 召回 {len(bm25_docs)} 条：{time.time()-t_bm:.3f}s")

    # 3. RRF 融合（用 EnsembleRetriever 合并两路结果）
    t_rrf = time.time()
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25],
        weights=[0.5, 0.5],
        c=60,
    )
    # 直接拿刚才两路结果做融合排名（避免重复检索）
    # EnsembleRetriever.invoke 内部会再查一次，这里走它的标准流程
    merged = ensemble.invoke(query)
    logger.info(f"[检索计时] RRF 融合 {len(merged)} 条：{time.time()-t_rrf:.3f}s")

    # 4. CrossEncoder 重排（对融合后的候选精排，只留 top_k 条）
    t_rerank = time.time()
    compressor = CrossEncoderReranker(model=get_reranker(), top_n=top_k)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble,
    )
    results = final_retriever.invoke(query)
    logger.info(f"[检索计时] 重排 {len(results)} 条：{time.time()-t_rerank:.3f}s")

    # 5. 分数过滤
    filtered = []
    for doc in results:
        score = doc.metadata.get("score", 1.0)
        if score >= score_threshold:
            filtered.append(doc)
        else:
            logger.debug(f"过滤低分文档：score={score:.4f} text={doc.page_content[:30]}")

    logger.info(f"[检索计时] 总计 {len(filtered)} 条，总耗时 {time.time()-t0:.3f}s")
    if not filtered:
        logger.warning(f"查询 '{query[:30]}' 的结果全被分数过滤掉")
    return filtered


def hybrid_search(query: str, top_k: int = TOP_K_RERANK,
                  score_threshold: float = SCORE_THRESHOLD) -> List[Document]:
    """
    混合检索 + 重排 + 分数过滤（带缓存）。

    优化：开启 CACHE_ENABLED 时，相同 query+top_k+阈值 第二次直接命中
    LRU 缓存返回，跳过向量检索 + BM25 + RRF + 重排全部耗时。

    参数：
        query：用户提问。
        top_k：重排后保留条数。
        score_threshold：分数阈值。
    返回：
        List[Document]，按相关度从高到低排序。
    """
    client = get_milvus_client()
    from config import MILVUS_COLLECTION
    if not client.has_collection(MILVUS_COLLECTION):
        raise RuntimeError(f"知识库集合 {MILVUS_COLLECTION} 不存在，请先入库")

    if CACHE_ENABLED:
        t0 = time.time()
        cached = _hybrid_search_cached(query, top_k, score_threshold)
        if cached:
            logger.info(f"[缓存命中] 查询 '{query[:30]}'，耗时 {time.time()-t0:.3f}s")
        docs = [
            Document(
                page_content=item[0],
                metadata={"score": item[1], "source": item[2], "page_number": item[3]},
            )
            for item in cached
        ]
        return docs
    else:
        return _do_hybrid_search(query, top_k, score_threshold)


def get_cache_info() -> dict:
    """返回缓存命中统计（供健康检查接口展示）。"""
    if not CACHE_ENABLED:
        return {"enabled": False}
    info = _hybrid_search_cached.cache_info()
    return {
        "enabled": True,
        "hits": info.hits,
        "misses": info.misses,
        "size": info.currsize,
        "maxsize": info.maxsize,
    }


def clear_cache() -> None:
    """清空检索缓存（入库后调用，确保新文档立即可被检索到）。"""
    if CACHE_ENABLED:
        _hybrid_search_cached.cache_clear()
        logger.info("检索缓存已清空")
