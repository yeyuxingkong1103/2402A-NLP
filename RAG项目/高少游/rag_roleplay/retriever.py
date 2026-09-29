# -*- coding: utf-8 -*-
"""
检索模块：向量检索 + BM25 关键词检索 -> RRF 融合 -> CrossEncoder 重排 -> 分数过滤
支持多集合（按角色路由）和多路召回（知识库 + 长期记忆）

入口函数（外部只需关注这两个）：
    - multi_route_recall  : 对外主入口，多路召回 + 去重（main.py 调它）
    - hybrid_search      : 单集合混合检索（被 multi_route_recall 调用）
"""

import os
from typing import List, Optional, Dict

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
    MILVUS_COLLECTION, MILVUS_COLLECTIONS,
    get_collection_for_role,
)
from db_milvus import get_vectorstore, get_milvus_client, fetch_all_text
from logger import get_logger

logger = get_logger(__name__)

# 模块级单例：重排模型加载一次后复用；BM25 按集合分别缓存
_reranker: Optional[HuggingFaceCrossEncoder] = None
_bm25_cache: Dict[str, BM25Retriever] = {}


def _chinese_tokenize(text: str) -> List[str]:
    """BM25 预分词：jieba 分词后去空白、转小写。大小写不统一会把同一个词算成两个。"""
    return [w.strip().lower() for w in jieba.lcut(text) if w.strip()]


def get_reranker() -> HuggingFaceCrossEncoder:
    """获取重排模型单例（首次调用加载，之后复用）"""
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


# ================================ BM25 索引管理 ================================
# BM25 需要从 Milvus 全量拉文本重建，入库后必须调 refresh_bm25 让新文档可被检索。


def refresh_bm25(collection_name: str = None):
    """从指定集合全量拉文本重建 BM25 索引（入库后必调）"""
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    global _bm25_cache
    docs = fetch_all_text(collection_name)
    if docs:
        _bm25_cache[collection_name] = BM25Retriever.from_documents(
            documents=docs,
            k=TOP_K_RECALL,
            preprocess_func=_chinese_tokenize,
        )
        logger.info(f"BM25 索引已重建：集合 {collection_name}，语料 {len(docs)} 条")
    else:
        _bm25_cache.pop(collection_name, None)
        logger.warning(f"BM25 索引为空：集合 {collection_name}（无数据）")


def refresh_all_bm25():
    """启动时重建所有角色的 BM25 索引"""
    for role_key, collection_name in MILVUS_COLLECTIONS.items():
        try:
            refresh_bm25(collection_name)
        except Exception as e:
            logger.warning(f"刷新 BM25 失败 {collection_name}：{e}")


# ================================ 检索入口 ================================
# hybrid_search：单集合的「向量 + BM25 + RRF + 重排 + 过滤」五步流水线
# multi_route_recall：在 hybrid_search 之上加长期记忆路 + 去重，对外的总入口


def hybrid_search(query: str, top_k: int = TOP_K_RERANK,
                  score_threshold: float = SCORE_THRESHOLD,
                  collection_name: str = None) -> List[Document]:
    """
    单集合混合检索：向量 + BM25 → RRF 融合 → 重排 → 分数过滤

    流程：
        1. 向量检索召回 TOP_K_RECALL 条（按语义，Dense）
        2. BM25 召回 TOP_K_RECALL 条（按词频，Sparse）
        3. EnsembleRetriever 用 RRF 融合两路排名（权重 0.5/0.5，c=60）
        4. CrossEncoder 精排，只留 top_k 条
        5. score < score_threshold 的丢弃
    """
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    # 前置校验：集合存在 + BM25 已初始化
    client = get_milvus_client()
    if not client.has_collection(collection_name):
        raise RuntimeError(f"知识库集合 {collection_name} 不存在，请先入库")
    if collection_name not in _bm25_cache:
        raise RuntimeError(f"BM25 索引未初始化：集合 {collection_name}，请先入库或重启服务")

    # 1-2. Dense 路 + Sparse 路
    vector_retriever = get_vectorstore(collection_name).as_retriever(
        search_kwargs={"k": TOP_K_RECALL}
    )
    bm25 = _bm25_cache[collection_name]

    # 3. RRF 融合两路排名
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25],
        weights=[0.5, 0.5],
        c=60,
    )

    # 4. CrossEncoder 重排（精排）取 top_k
    compressor = CrossEncoderReranker(model=get_reranker(), top_n=top_k)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble,
    )
    results = final_retriever.invoke(query)

    # 5. 丢弃低分（防给模型喂不相关内容）
    filtered = []
    for doc in results:
        score = doc.metadata.get("score", 1.0)
        if score >= score_threshold:
            filtered.append(doc)
        else:
            logger.debug(f"过滤低分文档：score={score:.4f} text={doc.page_content[:30]}")

    if not filtered:
        logger.warning(f"查询 '{query[:30]}' 结果全被分数过滤掉（集合 {collection_name}）")
    return filtered


def multi_route_recall(query: str, user_id: int = 0, top_k: int = TOP_K_RERANK,
                       role_key: str = None) -> List[Document]:
    """
    多路召回（对外主入口）：知识库混合检索 + 长期记忆 → 去重

    路 1：按角色路由到对应 Milvus 集合，调 hybrid_search
    路 2：查该用户的长期记忆集合（仅 user_id > 0 时）
    去重：按正文前 100 字判重，最多返回 top_k × 2 条
    """
    # 按角色确定要检索的集合
    collection_name = get_collection_for_role(role_key) if role_key else MILVUS_COLLECTION

    all_docs: List[Document] = []

    # 路 1：知识库混合检索
    try:
        kb_docs = hybrid_search(query, top_k=top_k, collection_name=collection_name)
        all_docs.extend(kb_docs)
        logger.info(f"知识库召回 {len(kb_docs)} 条（集合 {collection_name}）")
    except RuntimeError as e:
        logger.warning(f"知识库召回失败：{e}")

    # 路 2：长期记忆（用户的过往对话要点）
    if user_id > 0:
        try:
            from db_milvus import search_long_term_memory
            mem_docs = search_long_term_memory(user_id, query, top_k=3)
            all_docs.extend(mem_docs)
            logger.info(f"长期记忆召回 {len(mem_docs)} 条")
        except Exception as e:
            logger.debug(f"长期记忆召回失败：{e}")

    # 去重：正文前 100 字相同的视为重复
    seen = set()
    unique_docs = []
    for doc in all_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs[:top_k * 2]
