# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化任务
"""
Milvus 数据层：向量数据库操作，支持表格与图像语义块的统一存储。

优化点：
    1. 独立集合 rag_pdf_qa_v4：与旧集合并存，支持表格/图像结构；
    2. 批次优化：入库批次 25，平衡速度和内存；
    3. 结构元数据：额外存储表格/图像信息，便于检索后处理。
"""

import uuid
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from pymilvus import MilvusClient

from config import (
    MILVUS_URI, MILVUS_COLLECTION,
    MILVUS_INDEX_TYPE, MILVUS_METRIC_TYPE,
    EMBED_MODEL_PATH, EMBED_DEVICE, EMBED_DIM,
)
from logger import get_logger

logger = get_logger(__name__)

_embedding: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[Milvus] = None
_milvus_client: Optional[MilvusClient] = None


def get_embedding() -> HuggingFaceEmbeddings:
    """获取向量模型单例。"""
    global _embedding
    if _embedding is None:
        _embedding = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_PATH,
            model_kwargs={"device": EMBED_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"向量模型已加载：{EMBED_MODEL_PATH}")
    return _embedding


def get_vectorstore() -> Milvus:
    """获取向量存储单例（独立集合 rag_pdf_qa_v3）。"""
    global _vectorstore
    if _vectorstore is None:
        index_params = {
            "index_type": MILVUS_INDEX_TYPE,
            "metric_type": MILVUS_METRIC_TYPE,
            "params": {},
        }
        _vectorstore = Milvus(
            embedding_function=get_embedding(),
            collection_name=MILVUS_COLLECTION,
            connection_args={"uri": MILVUS_URI},
            index_params=index_params,
            auto_id=False,
        )
        logger.info(f"Milvus 向量存储已连接：{MILVUS_URI}/{MILVUS_COLLECTION}")
    return _vectorstore


def get_milvus_client() -> MilvusClient:
    """获取 Milvus 客户端单例。"""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(uri=MILVUS_URI)
        logger.info(f"Milvus 客户端已连接：{MILVUS_URI}")
    return _milvus_client


def ingest_documents(documents: List[Document], batch_size: int = 25) -> int:
    """将 Document 列表写入 Milvus。

    Args:
        documents: 待入库的 Document 列表
        batch_size: 每批次写入数量

    Returns:
        int: 成功写入的文档数量
    """
    if not documents:
        logger.warning("没有待入库的文档")
        return 0

    vectorstore = get_vectorstore()
    total = len(documents)
    success_count = 0

    logger.info(f"开始入库：共 {total} 个文档，批次大小 {batch_size}")

    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        try:
            # 生成唯一 ID
            ids = [str(uuid.uuid4()) for _ in batch]
            vectorstore.add_documents(batch, ids=ids)
            success_count += len(batch)
            logger.info(f"批次 {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} 入库成功")
        except Exception as e:
            logger.error(f"批次 {i//batch_size + 1} 入库失败：{e}")

    logger.info(f"入库完成：成功 {success_count}/{total}")
    return success_count


def get_collection_count() -> int:
    """获取集合中的文档数量。"""
    try:
        client = get_milvus_client()
        stats = client.get_collection_stats(MILVUS_COLLECTION)
        return int(stats.get("row_count", 0))
    except Exception as e:
        logger.error(f"获取集合统计失败：{e}")
        return 0


def clear_collection():
    """清空集合（用于重建）。"""
    try:
        client = get_milvus_client()
        if client.has_collection(MILVUS_COLLECTION):
            client.drop_collection(MILVUS_COLLECTION)
            logger.info(f"集合 {MILVUS_COLLECTION} 已清空")
    except Exception as e:
        logger.error(f"清空集合失败：{e}")


def get_all_texts() -> List[dict]:
    """获取所有文本（用于 BM25 索引重建）。"""
    try:
        client = get_milvus_client()
        results = client.query(
            collection_name=MILVUS_COLLECTION,
            filter="",
            output_fields=["pk", "text"],
            limit=10000,
        )
        return results
    except Exception as e:
        logger.error(f"获取所有文本失败：{e}")
        return []
