# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""
Milvus 数据层（优化版）：单例连接 + 幂等去重入库 + 全量拉文本。

优化点：
    1. 独立集合 rag_pdf_qa_v2：与旧集合并存，便于前后对比；
    2. 入库批次 10→25：减少网络往返，加快入库速度；
    3. 其余逻辑保持工单一一致，确保对比公平。
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
    """获取向量存储单例（独立集合 rag_pdf_qa_v2）。"""
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
    """获取原生客户端单例。"""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(uri=MILVUS_URI)
        logger.info(f"Milvus 原生客户端已连接：{MILVUS_URI}")
    return _milvus_client


def ingest_documents(docs: List[Document], batch_size: int = 25) -> int:
    """向量化并写入 Milvus 集合，返回语料总数。

    优化：batch_size 10→25，减少网络往返。
    """
    if not docs:
        logger.warning("入库文档列表为空，跳过")
        return 0

    client = get_milvus_client()
    existing_keys = set()
    # 优化：先查行数，空集合直接跳过去重查询（避免新集合未加载导致卡住）
    need_dedup = False
    if client.has_collection(MILVUS_COLLECTION):
        try:
            stats = client.get_collection_stats(MILVUS_COLLECTION)
            need_dedup = int(stats.get("row_count", 0)) > 0
        except Exception:
            need_dedup = True  # 查不到就保守做去重
    if need_dedup:
        try:
            client.load_collection(MILVUS_COLLECTION)  # 确保集合已加载
            offset = 0
            page_size = 500
            while True:
                rows = client.query(
                    collection_name=MILVUS_COLLECTION,
                    filter="source != ''",
                    output_fields=["source", "chunk_index"],
                    limit=page_size,
                    offset=offset,
                )
                if not rows:
                    break
                for r in rows:
                    existing_keys.add((r.get("source", ""), r.get("chunk_index", -1)))
                if len(rows) < page_size:
                    break
                offset += page_size
        except Exception as e:
            logger.warning(f"查询已有键失败（{e}），本次跳过去重")

    filtered = []
    skipped = 0
    for d in docs:
        key = (d.metadata.get("source", ""), d.metadata.get("chunk_index", -1))
        if key in existing_keys:
            skipped += 1
            continue
        existing_keys.add(key)
        filtered.append(d)

    if not filtered:
        logger.info(f"无新文档需要入库（跳过 {skipped} 条重复）")
        return 0

    vs = get_vectorstore()
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    total_batches = (len(filtered) + batch_size - 1) // batch_size
    bar = tqdm(total=total_batches, desc=f"[{MILVUS_COLLECTION}]", unit="batch") if tqdm else None
    for i in range(0, len(filtered), batch_size):
        batch = filtered[i:i + batch_size]
        batch_ids = [str(uuid.uuid4()) for _ in batch]
        vs.add_documents(documents=batch, ids=batch_ids)
        if bar:
            bar.set_postfix_str(f"{len(batch)}条/批")
            bar.update(1)
        else:
            logger.info(f"  写入批次 {i // batch_size + 1}/{total_batches}（{len(batch)} 条）")
    if bar:
        bar.close()

    total = client.get_collection_stats(MILVUS_COLLECTION).get("row_count", 0) if client.has_collection(MILVUS_COLLECTION) else len(filtered)
    logger.info(f"入库 {len(filtered)} 条（跳过重复 {skipped} 条），当前语料 {total} 条")
    return total


def fetch_all_text() -> List[str]:
    """分页拉取集合内全部文本（用于重建 BM25）。"""
    client = get_milvus_client()
    if not client.has_collection(MILVUS_COLLECTION):
        return []

    all_text: List[str] = []
    offset = 0
    page_size = 500
    while True:
        rows = client.query(
            collection_name=MILVUS_COLLECTION,
            filter="",
            output_fields=["text"],
            limit=page_size,
            offset=offset,
        )
        if not rows:
            break
        for row in rows:
            all_text.append(row.get("text", ""))
        if len(rows) < page_size:
            break
        offset += page_size
    logger.info(f"从 Milvus 拉取 {len(all_text)} 条文本（用于 BM25）")
    return all_text


def get_collection_count() -> int:
    """获取集合中的文档数。"""
    client = get_milvus_client()
    if not client.has_collection(MILVUS_COLLECTION):
        return 0
    try:
        stats = client.get_collection_stats(MILVUS_COLLECTION)
        return int(stats.get("row_count", 0))
    except Exception as e:
        logger.warning(f"获取集合统计失败：{e}")
        return 0
