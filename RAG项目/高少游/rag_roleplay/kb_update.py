# -*- coding: utf-8 -*-
"""
Milvus 知识库动态更新模块（多集合版）

功能：
    1. 按来源删除：删除某文件的所有 chunk（文档被废弃时）
    2. 按来源更新：删除旧 chunk -> 重新切分 -> 重新写入（文档修改后）
    3. 列出来源：查看知识库里有哪些文件、每个文件多少条
    4. 全量重建：删集合 -> 重新入库（模型/维度变了时）

所有操作都接受 collection_name 参数，支持多集合架构
"""

import uuid  # 生成主键
from datetime import datetime  # 时间戳
from typing import List, Dict, Optional  # 类型标注

from langchain_core.documents import Document  # 统一文档结构

from db_milvus import (  # 复用已建好的单例
    get_embedding,  # 向量模型
    get_vectorstore,  # LangChain Milvus 封装
    get_milvus_client,  # 原生客户端
)
from config import MILVUS_COLLECTION, MILVUS_INDEX_TYPE, MILVUS_METRIC_TYPE, EMBED_DIM
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger


# ==================== 1. 按来源删除 ====================

def delete_by_source(source: str, collection_name: str = None) -> int:
    """删除指定来源文件的所有 chunk"""
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    client = get_milvus_client()
    if not client.has_collection(collection_name):
        logger.warning(f"集合 {collection_name} 不存在，无法删除")
        return 0

    rows = client.query(
        collection_name=collection_name,
        filter=f'source == "{source}"',
        output_fields=["pk"],
        limit=10000,
    )
    pk_list = [row["pk"] for row in rows]
    if not pk_list:
        logger.info(f"来源 '{source}' 在集合 {collection_name} 中不存在")
        return 0

    client.delete(
        collection_name=collection_name,
        pks=pk_list,
    )
    logger.info(f"已删除来源 '{source}' 的 {len(pk_list)} 条 chunk（集合 {collection_name}）")
    return len(pk_list)


# ==================== 2. 按来源更新 ====================

def update_by_source(source: str, new_text: str, is_markdown: bool = False,
                     strategy: str = "paragraph", collection_name: str = None) -> Dict:
    """更新某个来源的文档：删旧 -> 重新切分 -> 重新写入"""
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    from text_splitter import split_text

    deleted_count = delete_by_source(source, collection_name)
    logger.info(f"更新 '{source}'：已删除旧 chunk {deleted_count} 条")

    docs = split_text(
        raw_text=new_text,
        source=source,
        strategy=strategy,
        is_markdown=is_markdown,
    )
    if not docs:
        logger.warning(f"更新 '{source}'：新文本切分结果为空")
        return {
            "source": source,
            "deleted": deleted_count,
            "inserted": 0,
            "warning": "新文本切分为空，旧数据已删除但未写入新数据",
        }

    vs = get_vectorstore(collection_name)
    for i in range(0, len(docs), 10):
        batch = docs[i:i + 10]
        batch_ids = [str(uuid.uuid4()) for _ in batch]
        vs.add_documents(documents=batch, ids=batch_ids)

    client = get_milvus_client()
    total = client.get_collection_stats(collection_name).get("row_count", 0) if client.has_collection(collection_name) else len(docs)
    logger.info(f"更新 '{source}' 完成：删除 {deleted_count} 条 -> 写入 {len(docs)} 条，集合 {collection_name} 总计 {total} 条")
    return {
        "source": source,
        "deleted": deleted_count,
        "inserted": len(docs),
        "corpus_total": total,
        "collection": collection_name,
    }


# ==================== 3. 列出所有来源 ====================

def list_sources(collection_name: str = None) -> List[Dict]:
    """列出指定集合中所有文档来源及其 chunk 数量"""
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    client = get_milvus_client()
    if not client.has_collection(collection_name):
        return []

    source_counts: Dict[str, int] = {}
    offset = 0
    page_size = 500

    while True:
        rows = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["source"],
            limit=page_size,
            offset=offset,
        )
        if not rows:
            break
        for row in rows:
            src = row.get("source", "未知来源")
            source_counts[src] = source_counts.get(src, 0) + 1
        if len(rows) < page_size:
            break
        offset += page_size

    result = [{"source": k, "chunks": v} for k, v in source_counts.items()]
    result.sort(key=lambda x: x["chunks"], reverse=True)
    logger.info(f"集合 {collection_name} 共 {len(result)} 个来源，{sum(source_counts.values())} 条 chunk")
    return result


# ==================== 4. 获取来源详情 ====================

def get_source_detail(source: str, collection_name: str = None, limit: int = 100) -> Dict:
    """查看某个来源的 chunk 列表"""
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    client = get_milvus_client()
    if not client.has_collection(collection_name):
        return {"source": source, "count": 0, "chunks": []}

    rows = client.query(
        collection_name=collection_name,
        filter=f'source == "{source}"',
        output_fields=["pk", "text", "section", "chunk_index"],
        limit=limit,
    )
    chunks = [
        {
            "pk": row.get("pk", ""),
            "text": row.get("text", "")[:200],
            "section": row.get("section", ""),
            "chunk_index": row.get("chunk_index", 0),
        }
        for row in rows
    ]
    return {
        "source": source,
        "count": len(rows),
        "chunks": chunks,
        "collection": collection_name,
    }


# ==================== 5. 全量重建 ====================

def rebuild_collection(docs: List[Document], collection_name: str = None) -> Dict:
    """删除旧集合 -> 创建新集合 -> 全量写入"""
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    from pymilvus import CollectionSchema, FieldSchema, DataType

    client = get_milvus_client()

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        logger.warning(f"已删除旧集合 {collection_name}（全量重建）")

    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
    client.create_collection(collection_name, schema=schema)

    client.create_index(
        collection_name=collection_name,
        field_name="vector",
        index_type=MILVUS_INDEX_TYPE,
        metric_type=MILVUS_METRIC_TYPE,
    )

    if docs:
        vs = get_vectorstore(collection_name)
        for i in range(0, len(docs), 10):
            batch = docs[i:i + 10]
            batch_ids = [str(uuid.uuid4()) for _ in batch]
            vs.add_documents(documents=batch, ids=batch_ids)

    total = client.get_collection_stats(collection_name).get("row_count", 0)
    logger.info(f"全量重建完成：集合 {collection_name}，共 {total} 条")
    return {
        "message": "全量重建完成",
        "deleted_old": True,
        "inserted": len(docs),
        "corpus_total": total,
        "collection": collection_name,
    }
