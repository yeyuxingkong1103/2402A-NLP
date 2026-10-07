# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-混合检索任务
"""
文本切分模块：将 Document 切分为适合向量化的块。

优化点：
    1. 结构块保护：表格块、图像语义块不进一步切分，保持结构/语义完整；
    2. 文本块智能切分：RecursiveCharacterTextSplitter，保持语义连贯；
    3. 保留元数据：每块保留来源文件、页码、块类型等信息。
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from logger import get_logger

logger = get_logger(__name__)


def split_documents(documents: List[Document]) -> List[Document]:
    """将 Document 列表切分为适合向量化的块。

    Args:
        documents: 原始 Document 列表（文本/表格/图像块）

    Returns:
        List[Document]: 切分后的 Document 列表
    """
    if not documents:
        logger.warning("输入文档列表为空")
        return []

    # 文本块需要切分；表格块、图像语义块保持完整不切分
    text_docs = [d for d in documents if d.metadata.get("block_type") == "text"]
    keep_docs = [d for d in documents if d.metadata.get("block_type") in ("table", "image")]

    logger.info(f"切分前：文本块 {len(text_docs)} 个，结构块(表格/图像) {len(keep_docs)} 个")

    # 1. 切分文本块
    split_text_docs = []
    if text_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        split_text_docs = text_splitter.split_documents(text_docs)
        logger.info(f"文本块切分完成：{len(split_text_docs)} 个")

    # 2. 合并文本块与结构块，统一重新编号 chunk_index
    all_docs = split_text_docs + keep_docs
    for idx, doc in enumerate(all_docs):
        doc.metadata["chunk_index"] = idx

    logger.info(
        f"切分完成：共 {len(all_docs)} 个块（文本 {len(split_text_docs)} + 结构 {len(keep_docs)}）"
    )
    return all_docs


def get_chunk_stats(documents: List[Document]) -> dict:
    """获取切分统计信息。"""
    if not documents:
        return {}

    text_count = sum(1 for d in documents if d.metadata.get("block_type") == "text")
    table_count = sum(1 for d in documents if d.metadata.get("block_type") == "table")
    image_count = sum(1 for d in documents if d.metadata.get("block_type") == "image")
    total_chars = sum(len(d.page_content) for d in documents)
    avg_chunk_size = total_chars / len(documents) if documents else 0

    return {
        "total_chunks": len(documents),
        "text_chunks": text_count,
        "table_chunks": table_count,
        "image_chunks": image_count,
        "total_chars": total_chars,
        "avg_chunk_size": round(avg_chunk_size, 2),
    }
