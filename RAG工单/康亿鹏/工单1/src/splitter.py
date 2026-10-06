# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：文本切分模块。将解析后的长文本切分为带重叠的语义片段，
          作为向量检索的最小单元。
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from src.utils import logger


def build_splitter(chunk_size: int = None, chunk_overlap: int = None) -> RecursiveCharacterTextSplitter:
    """构建文本切分器（优先按中英文标点/换行切分，尽量保持语义完整）。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP,
        separators=config.CHUNK_SEPARATORS,
        length_function=len,
        add_start_index=True,
    )


def split_documents(documents: list, chunk_size: int = None, chunk_overlap: int = None) -> list:
    """切分 Document 列表，并补充 chunk_index 元数据（便于溯源与去重）。"""
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        # 统一 chunk_id：文件名_页码_序号，作为向量库主键
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata.get('source', 'doc')}"
            f"_{chunk.metadata.get('page', 0)}"
            f"_{chunk.metadata.get('chunk_index', 0)}"
        )
    logger.info("文本切分完成：%d 个文档 → %d 个片段（chunk_size=%d, overlap=%d）",
                len(documents), len(chunks), chunk_size or config.CHUNK_SIZE,
                config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap)
    return chunks
