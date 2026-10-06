# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""
文本切分模块（优化版）：递归字符切分 + 招股书专用分隔符优化。

优化点：
    1. 块大小 500→400：语义更聚焦，向量召回精度提升；
    2. 重叠 80→50：减少冗余，同时仍兜住切口；
    3. 分隔符优先级调优：把中文句号、分号提到前面，让切分更贴合中文句法边界；
    4. 去掉英文句号兜底优先级，避免中文文本被英文规则误切。
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from logger import get_logger

logger = get_logger(__name__)

# 优化：分隔符优先级调优，更贴合中文文档结构
NATURAL_SEPARATORS = [
    "\n\n",  # 空行（段落之间）
    "\n",  # 换行（标题/条目之间）
    "。",  # 中文句号
    "；",  # 分号
    "！",  # 感叹号
    "？",  # 问号
    ". ",  # 英文句号+空格
    " ",  # 空格
    "",  # 兜底：硬切
]


def _make_splitter(chunk_size: int = CHUNK_SIZE,
                   chunk_overlap: int = CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    """构造递归字符切分器。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=NATURAL_SEPARATORS,
    )


def split_text(text: str, source: str = "") -> List[Document]:
    """切分纯文本，返回 Document 列表。"""
    if not text or not text.strip():
        logger.warning("切分输入为空，跳过")
        return []

    splitter = _make_splitter()
    chunks = splitter.create_documents([text])

    for i, c in enumerate(chunks):
        c.metadata = {
            "source": source,
            "chunk_index": i,
            "page_number": 0,
        }
    logger.info(f"切分完成：source={source or '(未指定)'}，共 {len(chunks)} 块（chunk_size={CHUNK_SIZE}）")
    return chunks


def split_documents(documents: List[Document]) -> List[Document]:
    """对 Document 列表做切分（保留原有 metadata）。"""
    if not documents:
        logger.warning("切分文档列表为空，跳过")
        return []

    splitter = _make_splitter()
    chunks = splitter.split_documents(documents)

    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    logger.info(f"切分 Document 完成：输入 {len(documents)} 篇，输出 {len(chunks)} 块（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")
    return chunks
