# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：文本切分模块。将解析后的长文本切分为带重叠的语义片段，
          作为向量检索的最小单元。

工单02 优化：
    1) 表格按行分段并重复表头，避免表头与数据行被拆散（优化前表格被普通切分器切碎）；
    2) 在每个片段前注入所属章节标题，为检索补充上下文（优化前片段不含章节信息）。
"""
from langchain_core.documents import Document
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


def _split_table_document(doc: Document) -> list:
    """表格文档按行分段，每段重复表头，保证每个片段都自含列含义。"""
    lines = [ln for ln in doc.page_content.strip().split("\n") if ln.strip()]
    if len(lines) <= 3:  # 表头 + 分隔行 + 至多 1 行数据，无需切分
        return [doc]

    header, separator, rows = lines[0], lines[1], lines[2:]
    max_chars = config.TABLE_CHUNK_MAX_CHARS
    head_size = len(header) + len(separator)

    segments, current, size = [], [header, separator], head_size
    for row in rows:
        if len(current) > 2 and size + len(row) > max_chars:
            segments.append("\n".join(current))
            current, size = [header, separator], head_size
        current.append(row)
        size += len(row)
    segments.append("\n".join(current))

    return [Document(page_content=seg, metadata=dict(doc.metadata)) for seg in segments]


def split_documents(documents: list, chunk_size: int = None, chunk_overlap: int = None) -> list:
    """切分 Document 列表，并补充 chunk_index / chunk_id 元数据（便于溯源与去重）。"""
    splitter = build_splitter(chunk_size, chunk_overlap)
    if config.ENABLE_TABLE_AWARE_SPLIT:
        text_docs = [d for d in documents if d.metadata.get("type") != "table"]
        table_docs = [d for d in documents if d.metadata.get("type") == "table"]
    else:  # 关闭表格专用切分时，表格与正文一起走通用切分器
        text_docs, table_docs = documents, []

    chunks = splitter.split_documents(text_docs)
    for doc in table_docs:
        chunks.extend(_split_table_document(doc))

    injected = 0
    for idx, chunk in enumerate(chunks):
        # 工单02 优化：注入章节标题
        if config.INJECT_SECTION_TITLE:
            section = chunk.metadata.get("section")
            if section and not chunk.page_content.startswith(section):
                chunk.page_content = f"{section}\n{chunk.page_content}"
                injected += 1
        chunk.metadata["chunk_index"] = idx
        # 统一 chunk_id：文件名_页码_序号，作为向量库主键
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata.get('source', 'doc')}"
            f"_{chunk.metadata.get('page', 0)}"
            f"_{chunk.metadata.get('chunk_index', 0)}"
        )
    logger.info(
        "文本切分完成：正文 %d 篇 + 表格 %d 个 → %d 个片段"
        "（chunk_size=%d, 表格段上限=%d, 章节标题注入 %d 个）",
        len(text_docs), len(table_docs), len(chunks), chunk_size or config.CHUNK_SIZE,
        config.TABLE_CHUNK_MAX_CHARS, injected,
    )
    return chunks
