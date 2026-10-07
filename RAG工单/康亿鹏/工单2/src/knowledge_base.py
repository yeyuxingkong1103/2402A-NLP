# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：知识库管理模块。负责 PDF 解析 → 切分 → 向量化 → 入库的完整流程，
          以及知识库的统计与清空，对应“知识库管理”功能需求。
"""
from src.document_loader import load_pdf
from src.embeddings import get_embedding_model
from src.splitter import split_documents
from src.utils import logger, timer
from src.vector_store import get_vector_store


def ingest_pdf(pdf_path: str, drop_old: bool = True, enable_tables: bool = None,
               chunk_size: int = None, chunk_overlap: int = None) -> dict:
    """解析 PDF 并写入向量库，返回入库统计信息。"""
    stats = {"pdf": pdf_path, "pages_docs": 0, "chunks": 0, "inserted": 0, "elapsed": 0.0}

    with timer() as t:
        documents = load_pdf(pdf_path, enable_tables=enable_tables)
        stats["pages_docs"] = len(documents)

        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        stats["chunks"] = len(chunks)

        embedding = get_embedding_model()
        vectors = embedding.embed_documents([c.page_content for c in chunks])

        store = get_vector_store()
        store.create_collection(drop_old=drop_old)
        stats["inserted"] = store.insert(chunks, vectors)

    stats["elapsed"] = t["elapsed"]
    logger.info("知识库入库完成：%s，片段 %d 条，耗时 %.2fs", pdf_path, stats["inserted"], stats["elapsed"])
    return stats


def knowledge_base_stats() -> dict:
    """返回知识库统计信息。"""
    store = get_vector_store()
    return {
        "collection": store.collection,
        "uri": store.uri,
        "exists": store.has_collection(),
        "count": store.count(),
    }


def clear_knowledge_base():
    """清空知识库。"""
    get_vector_store().drop_collection()
