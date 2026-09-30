# -*- coding: utf-8 -*-
"""离线知识入库链路：``PDF/文本 -> 解析 -> 分块 -> 向量化 -> Milvus``。"""
from .chunker import build_chunks, chunk_text
from .loaders import Document, iter_document_paths, load_document
from .pdf_fixture import build_chinese_pdf, ensure_sample_pdf, write_chinese_pdf
from .pipeline import IngestReport, KnowledgeBasePipeline

__all__ = [
    "Document",
    "load_document",
    "iter_document_paths",
    "chunk_text",
    "build_chunks",
    "IngestReport",
    "KnowledgeBasePipeline",
    "build_chinese_pdf",
    "write_chinese_pdf",
    "ensure_sample_pdf",
]
