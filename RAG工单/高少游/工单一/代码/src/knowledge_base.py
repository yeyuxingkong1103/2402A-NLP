# -*- coding: utf-8 -*-
"""知识库模块：切片、向量化与向量数据库管理
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

功能：
- 把《招股说明书1.pdf》解析结果切分为可检索的文本块；
- 使用本地 BGE-M3 向量模型把文本块嵌入向量空间；
- 使用 FAISS 向量索引持久化与管理文档；支持构建/加载/清洗/重建。

技术选型（对应 langchain2 环境）：
- 向量化：langchain-ollama 的 OllamaEmbeddings（本地 bge-m3，多语言强，分批嵌入）
- 向量索引：langchain-community 的 FAISS（本地单文件持久化，轻量、稳定）
- 切片：langchain-text-splitters 的 RecursiveCharacterTextSplitter
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.pdf_parser import PageDoc, parse_pdf

logger = logging.getLogger(__name__)


class _BatchedEmbeddings(OllamaEmbeddings):
    """分批向量化包装器：避免一次把全库文本塞进一个 /api/embed 请求。

    本地 bge-m3 在超大批量时会触发 Ollama tokenizer 服务（127.0.0.1:1510）
    连接被拒，故改为小批量逐段调用，兼容长文档向量库构建。
    """

    batch_size: int = 50

    def embed_documents(self, texts):
        result = []
        for i in range(0, len(texts), self.batch_size):
            result.extend(super().embed_documents(texts[i:i + self.batch_size]))
        return result


def get_embeddings() -> _BatchedEmbeddings:
    """构造本地向量模型（分批执行，兼容大批量构建）。"""
    return _BatchedEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        batch_size=int(os.getenv("EMBED_BATCH_SIZE", "50")),
    )


def build_documents(
    pdf_path: str | Path = config.PDF_PATH,
) -> List[Document]:
    """解析 PDF 并按页切片，生成带元数据的文档块。"""
    pages: List[PageDoc] = parse_pdf(pdf_path, with_tables=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        keep_separator=True,
    )
    documents: List[Document] = []
    for pg in pages:
        content = pg.content
        if not content:
            continue
        chunks = splitter.split_text(content)
        for idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": str(pdf_path),
                        "page": pg.page_no,
                        "chunk": idx,
                    },
                )
            )
    return documents


def _kv_file(persist_directory: str | Path) -> Path:
    # save_local 实际写出的文件名是 index.faiss
    return Path(persist_directory) / "index.faiss"


def build_kb(
    pdf_path: str | Path = config.PDF_PATH,
    persist_directory: str | Path = config.DB_DIR,
    force_rebuild: bool = False,
):
    """构建（或重建）向量知识库。

    Args:
        pdf_path: PDF 源文档路径。
        persist_directory: FAISS 持久化目录。
        force_rebuild: 为 True 时先清空已有向量库再重建。
    """
    from langchain_community.vectorstores import FAISS

    persist_directory = Path(persist_directory)
    if force_rebuild:
        clear_kb(persist_directory)

    if not _kv_file(persist_directory).exists():
        persist_directory.mkdir(parents=True, exist_ok=True)
        docs = build_documents(pdf_path)
        if not docs:
            raise RuntimeError(f"PDF 解析后未得到任何文档块: {pdf_path}")
        logger.info("embedded %d chunks ...", len(docs))
        store = FAISS.from_documents(docs, get_embeddings())
        store.save_local(str(persist_directory))
        return store
    return load_kb(persist_directory)


def load_kb(persist_directory: str | Path = config.DB_DIR):
    """加载已构建的向量知识库。"""
    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(
        str(persist_directory),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def clear_kb(persist_directory: str | Path = config.DB_DIR) -> None:
    """清空向量库目录。"""
    import shutil

    d = Path(persist_directory)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)


def kb_stats(persist_directory: str | Path = config.DB_DIR) -> dict:
    """返回知识库统计信息（用于知识库管理）。"""
    try:
        store = load_kb(persist_directory)
        return {"name": "faiss", "chunks": int(store.index.ntotal)}
    except Exception as exc:  # 未构建或不可用
        return {"name": "faiss", "chunks": 0, "error": str(exc)}


def get_all_documents(persist_directory: str | Path = config.DB_DIR) -> List[Document]:
    """取回知识库中的全部文档块（供 BM25 词法检索做全局语料）。"""
    store = load_kb(persist_directory)
    idx_to_doc = getattr(store, "index_to_docstore_id", {}) or {}
    ds = getattr(store, "docstore", None)
    items = getattr(ds, "_dict", {}) if ds is not None else {}
    docs: List[Document] = []
    seen: set = set()
    for doc_id in idx_to_doc.values():
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc = items.get(doc_id)
        if doc is not None:
            docs.append(doc)
    return docs