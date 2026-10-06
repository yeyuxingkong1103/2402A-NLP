# -*- coding: utf-8 -*-
"""知识库模块（优化版）：向量化与向量数据库管理
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

- 向量化：langchain-ollama 的 OllamaEmbeddings（本地 bge-m3，分批嵌入）
- 向量索引：langchain-community 的 FAISS（本地单文件持久化）
- 切片来源：src.chunking.build_chunks（结构感知 + 表格原子块 + 父子块）
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src import config
from src.chunking import build_chunks, chunks_to_documents

logger = logging.getLogger(__name__)


class _BatchedEmbeddings(OllamaEmbeddings):
    """分批向量化包装器：避免一次把全库文本塞进单个 /api/embed 请求。"""

    batch_size: int = 50

    def embed_documents(self, texts):
        result = []
        for i in range(0, len(texts), self.batch_size):
            result.extend(super().embed_documents(texts[i:i + self.batch_size]))
        return result


def get_embeddings() -> _BatchedEmbeddings:
    return _BatchedEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        batch_size=int(os.getenv("EMBED_BATCH_SIZE", "50")),
    )


def build_documents(pdf_path=config.PDF_PATH) -> List[Document]:
    """解析 PDF → 结构感知分块 → Document 列表。"""
    return chunks_to_documents(build_chunks(pdf_path))


def build_baseline_documents(pdf_path=config.PDF_PATH) -> List[Document]:
    """【优化前】基线分块 → Document 列表（固定长度切片，无结构感知）。"""
    from src.chunking import build_baseline_chunks

    return chunks_to_documents(build_baseline_chunks(pdf_path))


def _kv_file(persist_directory: str | Path) -> Path:
    return Path(persist_directory) / "index.faiss"


def _is_ascii(path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def save_local(store, persist_directory: str | Path) -> None:
    """保存 FAISS 索引。

    兼容性说明：Windows 版 faiss 的 C++ 层用窄字符 API 打开文件，无法处理
    含中文的路径（本项目路径含“工单代码/工单二”）。故当路径非 ASCII 时，
    先写入临时 ASCII 目录，再拷贝回目标目录，保证持久化正常。
    """
    path = Path(persist_directory)
    path.mkdir(parents=True, exist_ok=True)
    if _is_ascii(path):
        store.save_local(str(path))
        return
    with tempfile.TemporaryDirectory() as td:
        store.save_local(td)
        for name in os.listdir(td):
            shutil.copyfile(Path(td) / name, path / name)


def load_local(persist_directory: str | Path):
    """加载 FAISS 索引（同样做非 ASCII 路径桥接）。"""
    from langchain_community.vectorstores import FAISS

    path = Path(persist_directory)
    if _is_ascii(path):
        return FAISS.load_local(str(path), get_embeddings(),
                                allow_dangerous_deserialization=True)
    with tempfile.TemporaryDirectory() as td:
        for name in os.listdir(path):
            shutil.copyfile(path / name, Path(td) / name)
        return FAISS.load_local(td, get_embeddings(),
                                allow_dangerous_deserialization=True)


def build_kb(pdf_path=config.PDF_PATH, persist_directory=config.DB_DIR,
             force_rebuild: bool = False, baseline: bool = False):
    """构建（或重建）向量知识库。

    Args:
        baseline: 为 True 时使用基线分块（复刻 01 工单），否则使用优化分块。
    """
    from langchain_community.vectorstores import FAISS

    persist_directory = Path(persist_directory)
    if force_rebuild:
        clear_kb(persist_directory)

    if not _kv_file(persist_directory).exists():
        persist_directory.mkdir(parents=True, exist_ok=True)
        docs = build_baseline_documents(pdf_path) if baseline else build_documents(pdf_path)
        if not docs:
            raise RuntimeError(f"PDF 解析后未得到任何文档块: {pdf_path}")
        logger.info("embedded %d chunks ...", len(docs))
        store = FAISS.from_documents(docs, get_embeddings())
        save_local(store, persist_directory)
        return store
    return load_kb(persist_directory)


def load_kb(persist_directory=config.DB_DIR):
    return load_local(persist_directory)


def clear_kb(persist_directory=config.DB_DIR) -> None:
    d = Path(persist_directory)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)


def get_all_documents(persist_directory=config.DB_DIR) -> List[Document]:
    """取回知识库全部文档块（供 BM25 词法检索做全局语料）。"""
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


def kb_stats(persist_directory=config.DB_DIR) -> dict:
    try:
        store = load_kb(persist_directory)
        return {"name": "faiss", "chunks": int(store.index.ntotal)}
    except Exception as exc:
        return {"name": "faiss", "chunks": 0, "error": str(exc)}