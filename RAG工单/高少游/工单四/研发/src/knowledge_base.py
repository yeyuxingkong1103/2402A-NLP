# -*- coding: utf-8 -*-
"""知识库模块（图像内容解析及检索优化版）：向量化与向量数据库管理
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

- 向量化：langchain-ollama 的 OllamaEmbeddings（本地 bge-m3，分批嵌入）
- 向量索引：langchain-community 的 FAISS（本地单文件持久化）
- 多文档：知识库同时收录《招股说明书1.pdf》《招股说明书2.pdf》
- 切片来源：src.chunking.build_chunks（结构感知 + 表格结构化双块 + 父子块）
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


def build_documents(pdf_paths=None) -> List[Document]:
    """解析多份 PDF → 结构感知分块 → Document 列表。

    【本工单新增】在文本/表格块基础上并入「图像语义块」（ctype="figure"），
    使图形内容（组织结构图、统计图）与正文、表格一起进入同一检索链路。
    """
    chunks = build_chunks(pdf_paths)
    if config.USE_FIGURE_PARSER:
        try:
            from src.image_index import build_figure_chunks

            fig_chunks = build_figure_chunks(pdf_paths)
            # 重新编号，避免与文本块索引冲突
            base = len(chunks)
            for i, c in enumerate(fig_chunks):
                c.index = base + i
            chunks = chunks + fig_chunks
            logger.info("并入图像语义块 %d 个", len(fig_chunks))
        except Exception as exc:                 # 容错：图形解析失败不阻断文本链路
            logger.warning("图像语义块构建失败（仅使用文本/表格块）: %s", exc)
    return chunks_to_documents(chunks)


def build_baseline_documents(pdf_paths=None) -> List[Document]:
    """【优化前】基线分块 → Document 列表（表格拍平，无结构化解析）。"""
    from src.chunking import build_baseline_chunks

    return chunks_to_documents(build_baseline_chunks(pdf_paths))


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
    含中文的路径（本项目路径含“工单代码/工单三”）。故当路径非 ASCII 时，
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


def build_kb(pdf_paths=None, persist_directory=config.DB_DIR,
             force_rebuild: bool = False, baseline: bool = False,
             docs: List[Document] | None = None):
    """构建（或重建）向量知识库。

    Args:
        baseline: 为 True 时使用基线分块（复刻 01/02 工单），否则使用优化分块。
        docs: 可选的预分块文档（避免重复解析 PDF）。
    """
    from langchain_community.vectorstores import FAISS

    persist_directory = Path(persist_directory)
    if force_rebuild:
        clear_kb(persist_directory)

    if not _kv_file(persist_directory).exists():
        persist_directory.mkdir(parents=True, exist_ok=True)
        if docs is None:
            docs = (build_baseline_documents(pdf_paths) if baseline
                    else build_documents(pdf_paths))
        if not docs:
            raise RuntimeError("PDF 解析后未得到任何文档块")
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
        docs = get_all_documents(persist_directory)
        return {
            "name": "faiss",
            "chunks": int(store.index.ntotal),
            "table_chunks": sum(1 for d in docs if d.metadata.get("ctype") == "table"),
            "table_kv_chunks": sum(1 for d in docs if d.metadata.get("ctype") == "table_kv"),
            "sources": sorted({d.metadata.get("source", "") for d in docs if d.metadata.get("source")}),
        }
    except Exception as exc:
        return {"name": "faiss", "chunks": 0, "error": str(exc)}