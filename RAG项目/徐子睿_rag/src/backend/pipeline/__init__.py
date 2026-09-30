# -*- coding: utf-8 -*-
"""pipeline —— PDF 构建管线（同名包）。

在链路中的位置（离线侧）：
    上传的 PDF → 【本包】 → Milvus 集合 rag_docs_v6 → backend/retrieval.py 检索
    调用方是 backend/server.py 的后台构建线程。

本包解决的问题：一份 PDF 怎么变成"既能被检索、又能说清出处"的向量数据。

为什么改成了包：
    原 pipeline.py 有 524 行（其中注释 208 行），超出"单文件 300 行"的上限。
    按职责拆成 config / pdf_parse / text_clean / chunking / vectors / build 六个模块后，
    每个文件都在 300 行以内。

关键设计：本文件把各子模块的公开名全部再导出，
    所以 `from pipeline import build_pipeline, parse_with_mineru, chunk_pages` 这类用法
    与拆分前完全一致 —— server / retrieval / roleplay / tests / src.offline.parsers
    一行都不用改。

包内分工：
    config.py      路径与调参常量（BASE_DIR / PDF_DIR / MAX_CHUNK / OVERLAP / 三个正则）
    pdf_parse.py   PDF 解析：MinerU 优先、pypdf 兜底
    text_clean.py  术语归一化与噪声过滤
    chunking.py    分块
    vectors.py     向量化与写入 Milvus
    build.py       build_pipeline 总编排
    本文件          再导出 + 两个兼容旧调用方的壳函数
"""
from __future__ import annotations

try:
    from ..vector_store import COLLECTION, VECTOR_DIMENSION, and_filter, delete_vectors, ensure_collection, equal_filter, upsert_vectors
except ImportError:
    from vector_store import COLLECTION, VECTOR_DIMENSION, and_filter, delete_vectors, ensure_collection, equal_filter, upsert_vectors

from .build import build_pipeline
from .chunking import chunk_pages
from .config import BASE_DIR, EMBED_MODEL, HEADER_RE, HEADING_RE, MAX_CHUNK, OLLAMA_BASE, OVERLAP, PDF_DIR, TOC_RE
from .pdf_parse import mineru_available, mineru_pages, parse_pdf, parse_with_mineru, parse_with_pypdf, table_to_text
from .text_clean import clean_text, is_noise, keep_body_pages, normalize_text
from .vectors import delete_document_vectors, embed, store_chunks


# ------------------------------ 客户端（兼容旧调用方）


def milvus_client():
    """兼容旧调用方，返回 Milvus 客户端并确保知识库集合存在。

    返回：
        ensure_collection 得到的 MilvusClient，集合与索引已就绪、已 load。
    说明：
        这是历史接口的保留壳，新代码直接调 vector_store.ensure_collection。
    """
    return ensure_collection(COLLECTION)


def close_milvus_client() -> None:
    """关闭 Milvus 客户端连接，释放资源。

    说明：
        本函数由原 pipeline.py 的版本**原样搬迁**，只把相对导入上调一级
        （`.vector_store` -> `..vector_store`）—— 结构上仍保留 try/except 双导入，
        以便作为顶层包导入时也能回退到 `vector_store`。
    """
    try:
        from ..vector_store import close_milvus_client as close_client
    except ImportError:
        from vector_store import close_milvus_client as close_client

    close_client()


# 显式声明对外接口：这就是"拆包不改调用方"的契约清单。
# 除了本包自己的函数与常量，这里还刻意保留了原先"从 pipeline 命名空间能取到"的
# vector_store 名字（拆分前 pipeline.py 顶层 import 过它们，所以 `from pipeline import
# equal_filter` 是可用的）—— 一并再导出，避免任何一处隐式依赖被拆断。
__all__ = [
    "BASE_DIR",
    "COLLECTION",
    "EMBED_MODEL",
    "HEADER_RE",
    "HEADING_RE",
    "MAX_CHUNK",
    "OLLAMA_BASE",
    "OVERLAP",
    "PDF_DIR",
    "TOC_RE",
    "VECTOR_DIMENSION",
    "and_filter",
    "build_pipeline",
    "chunk_pages",
    "clean_text",
    "close_milvus_client",
    "delete_document_vectors",
    "delete_vectors",
    "embed",
    "ensure_collection",
    "equal_filter",
    "is_noise",
    "keep_body_pages",
    "milvus_client",
    "mineru_available",
    "mineru_pages",
    "normalize_text",
    "parse_pdf",
    "parse_with_mineru",
    "parse_with_pypdf",
    "store_chunks",
    "table_to_text",
    "upsert_vectors",
]
