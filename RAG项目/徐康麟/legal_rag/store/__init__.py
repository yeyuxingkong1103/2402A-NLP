# -*- coding: utf-8 -*-
"""向量库后端。

* :func:`build_store` —— 按名字构建（memory | chroma | milvus）；
* Milvus 相关符号在这里统一导出，方便 ``from legal_rag.store import MilvusVectorStore``。
"""
from .base import VectorStore, build_store
from .memory_store import MemoryVectorStore
from .milvus_store import (
    CHUNK_SCHEMA_VERSION,
    MANIFEST_NAME,
    MilvusError,
    MilvusUnavailableError,
    MilvusVectorStore,
    VectorDimMismatchError,
    build_dim_mismatch_message,
    build_milvus_store,
)

__all__ = [
    "VectorStore", "build_store", "MemoryVectorStore",
    "MilvusVectorStore", "build_milvus_store", "MilvusError",
    "MilvusUnavailableError", "VectorDimMismatchError",
    "build_dim_mismatch_message", "MANIFEST_NAME", "CHUNK_SCHEMA_VERSION",
]
