# -*- coding: utf-8 -*-
"""vector_store —— Milvus 向量存储适配层（同名包）。

在链路中的位置（最底层）：
    backend/pipeline.py（写入） / backend/retrieval.py（读取） → 【本包】 → Milvus

为什么改成了包：
    原 vector_store.py 有 377 行（其中注释 169 行），超出"单文件 300 行"的上限。
    按职责拆成 client / filters / ops 三个模块后，每个文件都在 300 行以内。

关键设计：__init__.py 把三个子模块的公开名全部再导出，
    所以 `from vector_store import equal_filter, upsert_vectors` 这类用法
    与拆分前完全一致 —— 调用方（pipeline / retrieval / roleplay / tests / tools）
    一行都不用改。

包内分工：
    client.py   连接与集合生命周期：milvus_client / close_milvus_client /
                ensure_collection / milvus_health / collection_count
    filters.py  过滤表达式与记录整形：equal_filter / and_filter / _payload_from_row
    ops.py      记录级读写：query_vectors / search_vectors / upsert_vectors / delete_vectors

注意一处分担：
    VECTOR_DIMENSION 与 COLLECTION 这两个常量定义在 client.py（它们决定集合结构），
    由本文件再导出，因此 pipeline 里 `from ...vector_store import COLLECTION` 照旧可用。
"""
from __future__ import annotations

from .client import (
    COLLECTION,
    MILVUS_TOKEN,
    MILVUS_URI,
    VECTOR_DIMENSION,
    close_milvus_client,
    collection_count,
    ensure_collection,
    milvus_client,
    milvus_health,
)
from .filters import and_filter, equal_filter
from .ops import delete_vectors, query_vectors, search_vectors, upsert_vectors

# 显式声明对外接口：这就是"拆包不改调用方"的契约清单，
# 与拆分前 vector_store.py 里定义的公开名字一一对应
__all__ = [
    "COLLECTION",
    "MILVUS_TOKEN",
    "MILVUS_URI",
    "VECTOR_DIMENSION",
    "and_filter",
    "close_milvus_client",
    "collection_count",
    "delete_vectors",
    "ensure_collection",
    "equal_filter",
    "milvus_client",
    "milvus_health",
    "query_vectors",
    "search_vectors",
    "upsert_vectors",
]
