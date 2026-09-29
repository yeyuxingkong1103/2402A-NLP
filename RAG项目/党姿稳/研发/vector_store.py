"""
vector_store.py — 向量库统一抽象

long_term.py（长期记忆）和 knowledge_base.py（知识库）都依赖向量存储，
两者共用本模块的接口，通过 config.LOCAL_MODE 选择实现：

    LOCAL_MODE=True   local_store.LocalStore   numpy 内存索引 + JSON 持久化
    LOCAL_MODE=False  milvus_store.MilvusStore Milvus 集合

两套实现的对外行为一致（都返回按相似度降序的记录），上层业务无需关心底层。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import config

# 两个集合的字段定义，严格对应需求文档。
# 注意：milvus_store 的 schema 是 enable_dynamic_field=False，字段表与 schema
# 必须成对修改，漏一处该字段就会被静默丢弃。
KB_FIELDS = [
    "id", "text", "summary", "source", "page",
    "chunk_type", "parent_id", "created_at", "updated_at",
]
MEMORY_FIELDS = [
    "id", "text", "summary", "source", "user_id", "domain",
    "role", "fact_type", "created_at", "updated_at",
]

# 写入时允许为 None、需要补成空串的 VARCHAR 字段
_TEXT_FIELDS = {"text", "summary", "source", "chunk_type", "user_id", "domain", "role", "fact_type"}


class VectorStore(ABC):
    """向量库接口。filters 统一用字典表达，如 {"user_id": "u1"}。"""

    @abstractmethod
    def upsert(self, collection: str, records: list[dict]) -> list[int]:
        """写入记录（每条需含 embedding 字段），返回写入记录的主键列表。

        返回主键而非条数，是因为父子块入库时需要用父块主键回填子块的 parent_id。
        """

    @abstractmethod
    def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """向量检索，返回按相似度降序的记录，每条带 score 字段。"""

    @abstractmethod
    def query(
        self,
        collection: str,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """按条件拉取记录（不做向量检索），用于列举历史和构造 BM25 索引。"""

    @abstractmethod
    def delete(self, collection: str, filters: dict[str, Any]) -> int:
        """按条件删除记录，返回删除条数。"""

    @abstractmethod
    def count(self, collection: str) -> int:
        """统计记录条数。"""

    @abstractmethod
    def drop(self, collection: str) -> None:
        """删除整个集合。"""

    def list_collections(self) -> list[str]:
        """列出所有集合名。"""
        return []

    def close(self) -> None:
        """释放连接，默认无操作。"""


# ---------------------------------------------------------------- 工厂

_store: VectorStore | None = None


def get_store(force_local: bool = False) -> VectorStore:
    """返回全局向量库单例。

    LOCAL_MODE=False 时 Milvus 连不上就直接报错，**不再静默降级**到本地实现
    ——降级会让人误以为数据写进了 Milvus，实际落在本地 JSON 里，
    对"打开 Milvus 页面能看到数据"这种验收是致命的。
    """
    global _store
    if _store is not None:
        return _store

    if config.LOCAL_MODE or force_local:
        from local_store import LocalStore

        _store = LocalStore()
        return _store

    from milvus_store import MilvusStore

    try:
        _store = MilvusStore()
    except Exception as exc:
        raise RuntimeError(
            f"Milvus 连接失败（{exc}）。当前 LOCAL_MODE=False，数据必须写入 Milvus，"
            f"请确认 {config.MILVUS_URI} 可访问；本地调试可把 LOCAL_MODE 改回 True。"
        ) from exc
    return _store


def reset_store() -> None:
    """清空单例，主要用于测试。"""
    global _store
    if _store is not None:
        _store.close()
    _store = None


if __name__ == "__main__":
    import embeddings

    store = get_store()
    print(f"后端类型   : {type(store).__name__}")

    store.drop("_selftest")
    rows = [
        {"text": "劳动合同解除需要支付经济补偿", "source": "labour.pdf", "page": 3},
        {"text": "今天天气很好适合出门散步", "source": "life.pdf", "page": 1},
    ]
    for row in rows:
        row["embedding"] = embeddings.encode_query(row["text"])
    store.upsert("_selftest", rows)

    hits = store.search("_selftest", embeddings.encode_query("劳动合同解除赔偿"), top_k=2)
    for hit in hits:
        print(f"  {hit['score']:.4f}  page={hit.get('page')}  {hit['text']}")

    assert hits and "劳动" in hits[0]["text"], "检索结果不正确"
    assert hits[0].get("page") == 3, "page 字段没有正确落库"
    store.drop("_selftest")
    print("vector_store 自检通过。")
