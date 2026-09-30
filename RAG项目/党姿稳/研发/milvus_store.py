"""
milvus_store.py — 向量库的 Milvus 实现

LOCAL_MODE=False 时由 vector_store.get_store() 实例化。
集合 schema 见 _build_schema，字段与 vector_store 里的 KB_FIELDS / MEMORY_FIELDS
一一对应；因为 schema 关掉了动态字段，**两边必须成对修改**，
否则多出来的字段会被 _normalize 静默丢掉。
"""

from __future__ import annotations

import time
from typing import Any

import config
from vector_store import KB_FIELDS, MEMORY_FIELDS, VectorStore, _TEXT_FIELDS

# Milvus 规定 offset + limit 必须落在 [1, 16384] 内，超出会报 code=65535
_MAX_QUERY_WINDOW = 16384


def _build_schema(client, memory: bool, dim: int):
    """构造 Milvus 集合 schema，字段对应需求文档 4.5 节的两张表。"""
    from pymilvus import DataType

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("summary", DataType.VARCHAR, max_length=2048)
    schema.add_field("source", DataType.VARCHAR, max_length=512)

    if memory:
        # domain 让长期记忆能按"同一用户 + 同一领域"过滤，而不只是按用户
        schema.add_field("domain", DataType.VARCHAR, max_length=64)
        schema.add_field("user_id", DataType.VARCHAR, max_length=128)
        schema.add_field("role", DataType.VARCHAR, max_length=64)
        schema.add_field("fact_type", DataType.VARCHAR, max_length=64)
    else:
        # page 是独立的页码字段，供回答末尾生成"文件名 + 第几页"的引用
        schema.add_field("page", DataType.INT64)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=64)
        schema.add_field("parent_id", DataType.INT64)

    schema.add_field("created_at", DataType.INT64)
    schema.add_field("updated_at", DataType.INT64)
    return schema


class MilvusStore(VectorStore):
    """Milvus 实现。"""

    def __init__(self, uri: str = config.MILVUS_URI, token: str = config.MILVUS_TOKEN):
        from pymilvus import MilvusClient

        self.client = MilvusClient(uri=uri, token=token or None)
        self.dim = config.EMBEDDING_DIM

    def _fields(self, collection: str) -> list[str]:
        return MEMORY_FIELDS if collection == config.LONG_TERM_COLLECTION else KB_FIELDS

    def ensure_collection(self, collection: str) -> None:
        """按需创建集合（幂等），供初始化脚本与写入路径调用。"""
        self._ensure(collection)

    def _ensure(self, collection: str) -> None:
        if self.client.has_collection(collection):
            return

        memory = collection == config.LONG_TERM_COLLECTION
        schema = _build_schema(self.client, memory, self.dim)

        # 只给向量字段建索引。主键在 Milvus 里自带索引，再建一次是多余的。
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(
            collection_name=collection, schema=schema, index_params=index_params
        )

    @staticmethod
    def _to_expr(filters: dict[str, Any] | None) -> str:
        """把字典条件转成 Milvus 过滤表达式。"""
        if not filters:
            return ""
        parts = []
        for key, value in filters.items():
            if isinstance(value, str):
                parts.append(f'{key} == "{value.replace(chr(34), chr(92) + chr(34))}"')
            else:
                parts.append(f"{key} == {value}")
        return " and ".join(parts)

    def _normalize(self, collection: str, record: dict) -> dict:
        """裁剪为 schema 允许的字段，并把 None 补成中性值。"""
        allowed = set(self._fields(collection)) | {"embedding"}
        row = {k: v for k, v in record.items() if k in allowed}
        now = int(time.time())
        row.setdefault("created_at", now)
        row.setdefault("updated_at", now)
        for field in _TEXT_FIELDS & allowed:
            if not row.get(field):
                row[field] = ""
        if "parent_id" in allowed and row.get("parent_id") is None:
            row["parent_id"] = 0
        # 页码未知时记 0，前端据此不显示"第 0 页"
        if "page" in allowed and row.get("page") is None:
            row["page"] = 0
        return row

    def upsert(self, collection: str, records: list[dict]) -> list[int]:
        """写入记录。Milvus 的 upsert 在 auto_id=True 下不可用，这里用 insert。

        因此重复导入同一文件会产生重复行 —— 调用方（knowledge_base.import_pdf）
        在写入前按 source 删掉旧记录来保证幂等。
        """
        if not records:
            return []
        self._ensure(collection)
        rows = [self._normalize(collection, r) for r in records]
        result = self.client.insert(collection_name=collection, data=rows)
        # 必须 flush：否则数据留在内存 segment 里没落盘，
        # get_collection_stats 会返回 0，Milvus 页面和 /kb/stats 都显示成空集合。
        self.client.flush(collection)
        return [int(i) for i in (result.get("ids") or [])]

    def search(self, collection, vector, top_k=10, filters=None) -> list[dict]:
        if not self.client.has_collection(collection):
            return []

        results = self.client.search(
            collection_name=collection,
            data=[vector],
            limit=top_k,
            filter=self._to_expr(filters),
            output_fields=self._fields(collection),
            search_params={"metric_type": "COSINE"},
        )

        hits: list[dict] = []
        for item in results[0] if results else []:
            entity = dict(item.get("entity") or {})
            entity["id"] = item.get("id")
            entity["score"] = float(item.get("distance", 0.0))
            hits.append(entity)
        return hits

    def query(self, collection, filters=None, limit=100, offset=0) -> list[dict]:
        if not self.client.has_collection(collection):
            return []

        expr = self._to_expr(filters)
        fields = self._fields(collection)
        rows: list[dict] = []

        # Milvus 对单次查询的 (offset + limit) 有硬上限，超了直接抛 MilvusException。
        # 调用方常传一个"取全部"的大 limit（如知识库统计），这里在窗口内分批拉取。
        while len(rows) < limit:
            batch = min(_MAX_QUERY_WINDOW - offset, limit - len(rows))
            if batch <= 0:
                break
            chunk = self.client.query(
                collection_name=collection,
                filter=expr,
                output_fields=fields,
                limit=batch,
                offset=offset,
            )
            if not chunk:
                break
            rows.extend(dict(row) for row in chunk)
            offset += len(chunk)
            if len(chunk) < batch:
                break
        return rows

    def delete(self, collection, filters) -> int:
        if not self.client.has_collection(collection):
            return 0
        expr = self._to_expr(filters)
        if not expr:
            raise ValueError("delete 必须提供过滤条件，避免误删整个集合")
        result = self.client.delete(collection_name=collection, filter=expr)
        return int(result.get("delete_count", 0)) if isinstance(result, dict) else 0

    def count(self, collection) -> int:
        if not self.client.has_collection(collection):
            return 0
        return int(self.client.get_collection_stats(collection).get("row_count", 0))

    def drop(self, collection) -> None:
        if self.client.has_collection(collection):
            self.client.drop_collection(collection)

    def list_collections(self) -> list[str]:
        return list(self.client.list_collections())

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
