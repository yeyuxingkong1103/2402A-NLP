"""Milvus 向量库：多域知识块存储 + 混合检索（dense + BM25）。

- schema 字段：id / vector / sparse(BM25) / text / summary / source / domain /
  doc_id / chunk_type / parser / create_time / update_time
- 混合检索优先用 Milvus 2.5+ 原生 BM25 sparse + RRFRanker；
  建表或检索失败时自动回退为 dense 检索（配合上层 rank_bm25 融合）。
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Optional

from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    RRFRanker,
    connections,
    utility,
)

from app.config import settings
from app.logging_conf import log

_OUTPUT_FIELDS = [
    "text", "parent", "summary", "source", "domain", "doc_id", "chunk_type", "parser",
    "create_time", "update_time",
]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MilvusStore:
    def __init__(self) -> None:
        self._col: Optional[Collection] = None
        self._connected = False
        self.native_bm25 = settings.milvus_native_bm25

    # ===== 连接与建表 =====
    def _connect(self) -> None:
        if not self._connected:
            connections.connect(host=settings.milvus_host, port=settings.milvus_port)
            self._connected = True

    def _build_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=settings.embed_dim),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("parent", DataType.VARCHAR, max_length=65535),
            FieldSchema("summary", DataType.VARCHAR, max_length=4096),
            FieldSchema("source", DataType.VARCHAR, max_length=1024),
            FieldSchema("domain", DataType.VARCHAR, max_length=64),
            FieldSchema("doc_id", DataType.VARCHAR, max_length=64),
            FieldSchema("chunk_type", DataType.VARCHAR, max_length=32),
            FieldSchema("parser", DataType.VARCHAR, max_length=32),
            FieldSchema("create_time", DataType.VARCHAR, max_length=32),
            FieldSchema("update_time", DataType.VARCHAR, max_length=32),
        ]
        if self.native_bm25:
            fields.append(FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR))
        schema = CollectionSchema(fields, description="RAG 多角色知识库")
        if self.native_bm25:
            from pymilvus import Function, FunctionType

            schema.add_function(
                Function(
                    name="bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names=["sparse"],
                )
            )
        return schema

    def _create_indexes(self, col: Collection) -> None:
        col.create_index(
            "vector",
            {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}},
        )
        if self.native_bm25:
            col.create_index(
                "sparse",
                {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"},
            )

    def collection(self) -> Collection:
        self._connect()
        if self._col is not None:
            return self._col
        if utility.has_collection(settings.milvus_collection):
            col = Collection(settings.milvus_collection)
        else:
            col = self._create_collection()
        col.load()
        self._col = col
        return col

    def _create_collection(self) -> Collection:
        try:
            col = Collection(settings.milvus_collection, schema=self._build_schema())
        except Exception as exc:  # noqa: BLE001
            log.warning("原生 BM25 建表失败，回退为 dense-only: %s", exc)
            self.native_bm25 = False
            if utility.has_collection(settings.milvus_collection):
                utility.drop_collection(settings.milvus_collection)
            col = Collection(settings.milvus_collection, schema=self._build_schema())
        self._create_indexes(col)
        log.info("已创建 collection: %s (native_bm25=%s)", settings.milvus_collection, self.native_bm25)
        return col

    # ===== 写入 =====
    def insert(self, records: list[dict]) -> int:
        if not records:
            return 0
        col = self.collection()
        now = _now()
        vectors = [r["vector"] for r in records]
        data = [
            vectors,
            [r["text"][:65535] for r in records],
            [r.get("parent", "")[:65535] for r in records],
            [r.get("summary", "")[:4096] for r in records],
            [r.get("source", "")[:1024] for r in records],
            [r.get("domain", "general")[:64] for r in records],
            [r.get("doc_id", "")[:64] for r in records],
            [r.get("chunk_type", "")[:32] for r in records],
            [r.get("parser", "")[:32] for r in records],
            [now] * len(records),
            [now] * len(records),
        ]
        col.insert(data)
        col.flush()
        log.info("入库 %d 条到 %s", len(records), settings.milvus_collection)
        return len(records)

    def delete_by_doc(self, doc_id: str) -> None:
        col = self.collection()
        col.delete(f'doc_id == "{doc_id}"')
        col.flush()

    def drop(self) -> None:
        self._connect()
        if utility.has_collection(settings.milvus_collection):
            utility.drop_collection(settings.milvus_collection)
            self._col = None
            log.info("已删除 collection: %s", settings.milvus_collection)

    # ===== 检索 =====
    @staticmethod
    def _domain_expr(domain: Optional[str]) -> Optional[str]:
        return f'domain == "{domain}"' if domain else None

    @staticmethod
    def _hit_to_dict(hit) -> dict:
        e = hit.entity
        return {
            "id": hit.id,
            "text": e.get("text"),
            "parent": e.get("parent") or "",
            "summary": e.get("summary") or "",
            "source": e.get("source"),
            "domain": e.get("domain") or "general",
            "doc_id": e.get("doc_id") or "",
            "chunk_type": e.get("chunk_type") or "",
            "parser": e.get("parser") or "",
            "score": float(hit.score) if hit.score is not None else 0.0,
        }

    def search_dense(self, vector: list[float], top_k: int = 10, domain: Optional[str] = None) -> list[dict]:
        col = self.collection()
        res = col.search(
            [vector], "vector",
            {"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k, expr=self._domain_expr(domain), output_fields=_OUTPUT_FIELDS,
        )
        return [self._hit_to_dict(h) for h in res[0]]

    def search_hybrid(
        self, query: str, vector: list[float], top_k: int = 10, domain: Optional[str] = None
    ) -> list[dict]:
        if not self.native_bm25:
            return self.search_dense(vector, top_k, domain)
        try:
            col = self.collection()
            expr = self._domain_expr(domain)
            dense_req = AnnSearchRequest(
                data=[vector], anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=top_k, expr=expr,
            )
            sparse_req = AnnSearchRequest(
                data=[query], anns_field="sparse",
                param={"metric_type": "BM25"}, limit=top_k, expr=expr,
            )
            res = col.hybrid_search(
                [dense_req, sparse_req], rerank=RRFRanker(), limit=top_k, output_fields=_OUTPUT_FIELDS
            )
            return [self._hit_to_dict(h) for h in res[0]]
        except Exception as exc:  # noqa: BLE001
            log.warning("原生混合检索失败，回退 dense: %s", exc)
            return self.search_dense(vector, top_k, domain)

    def get_by_ids(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        col = self.collection()
        expr = "id in [" + ",".join(str(i) for i in ids) + "]"
        return col.query(expr=expr, output_fields=["id"] + _OUTPUT_FIELDS, limit=len(ids))

    # ===== 运维可视化 =====
    def all_entities(self, domain: Optional[str] = None, limit: int = 16384) -> list[dict]:
        col = self.collection()
        expr = self._domain_expr(domain) or "id > 0"
        return col.query(expr=expr, output_fields=["id"] + _OUTPUT_FIELDS, limit=limit)

    def page_entities(self, domain: Optional[str] = None, limit: int = 50, offset: int = 0) -> list[dict]:
        col = self.collection()
        expr = self._domain_expr(domain) or "id > 0"
        return col.query(expr=expr, output_fields=["id"] + _OUTPUT_FIELDS, limit=limit, offset=offset)

    def count(self) -> int:
        try:
            return self.collection().num_entities
        except Exception:  # noqa: BLE001
            return 0

    def stats(self) -> dict:
        col = self.collection()
        rows = col.query(expr="id > 0", output_fields=["domain", "parser"], limit=16384)
        domains: Counter = Counter(r.get("domain") or "general" for r in rows)
        parsers: Counter = Counter(r.get("parser") or "unknown" for r in rows)
        return {
            "collection": settings.milvus_collection,
            "num_entities": col.num_entities,
            "domains": dict(domains),
            "parsers": dict(parsers),
        }


milvus_store = MilvusStore()
