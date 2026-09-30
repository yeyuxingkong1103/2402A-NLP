"""Milvus 存储层：知识块集合 + 长期记忆集合。

两个集合都使用「稠密 + 稀疏」双向量字段：

* ``dense``：BGE-M3 CLS 向量，1024 维，COSINE，HNSW 索引；
* ``sparse``：BGE-M3 lexical 权重，SPARSE_FLOAT_VECTOR，IP，倒排索引；
* 过滤字段 ``role`` / ``scope`` 用于按角色做多租户隔离（shared 为公共知识）。

这样既可分别单路检索，也可走 Milvus 原生 ``hybrid_search`` + RRFRanker。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Iterable, Sequence

from ..config import Config, get_config
from ..errors import DependencyError
from ..logging_conf import get_logger

logger = get_logger(__name__)

KB_FIELDS = [
    "pk", "text", "role", "scope", "doc_id", "doc_title",
    "section", "chunk_index", "source", "tags", "char_len", "created_at",
]
MEMORY_FIELDS = ["pk", "text", "user_id", "role_id", "kind", "importance", "ts", "session_id"]


class MilvusStore:
    """Milvus 客户端封装（懒连接 + 自动建表）。"""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.uri = str(self.config.get("milvus.uri", "http://127.0.0.1:19530"))
        self.token = str(self.config.get("milvus.token", "") or "")
        self.kb_collection = str(self.config.get("milvus.kb_collection", "role_kb_chunks"))
        self.memory_collection = str(self.config.get("milvus.memory_collection", "role_memory"))
        self.dense_dim = int(self.config.get("models.embedder.dense_dim", 1024))
        self.search_timeout = int(self.config.get("milvus.search.timeout", 30))
        self._client: Any = None
        self._lock = threading.RLock()
        self._ready = False
        self._memory_rows_hint = (0.0, -1)   # (时间戳, 行数) 缓存，用于跳过空集合

    # ------------------------------------------------------------------ 连接
    @property
    def client(self) -> Any:
        if self._client is None:
            from pymilvus import MilvusClient

            try:
                self._client = MilvusClient(uri=self.uri, token=self.token or None, timeout=10)
                self._client.get_server_version()
            except Exception as exc:
                self._client = None
                raise DependencyError(
                    f"无法连接 Milvus：{self.uri}（{exc}）。"
                    "请在 WSL 中执行：cd ~ && bash standalone_embed.sh start",
                    uri=self.uri,
                ) from exc
        return self._client

    def ping(self) -> dict[str, Any]:
        version = self.client.get_server_version()
        return {"uri": self.uri, "version": version, "collections": self.client.list_collections()}

    # ------------------------------------------------------------------ 建表
    def ensure_collections(self, recreate: bool = False) -> None:
        """创建知识块与长期记忆集合（幂等）。"""

        with self._lock:
            if recreate:
                self.drop_all()
            self._ensure_kb_collection()
            self._ensure_memory_collection()
            self._ready = True

    def drop_all(self) -> None:
        for name in (self.kb_collection, self.memory_collection):
            if self.client.has_collection(name):
                self.client.drop_collection(name)
                logger.warning("已删除集合：%s", name)

    def _ensure_kb_collection(self) -> None:
        from pymilvus import DataType

        if self.client.has_collection(self.kb_collection):
            self.client.load_collection(self.kb_collection)
            return
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dense_dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("role", DataType.VARCHAR, max_length=64)
        schema.add_field("scope", DataType.VARCHAR, max_length=64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
        schema.add_field("doc_title", DataType.VARCHAR, max_length=512)
        schema.add_field("section", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("tags", DataType.VARCHAR, max_length=512)
        schema.add_field("char_len", DataType.INT64)
        schema.add_field("created_at", DataType.INT64)

        index = self.client.prepare_index_params()
        dense_cfg = self.config.section("milvus.dense_index")
        sparse_cfg = self.config.section("milvus.sparse_index")
        index.add_index(
            field_name="dense",
            index_type=str(dense_cfg.get("index_type", "HNSW")),
            metric_type=str(dense_cfg.get("metric_type", "COSINE")),
            params={"M": int(dense_cfg.get("M", 16)), "efConstruction": int(dense_cfg.get("efConstruction", 200))},
        )
        index.add_index(
            field_name="sparse",
            index_type=str(sparse_cfg.get("index_type", "SPARSE_INVERTED_INDEX")),
            metric_type=str(sparse_cfg.get("metric_type", "IP")),
        )
        self.client.create_collection(self.kb_collection, schema=schema, index_params=index)
        logger.info("已创建知识块集合：%s", self.kb_collection)

    def _ensure_memory_collection(self) -> None:
        from pymilvus import DataType

        if self.client.has_collection(self.memory_collection):
            self.client.load_collection(self.memory_collection)
            return
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dense_dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("text", DataType.VARCHAR, max_length=4096)
        schema.add_field("user_id", DataType.VARCHAR, max_length=64)
        schema.add_field("role_id", DataType.VARCHAR, max_length=64)
        schema.add_field("kind", DataType.VARCHAR, max_length=32)
        schema.add_field("importance", DataType.INT64)
        schema.add_field("ts", DataType.INT64)
        schema.add_field("session_id", DataType.VARCHAR, max_length=64)

        index = self.client.prepare_index_params()
        dense_cfg = self.config.section("milvus.dense_index")
        sparse_cfg = self.config.section("milvus.sparse_index")
        index.add_index(
            field_name="dense",
            index_type=str(dense_cfg.get("index_type", "HNSW")),
            metric_type=str(dense_cfg.get("metric_type", "COSINE")),
            params={"M": int(dense_cfg.get("M", 16)), "efConstruction": int(dense_cfg.get("efConstruction", 200))},
        )
        index.add_index(
            field_name="sparse",
            index_type=str(sparse_cfg.get("index_type", "SPARSE_INVERTED_INDEX")),
            metric_type=str(sparse_cfg.get("metric_type", "IP")),
        )
        self.client.create_collection(self.memory_collection, schema=schema, index_params=index)
        logger.info("已创建长期记忆集合：%s", self.memory_collection)

    # ------------------------------------------------------------------ 写入
    def insert_chunks(self, rows: Sequence[dict[str, Any]]) -> int:
        if not rows:
            return 0
        now = int(time.time())
        payload = []
        for row in rows:
            payload.append(
                {
                    "dense": row["dense"],
                    "sparse": row["sparse"],
                    "text": row["text"][:8000],
                    "role": row.get("role", "shared"),
                    "scope": row.get("scope", row.get("role", "shared")),
                    "doc_id": str(row.get("doc_id", ""))[:120],
                    "doc_title": str(row.get("doc_title", ""))[:500],
                    "section": str(row.get("section", ""))[:500],
                    "chunk_index": int(row.get("chunk_index", 0)),
                    "source": str(row.get("source", ""))[:500],
                    "tags": ",".join(row.get("tags", []))[:500] if isinstance(row.get("tags"), (list, tuple)) else str(row.get("tags", ""))[:500],
                    "char_len": int(row.get("char_len", len(row["text"]))),
                    "created_at": int(row.get("created_at", now)),
                }
            )
        result = self.client.insert(self.kb_collection, payload)
        return int(result.get("insert_count", len(payload))) if isinstance(result, dict) else len(payload)

    def flush(self, collection: str | None = None) -> None:
        self.client.flush(collection or self.kb_collection)

    # ------------------------------------------------------------------ 检索
    def _dense_params(self) -> dict[str, Any]:
        return {
            "metric_type": str(self.config.get("milvus.dense_index.metric_type", "COSINE")),
            "params": {"ef": int(self.config.get("milvus.search.ef", 128))},
        }

    def _sparse_params(self) -> dict[str, Any]:
        return {"metric_type": str(self.config.get("milvus.sparse_index.metric_type", "IP")), "params": {}}

    @staticmethod
    def _hit_pk(hit: dict[str, Any]) -> int:
        for key in ("pk", "id"):
            if key in hit:
                return int(hit[key])
        entity = hit.get("entity") or {}
        return int(entity.get("pk", entity.get("id", -1)))

    @staticmethod
    def _normalize_hits(raw: Any) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for hit in raw or []:
            item = {"pk": MilvusStore._hit_pk(hit), "score": float(hit.get("distance", 0.0))}
            entity = hit.get("entity") or {}
            item.update({key: value for key, value in entity.items() if key not in {"pk"}})
            hits.append(item)
        return hits

    def search_dense(
        self, vector: Sequence[float], expr: str, limit: int, output_fields: Sequence[str] = KB_FIELDS
    ) -> list[dict[str, Any]]:
        raw = self.client.search(
            self.kb_collection,
            data=[list(vector)],
            anns_field="dense",
            search_params=self._dense_params(),
            limit=limit,
            filter=expr or None,
            output_fields=list(output_fields),
            timeout=self.search_timeout,
        )
        return self._normalize_hits(raw[0] if raw else [])

    def search_sparse(
        self, sparse: dict[int, float], expr: str, limit: int, output_fields: Sequence[str] = KB_FIELDS
    ) -> list[dict[str, Any]]:
        if not sparse:
            return []
        raw = self.client.search(
            self.kb_collection,
            data=[dict(sparse)],
            anns_field="sparse",
            search_params=self._sparse_params(),
            limit=limit,
            filter=expr or None,
            output_fields=list(output_fields),
            timeout=self.search_timeout,
        )
        return self._normalize_hits(raw[0] if raw else [])

    def hybrid_search(
        self,
        vector: Sequence[float],
        sparse: dict[int, float],
        expr: str,
        limit: int,
        output_fields: Sequence[str] = KB_FIELDS,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Milvus 原生混合检索（稠密 + 稀疏，RRF 融合）。"""

        from pymilvus import AnnSearchRequest, RRFRanker

        requests = [
            AnnSearchRequest(
                data=[list(vector)], anns_field="dense", param=self._dense_params(), limit=limit,
                expr=expr or None,
            )
        ]
        if sparse:
            requests.append(
                AnnSearchRequest(
                    data=[dict(sparse)], anns_field="sparse", param=self._sparse_params(), limit=limit,
                    expr=expr or None,
                )
            )
        raw = self.client.hybrid_search(
            self.kb_collection,
            requests,
            ranker=RRFRanker(k=rrf_k),
            limit=limit,
            output_fields=list(output_fields),
            timeout=self.search_timeout,
        )
        return self._normalize_hits(raw[0] if raw else [])

    def query_chunks(
        self, expr: str, output_fields: Sequence[str] = KB_FIELDS, limit: int = 1000, offset: int = 0
    ) -> list[dict[str, Any]]:
        rows = self.client.query(
            self.kb_collection, filter=expr, output_fields=list(output_fields), limit=limit, offset=offset
        )
        return [dict(row) for row in rows or []]

    def count_chunks(self, expr: str = "") -> int:
        rows = self.client.query(self.kb_collection, filter=expr or None, output_fields=["count(*)"])
        if rows and isinstance(rows[0], dict):
            return int(rows[0].get("count(*)", 0))
        return 0

    def delete_chunks(self, expr: str) -> int:
        result = self.client.delete(self.kb_collection, filter=expr)
        if isinstance(result, dict):
            return int(result.get("delete_count", 0))
        return int(getattr(result, "delete_count", 0) or 0)

    # ------------------------------------------------------------ 长期记忆
    def insert_memories(self, rows: Sequence[dict[str, Any]]) -> int:
        if not rows:
            return 0
        payload = [
            {
                "dense": row["dense"],
                "sparse": row["sparse"],
                "text": str(row["text"])[:4000],
                "user_id": str(row.get("user_id", ""))[:64],
                "role_id": str(row.get("role_id", "*"))[:64],
                "kind": str(row.get("kind", "fact"))[:32],
                "importance": int(row.get("importance", 3)),
                "ts": int(row.get("ts", time.time())),
                "session_id": str(row.get("session_id", ""))[:64],
            }
            for row in rows
        ]
        result = self.client.insert(self.memory_collection, payload)
        self.invalidate_memory_hint()
        return int(result.get("insert_count", len(payload))) if isinstance(result, dict) else len(payload)

    def search_memory(
        self, vector: Sequence[float], sparse: dict[int, float] | None, expr: str, limit: int
    ) -> list[dict[str, Any]]:
        fields = MEMORY_FIELDS
        # 空集合上调用 Milvus 原生 hybrid_search 会报 "unsupported ID type"，
        # 这里先用带缓存的计数探针跳过，既避免无谓报错也省一次 RPC。
        if sparse and self._memory_has_rows():
            try:
                return self._memory_hybrid(vector, sparse, expr, limit)
            except Exception as exc:  # pragma: no cover - 降级为稠密检索
                logger.warning("长期记忆混合检索失败，降级为稠密检索：%s", exc)
        raw = self.client.search(
            self.memory_collection,
            data=[list(vector)],
            anns_field="dense",
            search_params=self._dense_params(),
            limit=limit,
            filter=expr or None,
            output_fields=list(fields),
            timeout=self.search_timeout,
        )
        return self._normalize_hits(raw[0] if raw else [])

    def _memory_hybrid(
        self, vector: Sequence[float], sparse: dict[int, float], expr: str, limit: int
    ) -> list[dict[str, Any]]:
        from pymilvus import AnnSearchRequest, RRFRanker

        requests = [
            AnnSearchRequest(data=[list(vector)], anns_field="dense", param=self._dense_params(),
                             limit=limit, expr=expr or None),
            AnnSearchRequest(data=[dict(sparse)], anns_field="sparse", param=self._sparse_params(),
                             limit=limit, expr=expr or None),
        ]
        raw = self.client.hybrid_search(
            self.memory_collection, requests, ranker=RRFRanker(k=60), limit=limit,
            output_fields=list(MEMORY_FIELDS), timeout=self.search_timeout,
        )
        return self._normalize_hits(raw[0] if raw else [])

    def delete_memories(self, expr: str) -> int:
        result = self.client.delete(self.memory_collection, filter=expr)
        if isinstance(result, dict):
            return int(result.get("delete_count", 0))
        return int(getattr(result, "delete_count", 0) or 0)

    def count_memories(self, expr: str = "") -> int:
        rows = self.client.query(self.memory_collection, filter=expr or None, output_fields=["count(*)"])
        if rows and isinstance(rows[0], dict):
            return int(rows[0].get("count(*)", 0))
        return 0

    def _memory_has_rows(self, ttl: float = 30.0) -> bool:
        """带 30 秒缓存的长期记忆行数探针。"""

        now = time.time()
        stamp, cached = self._memory_rows_hint
        if cached >= 0 and now - stamp < ttl:
            return cached > 0
        try:
            count = self.count_memories() if self.client.has_collection(self.memory_collection) else 0
        except Exception:  # pragma: no cover - 探测失败时按“有数据”处理，走原有降级逻辑
            count = 1
        self._memory_rows_hint = (now, count)
        return count > 0

    def invalidate_memory_hint(self) -> None:
        """写入长期记忆后调用，让行数探针立即失效。"""

        self._memory_rows_hint = (0.0, -1)

    # ------------------------------------------------------------------ 统计
    def stats(self) -> dict[str, Any]:
        info: dict[str, Any] = {"uri": self.uri, "kb_collection": self.kb_collection,
                                "memory_collection": self.memory_collection}
        try:
            info["version"] = self.client.get_server_version()
            info["kb_rows"] = self.count_chunks() if self.client.has_collection(self.kb_collection) else 0
            info["memory_rows"] = (
                self.count_memories() if self.client.has_collection(self.memory_collection) else 0
            )
            info["reachable"] = True
        except Exception as exc:  # pragma: no cover
            info["reachable"] = False
            info["error"] = str(exc)
        return info

    def scope_expr(self, role_id: str, include_shared: bool = True) -> str:
        """按角色生成 Milvus 过滤表达式。"""

        scopes = [f'scope == "{role_id}"']
        if include_shared:
            scopes.append('scope == "shared"')
        return " or ".join(scopes)

    def role_expr(self, role_id: str, include_shared: bool = True) -> str:
        """按角色生成表达式（role 字段视角，等价于 scope_expr）。"""

        roles = [f'role == "{role_id}"']
        if include_shared:
            roles.append('role == "shared"')
        return " or ".join(roles)

    def scope_distribution(self) -> list[dict[str, Any]]:
        """按 scope 统计块数量（知识库健康检查用）。"""

        rows = self.query_chunks("pk >= 0", output_fields=["scope"], limit=16384)
        counter: dict[str, int] = {}
        for row in rows:
            counter[str(row.get("scope", ""))] = counter.get(str(row.get("scope", "")), 0) + 1
        return [{"scope": key, "count": value} for key, value in sorted(counter.items())]


_store: MilvusStore | None = None
_store_lock = threading.Lock()


def get_milvus(config: Config | None = None) -> MilvusStore:
    """获取全局 MilvusStore 单例。"""

    global _store
    with _store_lock:
        if _store is None:
            _store = MilvusStore(config)
        return _store
