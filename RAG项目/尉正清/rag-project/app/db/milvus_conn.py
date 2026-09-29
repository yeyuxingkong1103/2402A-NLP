# app/db/milvus_conn.py
"""Milvus 连接与集合管理。

统一封装稠密 + 稀疏（BM25 风格）混合检索所需的读写操作：
  - rag_knowledge      知识库切片
  - long_term_memory   长期记忆摘要（按 user_id + role_id 隔离）
"""
import threading
from typing import Any, Dict, List, Optional

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

from app.config import settings
import logging

logger = logging.getLogger(__name__)


def expr_eq(**pairs) -> str:
    """构造 Milvus 过滤表达式。

        expr_eq(role_id="lawyer")                     -> 'role_id == "lawyer"'
        expr_eq(user_id="1", role_id="lawyer")        -> 'user_id == "1" and role_id == "lawyer"'

    统一在这里拼，有两个原因：
      1. 原先 7 处在 4 个模块里各拼各的，改一处容易漏
      2. 值里若含引号会拼出非法表达式 —— 在这里统一转义，杜绝这类隐患
    """
    out = []
    for k, v in pairs.items():
        if v is None:
            continue
        out.append('%s == "%s"' % (k, str(v).replace("\\", "\\\\").replace('"', '\\"')))
    return " and ".join(out)


class MilvusStore:
    """MilvusClient 的薄封装，只暴露项目用得到的操作。"""

    def __init__(self, uri: str, token: str = "", dim: int = 1024):
        self._client = MilvusClient(uri=uri, token=token or "")
        self.dim = dim

    @property
    def client(self) -> MilvusClient:
        return self._client

    # ---------------- 集合管理 ----------------
    def has_collection(self, name: str) -> bool:
        return self._client.has_collection(name)

    def list_collections(self) -> List[str]:
        return self._client.list_collections()

    def drop_collection(self, name: str) -> None:
        if self.has_collection(name):
            self._client.drop_collection(name)
            logger.warning("已删除集合 %s", name)

    def count(self, name: str) -> int:
        """精确统计行数。

        get_collection_stats 的 row_count 不会扣减已删除的行，删数据后
        数字会一直偏高，所以优先用 count(*) 聚合查询，失败再退回 stats。
        """
        if not self.has_collection(name):
            return 0
        try:
            rows = self._retry_after_load(lambda: self._client.query(
                collection_name=name, filter="",
                output_fields=["count(*)"]), name)
            if rows and "count(*)" in rows[0]:
                return int(rows[0]["count(*)"])
        except Exception as e:
            logger.debug("count(*) 失败，回退到 collection stats: %s", e)
        try:
            return self._client.get_collection_stats(name).get("row_count", 0)
        except Exception as e:                                  # pragma: no cover
            logger.warning("统计集合 %s 行数失败: %s", name, e)
            return 0

    def ensure_knowledge_collection(self, name: str) -> None:
        """知识库集合：稠密 + 稀疏双向量，带角色与元数据字段。

        字段用途：
          id            自增主键，唯一标识每条切片
          vector        稠密向量（BGE-M3）
          sparse_vector 稀疏权重，与稠密向量一起支撑混合检索（BM25 风格）
          text          向量对应的原文，命中后直接喂给大模型
          law/article   法条元数据，供「元数据路召回」精确过滤
          summary       切片摘要，用于结果展示与快速判读
          created_at / updated_at  便于增量更新与数据治理
          source        文档来源，支持按文件先删后增
        """
        if self.has_collection(name):
            return
        schema = self._client.create_schema(auto_id=True,
                                            enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("summary", DataType.VARCHAR, max_length=2048)
        schema.add_field("role_id", DataType.VARCHAR, max_length=64)
        schema.add_field("source", DataType.VARCHAR, max_length=255)
        schema.add_field("doc_type", DataType.VARCHAR, max_length=32)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("law", DataType.VARCHAR, max_length=128)
        schema.add_field("article", DataType.VARCHAR, max_length=64)
        schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
        schema.add_field("created_at", DataType.VARCHAR, max_length=32)
        schema.add_field("updated_at", DataType.VARCHAR, max_length=32)
        self._create(name, schema)
        logger.info("知识库集合 %s 创建完成（自增主键 + 元数据字段）", name)

    def ensure_memory_collection(self, name: str) -> None:
        """长期记忆集合：按 user_id + role_id 隔离。"""
        if self.has_collection(name):
            return
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("user_id", DataType.VARCHAR, max_length=64)
        schema.add_field("role_id", DataType.VARCHAR, max_length=64)
        schema.add_field("summary", DataType.VARCHAR, max_length=65535)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("created_at", DataType.VARCHAR, max_length=32)
        self._create(name, schema)
        logger.info("长期记忆集合 %s 创建完成", name)

    def _create(self, name: str, schema) -> None:
        self._client.create_collection(collection_name=name, schema=schema)
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector", index_type="HNSW", metric_type="IP",
            params={"M": 16, "efConstruction": 200})
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        self._client.create_index(collection_name=name, index_params=index_params)
        self.load(name)

    def load(self, name: str) -> None:
        """把集合载入内存。Milvus 重启后集合不会自动载入，检索前必须显式 load。"""
        try:
            self._client.load_collection(name)
        except Exception as e:                              # pragma: no cover
            logger.debug("load_collection(%s): %s", name, e)

    @staticmethod
    def _not_loaded(err: Exception) -> bool:
        return "not loaded" in str(err).lower()

    def _retry_after_load(self, fn, name: str):
        """集合未载入时重新 load 并重试一次。"""
        try:
            return fn()
        except Exception as e:
            if not self._not_loaded(e):
                raise
            logger.warning("集合 %s 未载入内存，重新 load 后重试", name)
            self.load(name)
            return fn()

    # ---------------- 写入 ----------------
    def insert(self, name: str, rows: List[Dict[str, Any]],
               batch_size: int = 500) -> int:
        """分批写入，避免单次请求超过 Milvus 的 64MB 限制。"""
        if not rows:
            return 0
        total = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            self._client.insert(collection_name=name, data=batch)
            total += len(batch)
            logger.debug("写入 %s/%s 到 %s", total, len(rows), name)
        self._client.flush(name)
        self.load(name)          # 新数据要能被检索到，集合必须处于载入状态
        return total

    def delete_by_expr(self, name: str, expr: str) -> None:
        """按表达式删除，并 flush 让删除**立刻可见**。

        Milvus 的删除是最终一致的：不 flush 的话，接口已经返回 200「删除成功」，
        紧接着的查询仍能查到这些数据（实测删除后计数还停在旧值）。
        项目其他地方确立了「200 即已生效」的口径（见 mysql_conn.get_db 的注释），
        这里对齐，避免调用方拿到 200 却看到旧数据。
        """
        self._client.delete(collection_name=name, filter=expr)
        self._client.flush(name)

    def query(self, name: str, expr: str,
              output_fields: Optional[List[str]] = None,
              limit: int = 10) -> List[Dict[str, Any]]:
        if not self.has_collection(name):
            return []
        return self._retry_after_load(lambda: self._client.query(
            collection_name=name, filter=expr,
            output_fields=output_fields or ["*"], limit=limit), name)

    # Milvus 单次查询的 offset+limit 上限是 16384，超过要翻页
    MAX_WINDOW = 16384

    def query_all(self, name: str, expr: str,
                  output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """翻页取全量。

        直接用 limit=20000 会抛 invalid max query result window，
        而知识库还在增长，写死一个「够大」的 limit 迟早会撞上限。
        """
        if not self.has_collection(name):
            return []
        fields = output_fields or ["*"]
        batch = self.MAX_WINDOW
        out, offset = [], 0
        while True:
            rows = self._retry_after_load(lambda: self._client.query(
                collection_name=name, filter=expr, output_fields=fields,
                limit=batch, offset=offset), name)
            if not rows:
                break
            out.extend(rows)
            if len(rows) < batch:
                break
            offset += batch
        return out

    # ---------------- 检索 ----------------
    def hybrid_search(self, name: str, dense_vec: List[float],
                      sparse_vec: Dict[int, float], expr: Optional[str] = None,
                      limit: int = 20, output_fields: Optional[List[str]] = None,
                      ranker_k: int = 60) -> List[Dict[str, Any]]:
        """稠密 + 稀疏双路召回，RRF 融合。返回 [{...fields, score}]。"""
        fields = output_fields or ["text", "role_id", "source", "doc_type",
                                   "title", "law", "article", "summary"]
        dense_req = AnnSearchRequest(
            data=[dense_vec], anns_field="vector",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=limit, expr=expr)
        sparse_req = AnnSearchRequest(
            data=[sparse_vec], anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=limit, expr=expr)
        res = self._retry_after_load(lambda: self._client.hybrid_search(
            collection_name=name, reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=ranker_k), limit=limit,
            output_fields=fields), name)
        return self._hits_to_dicts(res)

    def dense_search(self, name: str, dense_vec: List[float],
                     expr: Optional[str] = None, limit: int = 20,
                     output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """纯向量检索，混合检索失败时的降级路径。"""
        fields = output_fields or ["text", "role_id", "source", "doc_type",
                                   "title", "law", "article", "summary"]
        res = self._retry_after_load(lambda: self._client.search(
            collection_name=name, data=[dense_vec], anns_field="vector",
            search_params={"metric_type": "IP", "params": {"ef": 64}},
            limit=limit, filter=expr or "", output_fields=fields), name)
        return self._hits_to_dicts(res)

    @staticmethod
    def _hits_to_dicts(res) -> List[Dict[str, Any]]:
        out = []
        for hits in res:
            for h in hits:
                item = dict(h.get("entity", {}) or {})
                item["score"] = h.get("distance")
                item["id"] = h.get("id")
                out.append(item)
        return out


# ---------------- 单例 ----------------
_store: Optional[MilvusStore] = None
_lock = threading.Lock()


def get_milvus() -> MilvusStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = MilvusStore(
                    uri=settings.MILVUS_URI,
                    token=settings.MILVUS_TOKEN,
                    dim=settings.EMBEDDING_DIM)
                logger.info("Milvus 已连接: %s", settings.MILVUS_URI)
    return _store
