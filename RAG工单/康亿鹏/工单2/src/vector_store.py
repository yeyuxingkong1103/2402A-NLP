# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：向量库模块。基于 Milvus 实现知识库的存储、管理与相似度检索，
          是“知识库管理”功能的核心实现。
"""
import config
from src.utils import logger


class VectorStoreError(Exception):
    """向量库异常（连接失败、集合不存在等）。"""


class MilvusStore:
    """Milvus 向量库封装（使用 MilvusClient 轻量接口）。"""

    def __init__(self, uri: str = None, token: str = None, collection: str = None,
                 dim: int = None, metric: str = None):
        self.uri = uri or config.MILVUS_URI
        self.token = token if token is not None else config.MILVUS_TOKEN
        self.collection = collection or config.MILVUS_COLLECTION
        self.dim = dim or config.MILVUS_DIM
        self.metric = metric or config.MILVUS_METRIC
        self._client = self._connect()

    def _connect(self):
        """建立与 Milvus 的连接。"""
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise VectorStoreError("未安装 pymilvus，请执行：pip install pymilvus") from exc

        try:
            kwargs = {"uri": self.uri}
            if self.token:
                kwargs["token"] = self.token
            client = MilvusClient(**kwargs)
            logger.info("已连接 Milvus：%s（collection=%s）", self.uri, self.collection)
            return client
        except Exception as exc:
            raise VectorStoreError(
                f"连接 Milvus 失败：{self.uri}。请确认 Milvus 已启动（docker start milvus）。原因：{exc}"
            ) from exc

    # ------------------------------------------------------------------
    # 集合管理
    # ------------------------------------------------------------------
    def has_collection(self) -> bool:
        try:
            return self._client.has_collection(self.collection)
        except Exception as exc:
            raise VectorStoreError(f"检查集合失败：{exc}") from exc

    def create_collection(self, drop_old: bool = False):
        """创建集合（drop_old=True 时先删除已有集合，实现知识库重建）。"""
        if self.has_collection():
            if not drop_old:
                logger.info("集合已存在，跳过创建：%s", self.collection)
                return
            self.drop_collection()
        self._client.create_collection(
            collection_name=self.collection,
            dimension=self.dim,
            primary_field_name="chunk_id",
            id_type="string",
            max_length=512,
            vector_field_name="vector",
            metric_type=self.metric,
            auto_id=False,
            enable_dynamic_field=True,
        )
        logger.info("已创建集合：%s（dim=%d, metric=%s）", self.collection, self.dim, self.metric)

    def drop_collection(self):
        """删除集合（清空知识库）。"""
        if self.has_collection():
            self._client.drop_collection(self.collection)
            logger.info("已删除集合：%s", self.collection)

    def count(self) -> int:
        """统计知识库中的片段数量。"""
        if not self.has_collection():
            return 0
        try:
            res = self._client.query(self.collection, filter="", output_fields=["count(*)"])
            return int(res[0].get("count(*)", 0)) if res else 0
        except Exception as exc:
            raise VectorStoreError(f"统计失败：{exc}") from exc

    # ------------------------------------------------------------------
    # 数据写入
    # ------------------------------------------------------------------
    def insert(self, chunks: list, embeddings: list, batch_size: int = 500) -> int:
        """写入文档片段及其向量，返回写入条数。"""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise VectorStoreError("片段数量与向量数量不一致，写入中止。")

        rows = []
        for chunk, vec in zip(chunks, embeddings):
            meta = chunk.metadata or {}
            rows.append(
                {
                    "chunk_id": str(meta.get("chunk_id", ""))[:512],
                    "vector": vec,
                    "content": chunk.page_content[:65000],
                    "source": str(meta.get("source", ""))[:512],
                    "page": int(meta.get("page", 0)),
                    "type": str(meta.get("type", "text"))[:32],
                    "chunk_index": int(meta.get("chunk_index", 0)),
                }
            )

        total = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            self._client.insert(collection_name=self.collection, data=batch)
            total += len(batch)
        self._client.flush(self.collection)
        logger.info("向量写入完成：%d 条", total)
        return total

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, query_vector: list, top_k: int = None) -> list:
        """按向量相似度检索，返回 [{"content", "metadata", "score"}, ...]。"""
        top_k = top_k or config.RETRIEVE_TOP_K
        if not self.has_collection() or self.count() == 0:
            logger.warning("知识库为空，请先执行数据入库（python main.py ingest）。")
            return []

        results = self._client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=top_k,
            output_fields=["content", "source", "page", "type", "chunk_index"],
            search_params={"metric_type": self.metric, "params": {}},
        )

        hits = []
        for hit in results[0] if results else []:
            entity = hit.get("entity", {}) or {}
            hits.append(
                {
                    "content": entity.get("content", ""),
                    "score": float(hit.get("distance", 0.0)),
                    "metadata": {
                        # pymilvus 3.x 主键放在字段名（chunk_id）下，2.x 放在 "id" 下
                        "chunk_id": hit.get("chunk_id") or hit.get("id"),
                        "source": entity.get("source", ""),
                        "page": entity.get("page", 0),
                        "type": entity.get("type", "text"),
                        "chunk_index": entity.get("chunk_index", 0),
                    },
                }
            )
        return hits


_store_cache = {}


def get_vector_store() -> MilvusStore:
    """获取向量库单例。"""
    if "store" not in _store_cache:
        _store_cache["store"] = MilvusStore()
    return _store_cache["store"]
