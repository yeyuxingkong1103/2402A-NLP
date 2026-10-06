import json
import time
from collections.abc import Sequence
from typing import Any

from pymilvus import DataType

# 批次 37 新增字段：摘要一句话上限（VARCHAR 按字符算，中文一句话 200 字符足够宽裕）
SUMMARY_MAX_LENGTH = 1024


class LegalMilvusStore:
    def __init__(self, *, client: Any, collection_name: str, dimension: int) -> None:
        self.client = client
        self.collection_name = collection_name
        self.dimension = dimension

    def _build_schema(self) -> Any:
        """构建 collection schema（批次 37 抽取：ensure/recreate 之前逐字重复，漏改一处就是 schema 漂移）。

        字段清单（17 个 = 14 原有 + 批次 37 新增 3 个）：
        - 标识与正文：chunk_key / document_version_id / chunk_type / article_number / retrieval_text
        - 时效与法域：law_name / document_type / jurisdiction / authority_level /
          effective_date / expiration_date / is_current / article_path
        - 批次 37 新增：created_at / updated_at（索引写入时间戳，秒级 INT64）、
          summary（条文一句话摘要，nullable；来源 chunk_summaries 表，经 summarize_chunks.py 生成）
        - 向量：dense_vector
        """
        schema = self.client.create_schema(auto_id=False)
        schema.add_field("chunk_key", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("document_version_id", DataType.INT64)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=32)
        schema.add_field("article_number", DataType.VARCHAR, max_length=64, nullable=True)
        schema.add_field("retrieval_text", DataType.VARCHAR, max_length=65535)

        # 时效与法域字段
        schema.add_field("law_name", DataType.VARCHAR, max_length=512)
        schema.add_field("document_type", DataType.VARCHAR, max_length=64)
        schema.add_field("jurisdiction", DataType.VARCHAR, max_length=128)
        schema.add_field("authority_level", DataType.INT64)
        schema.add_field("effective_date", DataType.INT64)  # Unix timestamp
        schema.add_field("expiration_date", DataType.INT64, nullable=True)  # Unix timestamp or NULL
        # 是否现行有效。本字段是**非空** BOOL（Milvus 不接受 None），
        # 因此"状态未知"与"不适用"（案例材料）一律写 True ——
        # 不适用**不参与时效判断**，状态未知也不判失效
        # （缺数据不能成为静默隐藏条文的理由）。判定口径见
        # app/db/vector_index_service.py 与 app/db/law_status.py。
        schema.add_field("is_current", DataType.BOOL)
        schema.add_field("article_path", DataType.VARCHAR, max_length=128)  # 条文路径（如"47-2-1"）

        # 批次 37：写入与更新时间（索引 upsert 时刻，秒级时间戳）与条文摘要。
        # summary 可空：摘要生成（summarize_chunks.py）是异步离线任务，索引先跑、摘要后补，
        # 未生成摘要的 chunk 写 None，不阻塞索引链路。
        schema.add_field("created_at", DataType.INT64)
        schema.add_field("updated_at", DataType.INT64)
        schema.add_field("summary", DataType.VARCHAR, max_length=SUMMARY_MAX_LENGTH, nullable=True)

        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self.dimension)
        return schema

    def _index_params(self) -> Any:
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            "dense_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        return index_params

    def ensure_collection(self) -> None:
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                self.collection_name,
                schema=self._build_schema(),
                index_params=self._index_params(),
            )
        self.client.load_collection(self.collection_name)

    def recreate_collection(self) -> None:
        """删除现有 collection 并重新创建（用于 schema 变更）。

        注意：本方法会 drop 现有 collection，只允许由 CLI 显式传
        --recreate-collection 触达，禁止任何自动路径调用。
        """
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        self.client.create_collection(
            self.collection_name,
            schema=self._build_schema(),
            index_params=self._index_params(),
        )
        self.client.load_collection(self.collection_name)

    @staticmethod
    def now_timestamp() -> int:
        """索引写入时间戳（秒级）。created_at/updated_at 共用：upsert 语义下
        同一 chunk_key 重复写入即为更新，两者取同一时刻即可区分新旧（重索引后单调变化）。"""
        return int(time.time())

    def existing_chunk_keys(self, chunk_keys: Sequence[str]) -> set[str]:
        """查询已经存在的 chunk key，避免失败重试重复生成向量。"""
        if not chunk_keys:
            return set()

        filter_expression = (
            "chunk_key in ["
            + ",".join(json.dumps(chunk_key) for chunk_key in chunk_keys)
            + "]"
        )
        rows = self.client.query(
            self.collection_name,
            filter=filter_expression,
            output_fields=["chunk_key"],
            limit=len(chunk_keys),
        )
        return {row["chunk_key"] for row in rows}

    def upsert(self, rows: Sequence[dict[str, Any]]) -> int:
        now_ts = self.now_timestamp()
        payload = [
            {
                "chunk_key": row["chunk_key"],
                "document_version_id": row["document_version_id"],
                "chunk_type": row["chunk_type"],
                "article_number": row["article_number"],
                "retrieval_text": row["retrieval_text"][:65535],
                # 时效与法域字段
                "law_name": row["law_name"],
                "document_type": row["document_type"],
                "jurisdiction": row["jurisdiction"],
                "authority_level": row["authority_level"],
                "effective_date": row["effective_date"],
                "expiration_date": row.get("expiration_date"),
                "is_current": row["is_current"],
                "article_path": row["article_path"],
                # 批次 37：时间戳与摘要（row 未带 summary 时写 None——可空字段，摘要后补）
                "created_at": row.get("created_at", now_ts),
                "updated_at": now_ts,
                "summary": row.get("summary"),
                "dense_vector": row["vector"],
            }
            for row in rows
        ]
        result = self.client.upsert(self.collection_name, payload)
        return int(result.get("upsert_count", 0))

    def get_all_chunk_keys(self) -> set[str]:
        """查询向量库中的所有 chunk_key，分批查询避免超过 Milvus limit 限制。"""
        all_keys = set()
        batch_size = 10000  # Milvus 单次查询限制是 16384，用 10000 更安全
        offset = 0

        while True:
            rows = self.client.query(
                self.collection_name,
                filter="chunk_key != \"\"",
                output_fields=["chunk_key"],
                limit=batch_size,
                offset=offset,
            )
            if not rows:
                break

            all_keys.update(row["chunk_key"] for row in rows)
            offset += len(rows)

            # 如果返回的行数少于 batch_size，说明已经到底了
            if len(rows) < batch_size:
                break

        return all_keys

    def delete_by_chunk_keys(self, chunk_keys: Sequence[str]) -> int:
        """根据 chunk_key 批量删除向量。"""
        if not chunk_keys:
            return 0
        filter_expression = (
            "chunk_key in ["
            + ",".join(json.dumps(chunk_key) for chunk_key in chunk_keys)
            + "]"
        )
        result = self.client.delete(
            self.collection_name,
            filter=filter_expression,
        )
        return int(result.get("delete_count", 0))
