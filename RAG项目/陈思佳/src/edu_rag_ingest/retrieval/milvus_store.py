from __future__ import annotations

"""Milvus 数据访问层，负责 Collection、索引、向量写入和删除。"""

import json
import logging
from pathlib import Path
from typing import Any

from ..config.config import MilvusConfig

logger = logging.getLogger(__name__)


class MilvusChunkStore:
    def __init__(self, config: MilvusConfig) -> None:
        self.config = config
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as error:  # pragma: no cover - 依赖由运行环境提供
            raise RuntimeError("缺少 pymilvus，请先运行：pip install pymilvus") from error

        self.DataType = DataType
        self.client = MilvusClient(uri=config.uri, token=config.token or None)

    def ensure_collection(self) -> None:
        if self.client.has_collection(self.config.collection_name):
            if self.config.drop_existing:
                logger.warning("删除已有 Milvus collection：%s", self.config.collection_name)
                self.client.drop_collection(self.config.collection_name)
            else:
                logger.info("Milvus collection 已存在：%s", self.config.collection_name)
                self.client.load_collection(self.config.collection_name)
                return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", self.DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("document_id", self.DataType.VARCHAR, max_length=128)
        schema.add_field("chunk_index", self.DataType.INT64)
        schema.add_field("content", self.DataType.VARCHAR, max_length=32768)
        schema.add_field("subject", self.DataType.VARCHAR, max_length=64)
        schema.add_field("grade", self.DataType.VARCHAR, max_length=64)
        schema.add_field("document_type", self.DataType.VARCHAR, max_length=128)
        schema.add_field("source_file", self.DataType.VARCHAR, max_length=512)
        schema.add_field("metadata_json", self.DataType.VARCHAR, max_length=4096)
        schema.add_field("vector", self.DataType.FLOAT_VECTOR, dim=self.config.vector_dim)

        self.client.create_collection(
            collection_name=self.config.collection_name,
            schema=schema,
            consistency_level="Strong",
        )
        self.client.create_index(
            collection_name=self.config.collection_name,
            index_params=self._build_index_params(),
        )
        self.client.load_collection(self.config.collection_name)
        logger.info("已创建 Milvus collection：%s", self.config.collection_name)

    def insert_chunks(self, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 和 vectors 数量不一致")

        rows = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            metadata = chunk.get("metadata", {})
            rows.append(
                {
                    "id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "chunk_index": int(chunk["chunk_index"]),
                    "content": self._truncate_utf8(chunk["content"], 32768),
                    "subject": metadata.get("subject", ""),
                    "grade": metadata.get("grade", ""),
                    "document_type": metadata.get("document_type", ""),
                    "source_file": metadata.get("source_file", ""),
                    "metadata_json": self._truncate_utf8(json.dumps(metadata, ensure_ascii=False), 4096),
                    "vector": vector,
                }
            )

        if rows:
            self.client.insert(collection_name=self.config.collection_name, data=rows)
            logger.info("已写入 Milvus：%s 条", len(rows))

    def existing_chunk_ids(self, chunk_ids: list[str]) -> set[str]:
        """查询已存在的 chunk 主键，供增量入库跳过旧数据。"""
        existing: set[str] = set()
        for start in range(0, len(chunk_ids), 100):
            batch = chunk_ids[start : start + 100]
            if not batch:
                continue
            rows = self.client.get(
                collection_name=self.config.collection_name,
                ids=batch,
                output_fields=["id"],
            )
            existing.update(str(row["id"]) for row in rows if row.get("id") is not None)
        return existing

    def delete_document_chunks(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.config.collection_name,
            filter=f'document_id == "{document_id}"',
        )

    @staticmethod
    def build_filter(filters: dict[str, str] | None) -> str:
        if not filters:
            return ""
        allowed_fields = {"subject", "grade", "document_type"}
        expressions = []
        for key, value in filters.items():
            if key in allowed_fields and value:
                escaped = str(value).replace('\\', '\\\\').replace('"', '\\"')
                expressions.append(f'{key} == "{escaped}"')
        return " and ".join(expressions)

    def _build_index_params(self):
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=self.config.index_type,
            metric_type=self.config.metric_type,
            params=self.config.index_params,
        )
        return index_params

    @staticmethod
    def _truncate_utf8(value: str, max_bytes: int) -> str:
        encoded = value.encode("utf-8")
        if len(encoded) <= max_bytes:
            return value
        return encoded[:max_bytes].decode("utf-8", errors="ignore")


def read_chunks_jsonl(path: Path) -> list[dict[str, Any]]:
    chunks = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks
