"""Vector-store adapters used by the ingestion indexer."""

from __future__ import annotations

import logging
from typing import Any

from app.ingestion.indexer_models import IndexingError

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Milvus Standalone adapter for the ``kf_chunks`` collection."""

    def __init__(
        self,
        *,
        uri: str,
        collection_name: str = "kf_chunks",
        dimension: int = 1024,
        token: str | None = None,
        enable_sparse: bool = False,
    ) -> None:
        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )
        except ImportError as exc:
            raise IndexingError("pymilvus is required for MilvusVectorStore") from exc

        self._Collection = Collection
        self.collection_name = collection_name
        self.dimension = dimension
        self.enable_sparse = enable_sparse
        connections.connect(alias="default", uri=uri, token=token or "")
        if not utility.has_collection(collection_name):
            collection = Collection(
                name=collection_name,
                schema=CollectionSchema(
                    fields=self._fields(DataType, dimension, enable_sparse),
                    enable_dynamic_field=False,
                ),
            )
            collection.create_index(
                field_name="vector",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
        self.collection = Collection(collection_name)
        self.collection.load()

    @staticmethod
    def _fields(data_type: Any, dimension: int, enable_sparse: bool) -> list[Any]:
        """Build the fixed schema shared by newly created collections."""
        fields = [
            {"name": "id", "dtype": data_type.VARCHAR, "max_length": 128, "is_primary": True},
            {"name": "vector", "dtype": data_type.FLOAT_VECTOR, "dim": dimension},
            {"name": "text", "dtype": data_type.VARCHAR, "max_length": 65535},
            {"name": "source", "dtype": data_type.VARCHAR, "max_length": 1024},
            {"name": "doc_id", "dtype": data_type.VARCHAR, "max_length": 64},
            {"name": "chunk_id", "dtype": data_type.VARCHAR, "max_length": 128},
            {"name": "summary", "dtype": data_type.VARCHAR, "max_length": 4096},
            {"name": "created_at", "dtype": data_type.INT64},
            {"name": "updated_at", "dtype": data_type.INT64},
            {"name": "tenant_id", "dtype": data_type.INT64},
            {"name": "role_id", "dtype": data_type.INT64},
            {"name": "kb_id", "dtype": data_type.INT64},
            {"name": "document_version", "dtype": data_type.INT32},
            {"name": "is_published", "dtype": data_type.BOOL},
        ]
        if enable_sparse:
            fields.append({"name": "sparse_vector", "dtype": data_type.SPARSE_FLOAT_VECTOR})
        return [
            __import__("pymilvus").FieldSchema(**field) for field in fields
        ]

    def upsert(self, records: list[dict[str, Any]]) -> None:
        """Write records and flush before metadata becomes visible."""
        if not records:
            return
        self.collection.upsert(records)
        self.collection.flush()
        logger.info("milvus records upserted", extra={"count": len(records)})

    def delete_by_document(self, document_id: int | str) -> int:
        """Delete all vectors belonging to one document."""
        escaped = str(document_id).replace('"', '\\"')
        result = self.collection.delete(f'doc_id == "{escaped}"')
        self.collection.flush()
        return int(getattr(result, "delete_count", 0))


class InMemoryVectorStore:
    """Small vector store for unit tests and local pipeline smoke tests."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def upsert(self, records: list[dict[str, Any]]) -> None:
        """Replace records by their stable primary key."""
        for record in records:
            self.records[str(record["id"])] = record

    def delete_by_document(self, document_id: int | str) -> int:
        """Remove all records carrying the requested document id."""
        keys = [
            key for key, value in self.records.items() if value["doc_id"] == str(document_id)
        ]
        for key in keys:
            del self.records[key]
        return len(keys)
