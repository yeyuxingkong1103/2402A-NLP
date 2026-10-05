from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pymilvus import DataType, MilvusClient

from .config import Settings


@dataclass
class SearchHit:
    chunk_id: str
    content: str
    page_number: int
    section_title: str
    document_id: str
    document_name: str
    score: float
    content_type: str = "text"


class VectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or None)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        name = self.settings.milvus_collection
        if self.client.has_collection(collection_name=name):
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field("document_id", DataType.VARCHAR, max_length=64)
        schema.add_field("document_name", DataType.VARCHAR, max_length=255)
        schema.add_field("document_version", DataType.VARCHAR, max_length=64)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("section_title", DataType.VARCHAR, max_length=255)
        schema.add_field("content_type", DataType.VARCHAR, max_length=32)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.settings.embedding_dimension)
        index = self.client.prepare_index_params()
        index.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(collection_name=name, schema=schema, index_params=index)

    def upsert(self, rows: list[dict[str, Any]]) -> None:
        if rows:
            self.client.insert(collection_name=self.settings.milvus_collection, data=rows)
            self.client.flush(self.settings.milvus_collection)

    def delete_document(self, document_id: str) -> None:
        self.client.delete(collection_name=self.settings.milvus_collection, filter=f'document_id == "{document_id}"')

    def search(self, vector: list[float], document_id: str | None, limit: int) -> list[SearchHit]:
        filter_expr = f'document_id == "{document_id}"' if document_id else None
        result = self.client.search(collection_name=self.settings.milvus_collection, data=[vector], limit=limit, filter=filter_expr, output_fields=["chunk_id", "document_id", "document_name", "page_number", "section_title", "content_type", "content"])[0]
        return [SearchHit(score=float(item["distance"]), **item["entity"]) for item in result]
