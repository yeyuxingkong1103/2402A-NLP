import json
import logging
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.config import get_settings
from app.rag.types import ChunkRecord, RetrievedChunk

logger = logging.getLogger(__name__)


class LocalVectorStore:
    """Small persistent vector store used when Milvus is unavailable."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else get_settings().local_vector_index
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._records: dict[str, ChunkRecord] = {}
        self._load()

    def upsert(self, records: list[ChunkRecord]) -> None:
        with self._lock:
            for record in records:
                if record.embedding is None:
                    raise ValueError(f"chunk {record.id} has no embedding")
                self._records[record.id] = record
            self._save()
        logger.info("local vector upsert complete: %s records", len(records))

    def search(
        self,
        query_vector: list[float],
        top_k: int = 8,
        score_threshold: float = -1.0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        filters = filters or {}
        scored: list[RetrievedChunk] = []
        with self._lock:
            records = list(self._records.values())
        for record in records:
            if any(record.metadata.get(key) != value for key, value in filters.items()):
                continue
            vector = np.asarray(record.embedding or [], dtype=np.float32)
            if vector.size == 0:
                continue
            denominator = float(query_norm * np.linalg.norm(vector))
            score = float(np.dot(query, vector) / denominator) if denominator else 0.0
            if score >= score_threshold:
                scored.append(
                    RetrievedChunk(
                        chunk=record,
                        score=score,
                        dense_score=score,
                        retrieval_method="vector",
                    )
                )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def all_records(self) -> list[ChunkRecord]:
        with self._lock:
            return list(self._records.values())

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            ids = [key for key, value in self._records.items() if value.document_id == document_id]
            for key in ids:
                del self._records[key]
            if ids:
                self._save()
        return len(ids)

    def health(self) -> dict[str, Any]:
        return {"backend": "local", "records": len(self._records), "ready": True}

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            for item in payload:
                record = ChunkRecord(
                    id=item["id"],
                    document_id=item["document_id"],
                    text=item["text"],
                    source=item["source"],
                    metadata=item.get("metadata", {}),
                    embedding=item.get("embedding"),
                )
                self._records[record.id] = record
            logger.info("local vector index loaded: %s records", len(self._records))
        except Exception:
            logger.exception("local vector index load failed: %s", self.path)

    def _save(self) -> None:
        payload = [
            {
                "id": record.id,
                "document_id": record.document_id,
                "text": record.text,
                "source": record.source,
                "metadata": record.metadata,
                "embedding": record.embedding,
            }
            for record in self._records.values()
        ]
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(self.path)


class MilvusVectorStore:
    """Milvus adapter. The local store is used by the factory if this adapter cannot connect."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection_name = settings.milvus_collection
        self._client = None
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from pymilvus import MilvusClient  # type: ignore

        self._client = MilvusClient(uri=self.settings.milvus_uri, token=self.settings.milvus_token or None)
        if not self._client.has_collection(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                dimension=self.settings.embedding_dimension,
                primary_field_name="id",
                id_type="string",
                max_length=256,
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False,
                enable_dynamic_field=True,
            )
        logger.info("milvus collection ready: %s", self.collection_name)

    def upsert(self, records: list[ChunkRecord]) -> None:
        data = [
            {
                "id": record.id,
                "vector": record.embedding,
                "text": record.text,
                "document_id": record.document_id,
                "source": record.source,
                "role_id": str(record.metadata.get("role_id", "global")),
                "metadata": json.dumps(record.metadata, ensure_ascii=False),
            }
            for record in records
        ]
        self._client.upsert(collection_name=self.collection_name, data=data)
        logger.info("milvus upsert complete: %s records", len(records))

    def search(
        self,
        query_vector: list[float],
        top_k: int = 8,
        score_threshold: float = -1.0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        expr = None
        if filters:
            clauses = [f'{key} == "{value}"' for key, value in filters.items()]
            expr = " and ".join(clauses)
        result = self._client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            filter=expr,
            output_fields=["text", "document_id", "source", "role_id", "metadata"],
        )
        hits = result[0] if result else []
        output: list[RetrievedChunk] = []
        for hit in hits:
            entity = hit.get("entity", {})
            score = float(hit.get("distance", 0.0))
            if score < score_threshold:
                continue
            metadata = entity.get("metadata", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata or "{}")
            if entity.get("role_id") is not None:
                metadata.setdefault("role_id", entity.get("role_id"))
            record = ChunkRecord(
                id=str(hit.get("id")),
                document_id=str(entity.get("document_id", "")),
                text=str(entity.get("text", "")),
                source=str(entity.get("source", "")),
                metadata=metadata,
            )
            output.append(
                RetrievedChunk(chunk=record, score=score, dense_score=score, retrieval_method="vector")
            )
        return output

    def all_records(self) -> list[ChunkRecord]:
        try:
            rows = self._client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["id", "text", "document_id", "source", "role_id", "metadata"],
                limit=16384,
            )
        except Exception:
            logger.exception("milvus lexical index refresh failed")
            return []
        records: list[ChunkRecord] = []
        for row in rows or []:
            metadata = row.get("metadata", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata or "{}")
            if row.get("role_id") is not None:
                metadata.setdefault("role_id", row.get("role_id"))
            records.append(
                ChunkRecord(
                    id=str(row.get("id")),
                    document_id=str(row.get("document_id", "")),
                    text=str(row.get("text", "")),
                    source=str(row.get("source", "")),
                    metadata=metadata,
                )
            )
        return records

    def delete_document(self, document_id: str) -> int:
        result = self._client.delete(
            collection_name=self.collection_name,
            filter=f'document_id == "{document_id}"',
        )
        return int(result.get("delete_count", 0)) if isinstance(result, dict) else 0

    def health(self) -> dict[str, Any]:
        return {"backend": "milvus", "collection": self.collection_name, "ready": True}


def build_vector_store(settings: Settings) -> LocalVectorStore | MilvusVectorStore:
    if settings.milvus_enabled:
        try:
            return MilvusVectorStore(settings)
        except Exception:
            logger.exception("milvus unavailable; falling back to local vector store")
    return LocalVectorStore(settings.local_vector_index)
