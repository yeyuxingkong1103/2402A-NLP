"""Milvus connection and schema setup for long-term memory."""

from __future__ import annotations

from typing import Any


class MilvusMemoryClient:
    """Create the three tenant-scoped long-term memory collections."""

    def __init__(self, *, uri: str, dimension: int, token: str | None = None, alias: str = "long_term_memory", collection_prefix: str = "rag_v1") -> None:
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
        except ImportError as exc:
            raise RuntimeError("pymilvus is required for long-term memory") from exc
        self._Collection = Collection
        self._CollectionSchema = CollectionSchema
        self._DataType = DataType
        self._FieldSchema = FieldSchema
        self._utility = utility
        self.alias = alias
        self.dimension = dimension
        connections.connect(alias=alias, uri=uri, token=token or "")
        self.collection_names = {
            "preferences": f"{collection_prefix}_user_preferences",
            "summaries": f"{collection_prefix}_conversation_summaries",
            "facts": f"{collection_prefix}_important_facts",
        }
        self._collections = self._ensure_collections()

    def get_collection(self, name: str) -> Any:
        """Return a configured collection by its physical name."""
        try:
            return self._collections[name]
        except KeyError as exc:
            raise KeyError(f"unknown long-term memory collection: {name}") from exc

    def _ensure_collections(self) -> dict[str, Any]:
        schemas = self._schemas()
        collections: dict[str, Any] = {}
        for key, name in self.collection_names.items():
            vector_field, schema = schemas[key]
            if self._utility.has_collection(name, using=self.alias):
                collection = self._Collection(name=name, using=self.alias)
                self._validate_dimension(collection, vector_field)
            else:
                collection = self._Collection(name=name, schema=schema, using=self.alias, shards_num=2)
                collection.create_index(
                    field_name=vector_field,
                    index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 256}},
                )
            collection.load()
            collections[name] = collection
        return collections

    def _schemas(self) -> dict[str, tuple[str, Any]]:
        field, data_type, schema = self._FieldSchema, self._DataType, self._CollectionSchema
        common = [
            field(name="id", dtype=data_type.INT64, is_primary=True, auto_id=True),
            field(name="user_id", dtype=data_type.INT64),
            field(name="tenant_id", dtype=data_type.INT64),
        ]
        preferences = schema(
            fields=common + [
                field(name="preference_type", dtype=data_type.VARCHAR, max_length=50),
                field(name="preference_text", dtype=data_type.VARCHAR, max_length=1000),
                field(name="preference_vector", dtype=data_type.FLOAT_VECTOR, dim=self.dimension),
                field(name="confidence", dtype=data_type.FLOAT),
                field(name="created_at", dtype=data_type.INT64),
                field(name="updated_at", dtype=data_type.INT64),
            ], description="User preferences for personalized support", enable_dynamic_field=False,
        )
        summaries = schema(
            fields=common + [
                field(name="session_id", dtype=data_type.VARCHAR, max_length=100),
                field(name="summary_text", dtype=data_type.VARCHAR, max_length=2000),
                field(name="summary_vector", dtype=data_type.FLOAT_VECTOR, dim=self.dimension),
                field(name="turn_count", dtype=data_type.INT32),
                field(name="key_intents", dtype=data_type.ARRAY, element_type=data_type.VARCHAR, max_capacity=10, max_length=50),
                field(name="created_at", dtype=data_type.INT64),
                field(name="session_start", dtype=data_type.INT64),
                field(name="session_end", dtype=data_type.INT64),
            ], description="Cross-session conversation summaries", enable_dynamic_field=False,
        )
        facts = schema(
            fields=common + [
                field(name="fact_type", dtype=data_type.VARCHAR, max_length=50),
                field(name="fact_text", dtype=data_type.VARCHAR, max_length=1000),
                field(name="fact_vector", dtype=data_type.FLOAT_VECTOR, dim=self.dimension),
                field(name="source_session_id", dtype=data_type.VARCHAR, max_length=100),
                field(name="confidence", dtype=data_type.FLOAT),
                field(name="created_at", dtype=data_type.INT64),
                field(name="expires_at", dtype=data_type.INT64),
            ], description="Important facts supplied by a user", enable_dynamic_field=False,
        )
        return {"preferences": ("preference_vector", preferences), "summaries": ("summary_vector", summaries), "facts": ("fact_vector", facts)}

    def _validate_dimension(self, collection: Any, vector_field: str) -> None:
        existing = next((int(item.params.get("dim")) for item in collection.schema.fields if item.name == vector_field), None)
        if existing != self.dimension:
            raise RuntimeError(f"collection {collection.name} dimension mismatch: expected {self.dimension}, got {existing}")
