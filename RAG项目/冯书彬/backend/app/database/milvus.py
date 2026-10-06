import logging
from datetime import datetime, timezone
from typing import Any

from backend.app.core.config import AppSettings, settings
from backend.app.models.knowledge_base import KnowledgeMaterial
from backend.app.rag.result_merger import RetrievalFilters

logger = logging.getLogger(__name__)


class MilvusConfigurationError(RuntimeError):
    # Milvus 配置缺失时抛出受控异常，避免生产环境静默退回空检索。
    pass


class MilvusUnavailableError(RuntimeError):
    # Milvus 连接或查询失败时统一转为受控异常，调用方只记录脱敏原因。
    pass


class MilvusVectorStore:
    def __init__(self, uri: str, collection_name: str, timeout_seconds: float = 2.0) -> None:
        if not uri:
            raise MilvusConfigurationError("MILVUS_URI is required")
        self.uri = uri
        self.collection_name = collection_name
        self.timeout_seconds = timeout_seconds
        self._client = None

    @property
    def client(self):
        # 延迟创建客户端，避免单元测试导入模块时触发真实网络连接。
        if self._client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:
                raise MilvusUnavailableError("pymilvus is not installed") from exc
            try:
                self._client = MilvusClient(uri=self.uri, timeout=self.timeout_seconds)
            except Exception as exc:
                raise MilvusUnavailableError(type(exc).__name__) from exc
        return self._client

    def ping(self) -> bool:
        try:
            self.client.list_collections(timeout=self.timeout_seconds)
        except Exception as exc:
            logger.warning("Milvus 连接检查失败", extra={"error_type": type(exc).__name__})
            raise MilvusUnavailableError(type(exc).__name__) from exc
        logger.info("Milvus 连接检查通过")
        return True

    def ensure_collection(self, dimension: int) -> None:
        try:
            if self.client.has_collection(self.collection_name, timeout=self.timeout_seconds):
                return
            schema, index_params = _build_collection_schema(self.client, dimension)
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            logger.warning("Milvus 集合初始化失败", extra={"error_type": type(exc).__name__, "collection": self.collection_name})
            raise MilvusUnavailableError(type(exc).__name__) from exc
        logger.info("Milvus 集合初始化完成", extra={"collection": self.collection_name, "dimension": dimension})

    def upsert(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        try:
            result = self.client.upsert(collection_name=self.collection_name, data=rows, timeout=self.timeout_seconds)
        except AttributeError:
            result = self.client.insert(collection_name=self.collection_name, data=rows, timeout=self.timeout_seconds)
        except Exception as exc:
            logger.warning("Milvus 向量写入失败", extra={"error_type": type(exc).__name__, "row_count": len(rows)})
            raise MilvusUnavailableError(type(exc).__name__) from exc
        logger.info("Milvus 向量写入完成", extra={"row_count": len(rows)})
        return int(result.get("upsert_count") or result.get("insert_count") or len(rows))

    def search(self, vector: list[float], top_k: int, filters: RetrievalFilters | None = None) -> list[dict[str, Any]]:
        expr = _filters_to_expr(filters)
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[vector],
                anns_field="vector",
                search_params={"metric_type": "COSINE"},
                limit=top_k,
                filter=expr,
                output_fields=["material_id", "version_id", "text", "metadata"],
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            logger.warning("Milvus 向量检索失败", extra={"error_type": type(exc).__name__, "top_k": top_k})
            raise MilvusUnavailableError(type(exc).__name__) from exc
        rows = [_hit_to_row(hit) for hit in (results[0] if results else [])]
        logger.info("Milvus 向量检索完成", extra={"result_count": len(rows), "top_k": top_k})
        return rows


def create_milvus_store(app_settings: AppSettings = settings) -> MilvusVectorStore:
    return MilvusVectorStore(uri=app_settings.MILVUS_URI, collection_name=app_settings.MILVUS_COLLECTION)


def check_milvus_readiness(app_settings: AppSettings = settings) -> tuple[bool, str | None]:
    if not app_settings.MILVUS_URI:
        return False, "missing uri"
    try:
        create_milvus_store(app_settings).ping()
    except (MilvusConfigurationError, MilvusUnavailableError) as exc:
        return False, type(exc).__name__
    return True, None


def _build_collection_schema(client: Any, dimension: int):
    from pymilvus import DataType

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("material_id", DataType.VARCHAR, max_length=128)
    schema.add_field("version_id", DataType.VARCHAR, max_length=128)
    schema.add_field("relationship_type", DataType.VARCHAR, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    schema.add_field("metadata", DataType.JSON)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
    index_params = client.prepare_index_params()
    index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
    return schema, index_params


def _filters_to_expr(filters: RetrievalFilters | None) -> str | None:
    if filters is None:
        return None
    clauses: list[str] = []
    if filters.relationship_type and filters.relationship_type != "general":
        clauses.append(f'relationship_type == "{_escape_expr_value(filters.relationship_type)}"')
    return " and ".join(clauses) or None


def _escape_expr_value(value: str) -> str:
    # Milvus 过滤表达式只允许双引号字符串，这里移除引号避免表达式注入。
    return value.replace('"', "")


def _hit_to_row(hit: Any) -> dict[str, Any]:
    entity = hit.get("entity", {}) if isinstance(hit, dict) else getattr(hit, "entity", {})
    metadata = entity.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": str(metadata)}
    material_id = str(entity.get("material_id", ""))
    version_id = str(entity.get("version_id", ""))
    text = str(entity.get("text", ""))
    material = _metadata_to_material(material_id, version_id, text, metadata)
    if material is not None:
        metadata = dict(metadata)
        metadata["material"] = material
    return {
        "id": str(hit.get("id", "")) if isinstance(hit, dict) else str(getattr(hit, "id", "")),
        "material_id": material_id,
        "version_id": version_id,
        "text": text,
        "score": float(hit.get("distance", 0.0)) if isinstance(hit, dict) else float(getattr(hit, "distance", 0.0)),
        "metadata": metadata,
    }


def _metadata_to_material(material_id: str, version_id: str, text: str, metadata: dict[str, Any]) -> KnowledgeMaterial | None:
    required = ["source_url", "publisher", "material_type", "status", "searchable", "effective_from"]
    if not all(key in metadata for key in required):
        return None
    material = KnowledgeMaterial(
        id=material_id,
        snapshot_id=str(metadata.get("snapshot_id") or material_id),
        source_url=str(metadata["source_url"]),
        publisher=str(metadata["publisher"]),
        material_type=str(metadata["material_type"]),
        raw_text=text,
        attachments=[],
        status=str(metadata["status"]),
        searchable=bool(metadata["searchable"]),
        created_at=_metadata_datetime(metadata.get("created_at")),
        updated_at=_metadata_datetime(metadata.get("updated_at")),
        effective_from=str(metadata["effective_from"]) if metadata.get("effective_from") else None,
    )
    setattr(material, "version_id", version_id)
    relationship_types = metadata.get("relationship_types")
    if relationship_types:
        setattr(material, "relationship_types", relationship_types)
    return material


def _metadata_datetime(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now(timezone.utc)
