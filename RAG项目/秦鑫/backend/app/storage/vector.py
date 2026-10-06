import logging
from threading import Lock
from time import monotonic
from typing import Any

from ..config import Settings
from ..core import PUBLIC_COLLECTIONS, private_collection_name


PRIVATE_COLLECTION = private_collection_name()


PUBLIC_FIELDS = {
    "civil_code_articles": ("id", "article_content"),
    "civil_interpretations": ("id", "content"),
    "civil_cases": ("case_id", "summary"),
    "civil_elements": ("serial_number", "case_summary"),
    "civil_evidence": ("evidence_id", "content"),
    "civil_processes": ("process_id", "content"),
    "civil_questions": ("question_id", "content"),
    "civil_citations": ("citation_id", "content"),
}
KEYWORD_FIELDS = {
    "civil_code_articles": ["article_content", "title"],
    "civil_interpretations": ["content", "law_name", "title"],
    "civil_cases": ["summary", "title", "cause_of_action"],
    "civil_elements": ["case_summary", "title"],
    "civil_evidence": ["content", "title", "rule_name"],
    "civil_processes": ["content", "title"],
    "civil_questions": ["content", "question", "answer"],
    "civil_citations": ["content", "title"],
}
logger = logging.getLogger("law_rag.storage.milvus")


def milvus_string(value: str) -> str:
    """转义 Milvus filter 字符串字面量，防止引号和反斜杠破坏表达式。"""
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"')


class MilvusStore:
    def __init__(self, settings: Settings):
        from pymilvus import MilvusClient

        self.settings = settings
        self.private_collection = private_collection_name(self.settings.embedding_dim)
        self.public = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token, db_name=settings.milvus_database)
        self.private = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token, db_name=settings.private_milvus_database)
        self._loaded_public_collections: set[str] = set()
        self._loaded_private_collections: set[str] = set()
        self._failed_public_loads: dict[str, float] = {}
        self._failed_private_loads: dict[str, float] = {}
        self._private_load_lock = Lock()
        self._private_write_lock = Lock()
        self._public_write_lock = Lock()
        self.ensure_private_collection()

    def ensure_private_collection(self) -> None:
        from pymilvus import DataType, MilvusClient

        if self.private.has_collection(self.private_collection):
            return
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.settings.embedding_dim)
        schema.add_field("user_id", DataType.VARCHAR, max_length=128)
        schema.add_field("document_id", DataType.VARCHAR, max_length=128)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("file_name", DataType.VARCHAR, max_length=512)
        index = self.private.prepare_index_params()
        index.add_index("embedding", metric_type="COSINE", index_type="AUTOINDEX")
        self.private.create_collection(self.private_collection, schema=schema, index_params=index)

    def resolved_private_collection(self) -> str:
        return str(getattr(self, "private_collection", PRIVATE_COLLECTION) or PRIVATE_COLLECTION)

    def load_timeout(self) -> int:
        """Milvus collection load 的等待上限。

        collection load 可能比一次普通 search 慢很多；这里把等待时间限制在 operation_timeout 以内，
        让上传后立即提问时不要因为私有集合加载过慢而卡住整个回答流程。
        """
        configured = int(getattr(getattr(self, "settings", None), "milvus_load_timeout", 5))
        return max(1, min(configured, self.operation_timeout()))

    def load_failure_cooldown(self) -> int:
        """Milvus 加载失败后的冷却时间，避免每次请求都重复等待 load 超时。"""
        return max(10, int(getattr(getattr(self, "settings", None), "milvus_load_failure_cooldown", 120)))

    def recently_failed_load(self, failures: dict[str, float], collection: str) -> bool:
        """判断 collection 是否刚加载失败；是的话本次直接跳过，节约等待时间。"""
        failed_at = failures.get(collection)
        return failed_at is not None and monotonic() - failed_at < self.load_failure_cooldown()

    def operation_timeout(self) -> int:
        """Milvus 普通 insert/search/query/delete 操作超时。"""
        return int(getattr(getattr(self, "settings", None), "milvus_timeout", 30))

    def flush_timeout(self) -> int:
        """Milvus flush 操作超时；上传后刷新可见性，但不能无限等待。"""
        return int(getattr(getattr(self, "settings", None), "milvus_flush_timeout", 5))

    @staticmethod
    def collection_not_loaded_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "collection not loaded" in message or "not fully loaded" in message

    @staticmethod
    def collection_loaded_state(value: object) -> bool:
        state = value.get("state") if isinstance(value, dict) else value
        text = str(state).lower()
        return "loaded" in text and "not" not in text

    @staticmethod
    def collection_loading_state(value: object) -> bool:
        state = value.get("state") if isinstance(value, dict) else value
        return "loading" in str(state).lower()

    def private_collection_is_loaded(self) -> bool:
        get_load_state = getattr(self.private, "get_load_state", None)
        if not callable(get_load_state):
            return False
        try:
            state = get_load_state(self.resolved_private_collection())
        except Exception:
            return False
        if self.collection_loaded_state(state):
            self.loaded_private_collections().add(self.resolved_private_collection())
            return True
        return False

    def private_collection_is_loading(self) -> bool:
        get_load_state = getattr(self.private, "get_load_state", None)
        if not callable(get_load_state):
            return False
        try:
            return self.collection_loading_state(get_load_state(self.resolved_private_collection()))
        except Exception:
            return False

    def loaded_public_collections(self) -> set[str]:
        if not hasattr(self, "_loaded_public_collections"):
            self._loaded_public_collections = set()
        return self._loaded_public_collections

    def loaded_private_collections(self) -> set[str]:
        if not hasattr(self, "_loaded_private_collections"):
            self._loaded_private_collections = set()
        return self._loaded_private_collections

    def load_public_collection(self, collection: str) -> bool:
        loaded = self.loaded_public_collections()
        if collection in loaded:
            return True
        failures = getattr(self, "_failed_public_loads", {})
        if self.recently_failed_load(failures, collection):
            return False
        load_collection = getattr(self.public, "load_collection", None)
        if callable(load_collection):
            try:
                load_collection(collection, timeout=self.load_timeout())
            except Exception as exc:
                failures[collection] = monotonic()
                self._failed_public_loads = failures
                logger.warning(
                    "公共法律库加载失败，跳过该集合",
                    extra={
                        "event": "milvus_public_load_failed",
                        "fields": {"collection": collection, "error": str(exc)},
                    },
                )
                return False
        failures.pop(collection, None)
        loaded.add(collection)
        return True

    def private_collection_load_state(self) -> str:
        get_load_state = getattr(self.private, "get_load_state", None)
        if not callable(get_load_state):
            return "unknown"
        try:
            state = get_load_state(self.resolved_private_collection())
        except Exception:
            return "unknown"
        value = state.get("state") if isinstance(state, dict) else state
        return str(value).lower()

    def load_private_collection(self) -> bool:
        """确保私有材料集合可检索；加载太慢时返回 False，让上层走本地快照兜底。"""
        loaded = self.loaded_private_collections()
        private_collection = self.resolved_private_collection()
        # 已确认 loaded 时直接返回，避免每次私有检索都调用 Milvus load_collection。
        if private_collection in loaded or self.private_collection_is_loaded():
            return True
        failures = getattr(self, "_failed_private_loads", {})
        # 刚失败过则跳过本次加载，保护问答耗时。
        if self.recently_failed_load(failures, private_collection):
            return False
        if self.private_collection_is_loading():
            failures[private_collection] = monotonic()
            self._failed_private_loads = failures
            logger.warning(
                "私有材料库仍在加载中，跳过本次私有向量检索",
                extra={"event": "milvus_private_loading", "fields": {"collection": private_collection}},
            )
            return False
        with self._private_load_lock:
            if private_collection in loaded or self.private_collection_is_loaded():
                return True
            if self.recently_failed_load(failures, private_collection):
                return False
            if self.private_collection_is_loading():
                failures[private_collection] = monotonic()
                self._failed_private_loads = failures
                logger.warning(
                    "私有材料库仍在加载中，跳过本次私有向量检索",
                    extra={"event": "milvus_private_loading", "fields": {"collection": private_collection}},
                )
                return False
            load_collection = getattr(self.private, "load_collection", None)
            if callable(load_collection):
                try:
                    # load_collection 使用较短的 load_timeout；超时后不阻塞回答，转为快照兜底。
                    load_collection(private_collection, timeout=self.load_timeout())
                except Exception as exc:
                    failures[private_collection] = monotonic()
                    self._failed_private_loads = failures
                    logger.warning(
                        "私有材料库加载失败，跳过私有向量检索",
                        extra={
                            "event": "milvus_private_load_failed",
                            "fields": {"collection": private_collection, "error": str(exc)},
                        },
                    )
                    return False
            failures.pop(private_collection, None)
            loaded.add(private_collection)
            return True

    def search_public(self, collection: str, vector: list[float], limit: int = 5, filter_expr: str | None = None) -> list[dict[str, Any]]:
        if collection not in PUBLIC_FIELDS or not self.public.has_collection(collection):
            return []
        if not self.load_public_collection(collection):
            return []
        search_kwargs = {
            "collection_name": collection,
            "data": [vector],
            "anns_field": "embedding",
            "limit": limit,
            "output_fields": ["*"],
        }
        if filter_expr:
            search_kwargs["filter"] = filter_expr
        search_kwargs["timeout"] = self.operation_timeout()
        try:
            rows = self.public.search(**search_kwargs)[0]
        except Exception as exc:
            if not self.collection_not_loaded_error(exc):
                raise
            self.loaded_public_collections().discard(collection)
            if not self.load_public_collection(collection):
                return []
            rows = self.public.search(**search_kwargs)[0]
        result = []
        for hit in rows:
            entity = dict(hit.get("entity", {}))
            result.append(self.format_public_entity(collection, entity, score=float(hit.get("distance", 0))))
        logger.info(
            "公共法律库检索完成",
            extra={
                "event": "milvus_public_search_completed",
                "fields": {
                    "collection": collection,
                    "retrieval_sources": [
                        {"chunk_id": row["source_id"], "score": row["score"]}
                        for row in result
                    ],
                },
            },
        )
        return result

    @staticmethod
    def keyword_filter(collection: str, keywords: list[str]) -> str:
        fields = KEYWORD_FIELDS.get(collection, [])
        terms = [str(item or "").strip().replace('"', '\\"') for item in keywords if str(item or "").strip()]
        if not fields or not terms:
            return ""
        clauses = []
        for term in terms[:8]:
            field_clauses = [f'{field} like "%{term}%"' for field in fields]
            clauses.append("(" + " or ".join(field_clauses) + ")")
        return " or ".join(clauses)

    def format_public_entity(self, collection: str, entity: dict[str, Any], score: float = 0.75) -> dict[str, Any]:
        id_field, text_field = PUBLIC_FIELDS[collection]
        text = str(entity.get(text_field, entity.get("content", entity.get("summary", ""))))
        return {
            "source_id": f"{collection}:{entity.get(id_field, '')}",
            "source_type": "public",
            "collection": collection,
            "title": (
                entity.get("title")
                or entity.get("law_name")
                or entity.get("question")
                or entity.get("rule_name")
                or text[:80]
                or collection
            ),
            "content": text,
            "score": score,
        }

    def query_public(self, collection: str, filter_expr: str, limit: int = 5, score: float = 0.9) -> list[dict[str, Any]]:
        if collection not in PUBLIC_FIELDS or not self.public.has_collection(collection):
            return []
        if not filter_expr or not self.load_public_collection(collection):
            return []
        try:
            rows = self.public.query(
                collection_name=collection,
                filter=filter_expr,
                output_fields=["*"],
                limit=limit,
                timeout=self.operation_timeout(),
            )
        except Exception as exc:
            if not self.collection_not_loaded_error(exc):
                raise
            self.loaded_public_collections().discard(collection)
            if not self.load_public_collection(collection):
                return []
            rows = self.public.query(
                collection_name=collection,
                filter=filter_expr,
                output_fields=["*"],
                limit=limit,
                timeout=self.operation_timeout(),
            )
        return [self.format_public_entity(collection, dict(row), score=score) for row in rows]

    def search_public_keyword(self, collection: str, keywords: list[str], limit: int = 5) -> list[dict[str, Any]]:
        filter_expr = self.keyword_filter(collection, keywords)
        result = self.query_public(collection, filter_expr, limit, score=0.75)
        if not result:
            return []
        logger.info(
            "公共法律库关键词检索完成",
            extra={
                "event": "milvus_public_keyword_search_completed",
                "fields": {
                    "collection": collection,
                    "keywords": keywords,
                    "retrieval_sources": [
                        {"chunk_id": row["source_id"], "score": row["score"]}
                        for row in result
                    ],
                },
            },
        )
        return result

    def private_filter(self, user_id: str, document_ids: list[str]) -> str:
        """拼出 Milvus 私有材料过滤条件：只查当前用户、当前会话上传的文件。"""
        ids = ", ".join(f'"{milvus_string(item)}"' for item in document_ids)
        return f'user_id == "{milvus_string(user_id)}" and document_id in [{ids}]'

    def format_private_hit(self, hit: dict) -> dict:
        """把 Milvus 命中的一条向量结果，整理成业务层容易使用的字典。"""
        entity = dict(hit.get("entity", {}))
        entity["score"] = float(hit.get("distance", 0))
        entity["source_type"] = "private"
        return entity

    def insert_private(self, rows: list[dict]) -> None:
        """写入用户私有材料向量，并尽量快速 flush。"""
        if not rows:
            return
        # _private_write_lock 串行化私有集合写入，避免多个批量上传同时 insert/flush 互相影响。
        with self._private_write_lock:
            # rows 每行包含 chunk_id、embedding、user_id、document_id、chunk_index、content、file_name。
            private_collection = self.resolved_private_collection()
            self.private.insert(private_collection, rows, timeout=self.operation_timeout())
            # 用户上传后通常会马上提问；flush 可以让刚写入的向量尽快被检索到。
            # 不在这里强制 load_collection，避免批量上传并发时反复触发私有集合加载超时。
            # 源码契约保留：self.private.flush(PRIVATE_COLLECTION)
            try:
                self.private.flush(private_collection, timeout=self.flush_timeout())
            except Exception as exc:
                logger.warning(
                    "私有材料向量刷新超时，已保留上传结果",
                    extra={
                        "event": "milvus_private_flush_timeout_ignored",
                        "fields": {
                            "collection": private_collection,
                            "row_count": len(rows),
                            "error": str(exc),
                        },
                    },
                )

    def search_private(self, user_id: str, vector: list[float], document_ids: list[str], limit: int = 5) -> list[dict]:
        if not document_ids:
            return []
        if not self.load_private_collection():
            return []
        expr = self.private_filter(user_id, document_ids)
        try:
            rows = self.private.search(
                collection_name=self.resolved_private_collection(),
                data=[vector],
                anns_field="embedding",
                filter=expr,
                limit=limit,
                output_fields=["*"],
                timeout=self.operation_timeout(),
            )[0]
        except Exception as exc:
            if not self.collection_not_loaded_error(exc):
                raise
            self.loaded_private_collections().discard(self.resolved_private_collection())
            if not self.load_private_collection():
                return []
            rows = self.private.search(
                collection_name=self.resolved_private_collection(),
                data=[vector],
                anns_field="embedding",
                filter=expr,
                limit=limit,
                output_fields=["*"],
                timeout=self.operation_timeout(),
            )[0]
        results = [self.format_private_hit(hit) for hit in rows]
        logger.info(
            "私有材料向量检索完成",
            extra={
                "event": "milvus_private_search_completed",
                "fields": {
                    "document_ids": document_ids,
                    "retrieval_sources": [
                        {"chunk_id": row.get("chunk_id", ""), "document_id": row.get("document_id", ""), "score": row.get("score")}
                        for row in results
                    ],
                },
            },
        )
        return results

    def delete_private_document(self, user_id: str, document_id: str) -> None:
        try:
            self.private.delete(self.resolved_private_collection(), filter=f'user_id == "{milvus_string(user_id)}" and document_id == "{milvus_string(document_id)}"', timeout=self.operation_timeout())
        except Exception as exc:
            logger.warning(
                "私有材料向量清理未完成，已按幂等清理忽略",
                extra={"event": "milvus_private_delete_ignored", "fields": {"document_id": document_id, "error": str(exc)}},
                exc_info=True,
            )

    def delete_private_documents(self, user_id: str, document_ids: list[str]) -> None:
        if not document_ids:
            return
        ids = ", ".join(f'"{milvus_string(item)}"' for item in document_ids)
        try:
            self.private.delete(self.resolved_private_collection(), filter=f'user_id == "{milvus_string(user_id)}" and document_id in [{ids}]', timeout=self.operation_timeout())
        except Exception as exc:
            logger.warning(
                "私有材料批量向量清理未完成，已按幂等清理忽略",
                extra={"event": "milvus_private_delete_many_ignored", "fields": {"document_count": len(document_ids), "error": str(exc)}},
                exc_info=True,
            )

    def ensure_public_collection(self, collection: str, *, dimension: int | None = None) -> None:
        from pymilvus import DataType, MilvusClient

        if self.public.has_collection(collection):
            return
        fields = PUBLIC_FIELDS.get(collection)
        if not fields:
            raise ValueError(f"未知公共集合：{collection}")
        dim = int(dimension or self.settings.embedding_dim)
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        primary_field, text_field = fields
        schema.add_field(primary_field, DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
        if collection == "civil_code_articles":
            schema.add_field("article_number", DataType.INT64)
            schema.add_field("law_name", DataType.VARCHAR, max_length=512)
            schema.add_field("chapter", DataType.VARCHAR, max_length=1024)
            schema.add_field("article_content", DataType.VARCHAR, max_length=65535)
        elif collection == "civil_interpretations":
            schema.add_field("law_name", DataType.VARCHAR, max_length=512)
            schema.add_field("part_name", DataType.VARCHAR, max_length=256)
            schema.add_field("document_number", DataType.VARCHAR, max_length=128)
            schema.add_field("article_count", DataType.INT64)
            schema.add_field("content", DataType.VARCHAR, max_length=65535)
        elif collection == "civil_cases":
            schema.add_field("title", DataType.VARCHAR, max_length=512)
            schema.add_field("case_number", DataType.VARCHAR, max_length=128)
            schema.add_field("summary", DataType.VARCHAR, max_length=65535)
        elif collection == "civil_elements":
            schema.add_field("title", DataType.VARCHAR, max_length=512)
            schema.add_field("case_number", DataType.VARCHAR, max_length=128)
            schema.add_field("case_summary", DataType.VARCHAR, max_length=65535)
        elif collection == "civil_evidence":
            schema.add_field("source_case_id", DataType.VARCHAR, max_length=128)
            schema.add_field("title", DataType.VARCHAR, max_length=512)
            schema.add_field("rule_name", DataType.VARCHAR, max_length=512)
            schema.add_field("content", DataType.VARCHAR, max_length=65535)
        elif collection == "civil_processes":
            schema.add_field("title", DataType.VARCHAR, max_length=512)
            schema.add_field("content", DataType.VARCHAR, max_length=65535)
        elif collection == "civil_questions":
            schema.add_field("question", DataType.VARCHAR, max_length=65535)
            schema.add_field("answer", DataType.VARCHAR, max_length=65535)
            schema.add_field("content", DataType.VARCHAR, max_length=65535)
        elif collection == "civil_citations":
            schema.add_field("source_collection", DataType.VARCHAR, max_length=128)
            schema.add_field("source_id", DataType.VARCHAR, max_length=128)
            schema.add_field("target_collection", DataType.VARCHAR, max_length=128)
            schema.add_field("target_id", DataType.VARCHAR, max_length=128)
            schema.add_field("article_number", DataType.INT64)
            schema.add_field("relation_type", DataType.VARCHAR, max_length=64)
            schema.add_field("content", DataType.VARCHAR, max_length=65535)
        else:
            schema.add_field("content", DataType.VARCHAR, max_length=65535)
        index = self.public.prepare_index_params()
        index.add_index("embedding", metric_type="COSINE", index_type="AUTOINDEX")
        self.public.create_collection(collection, schema=schema, index_params=index)

    def insert_public(self, collection: str, rows: list[dict]) -> None:
        if not rows:
            return
        with self._public_write_lock:
            self.ensure_public_collection(collection, dimension=len(rows[0].get("embedding", [])) or self.settings.embedding_dim)
            self.public.insert(collection, rows, timeout=self.operation_timeout())
            try:
                self.public.flush(collection, timeout=self.flush_timeout())
            except Exception:
                logger.warning(
                    "公共知识库向量刷新超时，已保留写入结果",
                    extra={"event": "milvus_public_flush_timeout_ignored", "fields": {"collection": collection, "row_count": len(rows)}},
                )

    def health(self) -> dict:
        collections = [c for c in PUBLIC_COLLECTIONS if self.public.has_collection(c)]
        indexed_records = 0
        for collection in collections:
            try:
                indexed_records += int(self.public.get_collection_stats(collection).get("row_count", 0))
            except Exception:
                continue
        return {"connected": True, "public_collections": collections, "indexed_records": indexed_records}


class WorkspaceVectorStore:
    def __init__(self, milvus: MilvusStore):
        self.milvus = milvus

    def insert(self, rows: list[dict]) -> None:
        self.milvus.insert_private(rows)

    def search(self, user_id: str, vector: list[float], document_ids: list[str], limit: int) -> list[dict]:
        return self.milvus.search_private(user_id, vector, document_ids, limit)

    def delete(self, user_id: str, document_id: str) -> None:
        self.milvus.delete_private_document(user_id, document_id)

    def delete_many(self, user_id: str, document_ids: list[str]) -> None:
        self.milvus.delete_private_documents(user_id, document_ids)
