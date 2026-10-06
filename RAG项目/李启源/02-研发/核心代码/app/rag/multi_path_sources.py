"""Optional retrieval source adapters for multi-path recall."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from app.rag.multi_path_models import MultiSourceResult

logger = logging.getLogger(__name__)


class MySQLFullTextRetriever:
    """Search chunks through a MySQL FULLTEXT index."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def search(
        self, query: str, *, top_k: int = 20, filters: dict[str, Any] | None = None
    ) -> list[MultiSourceResult]:
        try:
            from sqlalchemy import text

            sql = (
                "SELECT c.chunk_id, c.content as text, c.source, c.summary, "
                "MATCH(c.content) AGAINST(:query IN NATURAL LANGUAGE MODE) as score "
                "FROM chunks c WHERE MATCH(c.content) AGAINST(:query IN NATURAL LANGUAGE MODE)"
            )
            params: dict[str, Any] = {"query": query, "top_k": top_k}
            if filters and "kb_id" in filters:
                sql += " AND c.document_id IN (SELECT id FROM documents WHERE knowledge_base_id = :kb_id)"
                params["kb_id"] = filters["kb_id"]
            sql += " ORDER BY score DESC LIMIT :top_k"
            with self.engine.connect() as connection:
                rows = connection.execute(text(sql), params).fetchall()
            return [
                MultiSourceResult(
                    chunk_id=row[0],
                    text=row[1],
                    source=row[2],
                    summary=row[3] or "",
                    score=float(row[4]),
                    retrieval_source="mysql",
                    metadata={"mysql_score": row[4]},
                )
                for row in rows
            ]
        except Exception as exc:
            logger.error("MySQL search failed: %s", exc)
            return []


class RedisCacheRetriever:
    """Read and write serialized query results in Redis."""

    def __init__(self, redis_client: Any) -> None:
        self.redis_client = redis_client

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[MultiSourceResult]:
        try:
            cached = self.redis_client.get(self._generate_cache_key(query, filters))
            if not cached:
                return []
            if isinstance(cached, bytes):
                cached = cached.decode("utf-8")
            return [
                MultiSourceResult(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    source=item["source"],
                    summary=item.get("summary", ""),
                    score=item["score"],
                    retrieval_source="redis",
                    metadata={"from_cache": True},
                )
                for item in json.loads(cached)[:top_k]
            ]
        except Exception as exc:
            logger.error("Redis search failed: %s", exc)
            return []

    def cache_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        ttl: int = 3600,
        filters: dict[str, Any] | None = None,
    ) -> None:
        try:
            self.redis_client.setex(
                self._generate_cache_key(query, filters), ttl, json.dumps(results, ensure_ascii=False)
            )
        except Exception as exc:
            logger.error("Redis cache write failed: %s", exc)

    @staticmethod
    def _generate_cache_key(
        query: str, filters: dict[str, Any] | None = None
    ) -> str:
        normalized = query.strip().lower()
        scope = json.dumps(filters or {}, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        digest = hashlib.sha256(f"{normalized}\0{scope}".encode()).hexdigest()
        return f"query_cache:{digest}"


class Neo4JGraphRetriever:
    """Optional graph-based retrieval adapter."""

    def __init__(self, driver: Any) -> None:
        self.driver = driver

    def search(
        self, query: str, *, top_k: int = 20, entity_types: list[str] | None = None
    ) -> list[MultiSourceResult]:
        try:
            cypher = (
                "MATCH (n:Entity) WHERE n.name CONTAINS $query "
                "OR n.description CONTAINS $query OPTIONAL MATCH (n)-[r]-(related:Entity) "
                "RETURN n.name as name, n.description as description, "
                "collect(related.name) as related_entities LIMIT $top_k"
            )
            with self.driver.session() as session:
                records = session.run(cypher, query=query, top_k=top_k)
                results = []
                for record in records:
                    text = f"{record['name']}: {record['description']}"
                    related = record["related_entities"]
                    if related:
                        text += f" (相关: {', '.join(related[:3])})"
                    results.append(
                        MultiSourceResult(
                            chunk_id=f"graph_{hashlib.md5(text.encode()).hexdigest()[:12]}",
                            text=text,
                            source="knowledge_graph",
                            score=0.8,
                            retrieval_source="neo4j",
                            summary=record["description"][:200],
                            metadata={"entity": record["name"]},
                        )
                    )
                return results
        except Exception as exc:
            logger.error("Neo4J search failed: %s", exc)
            return []


class InternetSearchRetriever:
    """Placeholder adapter for a future authorized internet source."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def search(self, query: str, *, top_k: int = 5) -> list[MultiSourceResult]:
        logger.info("Internet search not yet implemented")
        return []
