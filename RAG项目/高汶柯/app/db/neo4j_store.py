"""Neo4j 知识图谱检索（可选，默认关闭）。

未启用时检索返回空列表，使多路召回可平滑降级。
"""
from __future__ import annotations

from app.config import settings
from app.logging_conf import log


class Neo4jStore:
    def __init__(self) -> None:
        self._driver = None

    def _conn(self):
        if self._driver is not None:
            return self._driver
        if not settings.neo4j_enabled:
            raise RuntimeError("Neo4j 未启用")
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        self._driver.verify_connectivity()
        log.info("Neo4j 连接成功: %s", settings.neo4j_uri)
        return self._driver

    @property
    def enabled(self) -> bool:
        try:
            self._conn()
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Neo4j 不可用: %s", exc)
            return False

    def search(self, query: str, domain: str | None = None, top_k: int = 5) -> list[dict]:
        """按实体名做模糊匹配，返回关联知识片段。"""
        if not settings.neo4j_enabled:
            return []
        try:
            driver = self._conn()
            cypher = (
                "MATCH (n:Entity)-[r]-(m) "
                "WHERE n.name CONTAINS $q "
                + ("AND ($domain IS NULL OR n.domain = $domain) " if domain else "")
                + "RETURN n.name AS source, type(r) AS rel, coalesce(m.name, m.text) AS text "
                "LIMIT $k"
            )
            with driver.session() as session:
                rows = session.run(cypher, q=query, domain=domain, k=top_k)
                return [
                    {"text": f"{r['source']} -[{r['rel']}]- {r['text']}", "source": "neo4j",
                     "domain": domain or "general", "chunk_type": "graph"}
                    for r in rows
                ]
        except Exception as exc:  # noqa: BLE001
            log.warning("Neo4j 检索失败: %s", exc)
            return []


neo4j_store = Neo4jStore()
