# -*- coding: utf-8 -*-
"""图谱检索的上层适配器：把 Neo4j 记录转成项目统一的证据契约。

自 ``graph_retrieval.py`` 拆出。依赖 ``graph_retrieval.DrugGraphRetriever``
（可注入），供混合检索 router 使用。
"""
from __future__ import annotations

import logging
from typing import Any

from src import config
from src.core.retrieval.base.retriever_base import BaseRetriever, RetrievalResult

from .graph_retrieval import DrugGraphRetriever

logger = logging.getLogger(__name__)


def normalize_graph_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a graph row to the project's unified evidence contract."""

    def first(*keys: str):
        for key in keys:
            value = record.get(key)
            if value not in (None, "", [], {}):
                return value
        return ""

    disease = first("disease")
    content = {
        "id": first("id"),
        "name": first("entity_name"),
        "dosage": first("usage"),
        "contraindication": first("contraindications"),
        "notice": first("description"),
        "related_diseases": [disease] if disease else [],
        "category": first("category", "categories"),
        "drug_class": first("normalized_type", "entity_type"),
        "applicable_scope": first("applicable_scope", "scopes"),
        "source": first("evidence_sources", "evidence_source"),
        "target_kind": first("target_kind"),
    }
    content = {
        key: value for key, value in content.items()
        if value not in (None, "", [], {})
    }
    content["raw"] = record
    return content


class HybridGraphRetriever(BaseRetriever):
    """Adapter used by the project's hybrid retrieval pipeline."""

    name = "graph"

    def __init__(
        self,
        graph_service: DrugGraphRetriever | None = None,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self.top_k = config.GRAPH_TOP_K
        self._error = ""
        self._service = graph_service
        if self._service is None:
            try:
                self._service = DrugGraphRetriever(
                    uri=uri or config.NEO4J_URI,
                    username=user or config.NEO4J_USER,
                    password=(
                        password if password is not None
                        else config.NEO4J_PASSWORD
                    ),
                    database=database or config.NEO4J_DATABASE,
                )
            except Exception as exc:
                self._error = f"知识图谱服务初始化失败: {exc}"
                logger.warning("%s", self._error)

    def is_available(self) -> bool:
        return self._service is not None

    def close(self) -> None:
        if self._service is not None:
            self._service.close()
            self._service = None

    def retrieve(
        self, query: str, top_k: int | None = None, **kwargs
    ) -> list[RetrievalResult]:
        question = (query or "").strip()
        if not question or self._service is None:
            return []
        limit = top_k or self.top_k
        try:
            state = self._service.retrieve(
                question, limit=limit,
                retrieval_id=str(kwargs.get("retrieval_id", "")),
                plan=kwargs.get("graph_plan"),
            )
            records = state.get("records", []) if isinstance(state, dict) else []
        except Exception as exc:
            self._error = f"知识图谱检索失败: {exc}"
            logger.warning("%s", self._error)
            return []

        results: list[RetrievalResult] = []
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            content = normalize_graph_record(record)
            name = str(content.get("name", "")).strip()
            record_id = str(content.get("id", "")).strip()
            key = record_id or f"{name}:{content.get('applicable_scope', '')}"
            if not name or key in seen:
                continue
            seen.add(key)
            rank = len(results) + 1
            results.append(RetrievalResult(
                content=content, score=1.0 / rank, source=self.name, rank=rank,
                reason="knowledge_graph disease-treatment relation", raw=record,
            ))
            if len(results) >= limit:
                break
        return results
