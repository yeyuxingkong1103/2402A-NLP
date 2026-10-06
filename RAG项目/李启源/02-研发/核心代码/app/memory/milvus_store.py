"""Long-term memory writes and tenant-filtered Milvus searches."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)
_FACT_TYPES = {"order", "issue", "feedback", "general"}


class EmbeddingQueryAdapter:
    """Adapt the ingestion embedding client to one-query memory calls."""

    def __init__(self, embedding_client: Any) -> None:
        self.client = embedding_client
        self.dimension = int(embedding_client.dimension)

    def embed_query(self, text: str) -> list[float]:
        vectors = self.client.embed_texts([text])
        if len(vectors) != 1 or len(vectors[0]) != self.dimension:
            raise ValueError("embedding service returned an invalid query vector")
        return [float(value) for value in vectors[0]]


class MilvusMemory:
    """Persist and retrieve preferences, summaries, and important facts."""

    def __init__(self, milvus_client: Any, embedding_service: Any) -> None:
        self.client = milvus_client
        self.embedding_service = embedding_service
        names = getattr(milvus_client, "collection_names", {})
        self.preferences_collection = names.get("preferences", "rag_v1_user_preferences")
        self.summaries_collection = names.get("summaries", "rag_v1_conversation_summaries")
        self.facts_collection = names.get("facts", "rag_v1_important_facts")

    def embed(self, text: str) -> list[float]:
        return self.embedding_service.embed_query(text)

    def _insert(self, collection_name: str, payload: dict[str, Any]) -> bool:
        try:
            collection = self.client.get_collection(collection_name)
            collection.insert([payload])
            collection.flush()
            return True
        except Exception as exc:
            logger.error("Milvus memory write failed: %s", exc, exc_info=True)
            return False

    def save_user_preference(self, user_id: int, tenant_id: int, preference_type: str, preference_text: str, confidence: float = 1.0) -> bool:
        try:
            now = int(time.time())
            payload = {
                "user_id": user_id, "tenant_id": tenant_id, "preference_type": preference_type,
                "preference_text": preference_text, "preference_vector": self.embed(preference_text),
                "confidence": confidence, "created_at": now, "updated_at": now,
            }
        except Exception as exc:
            logger.error("Preference embedding failed: %s", exc, exc_info=True)
            return False
        return self._insert(self.preferences_collection, payload)

    def save_conversation_summary(self, session_id: str, user_id: int, tenant_id: int, summary_text: str, turn_count: int, key_intents: list[str], session_start: int, session_end: int) -> bool:
        try:
            payload = {
                "session_id": session_id, "user_id": user_id, "tenant_id": tenant_id,
                "summary_text": summary_text, "summary_vector": self.embed(summary_text),
                "turn_count": turn_count, "key_intents": key_intents[:10], "created_at": int(time.time()),
                "session_start": session_start, "session_end": session_end,
            }
        except Exception as exc:
            logger.error("Summary embedding failed: %s", exc, exc_info=True)
            return False
        return self._insert(self.summaries_collection, payload)

    def save_important_fact(self, user_id: int, tenant_id: int, fact_type: str, fact_text: str, source_session_id: str, confidence: float = 1.0, expires_at: int = 0) -> bool:
        try:
            if fact_type not in _FACT_TYPES:
                raise ValueError("unsupported fact type")
            payload = {
                "user_id": user_id, "tenant_id": tenant_id, "fact_type": fact_type,
                "fact_text": fact_text, "fact_vector": self.embed(fact_text),
                "source_session_id": source_session_id, "confidence": confidence,
                "created_at": int(time.time()), "expires_at": expires_at,
            }
        except Exception as exc:
            logger.error("Fact embedding failed: %s", exc, exc_info=True)
            return False
        return self._insert(self.facts_collection, payload)

    def _search(self, collection_name: str, field: str, query_vector: list[float], expr: str, output_fields: list[str], top_k: int) -> list[Any]:
        try:
            results = self.client.get_collection(collection_name).search(
                data=[query_vector], anns_field=field,
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k, expr=expr, output_fields=output_fields,
            )
            return results[0] if results else []
        except Exception as exc:
            logger.error("Milvus memory search failed: %s", exc, exc_info=True)
            return []

    @staticmethod
    def _hit_value(hit: Any, key: str) -> Any:
        return hit.entity.get(key)

    def retrieve_user_preferences(self, user_id: int, tenant_id: int, query_vector: list[float], top_k: int = 2) -> list[dict[str, Any]]:
        hits = self._search(self.preferences_collection, "preference_vector", query_vector, f"user_id == {user_id} and tenant_id == {tenant_id}", ["preference_type", "preference_text", "confidence", "created_at"], top_k)
        return [{"type": self._hit_value(hit, "preference_type"), "text": self._hit_value(hit, "preference_text"), "confidence": self._hit_value(hit, "confidence"), "score": hit.score, "created_at": self._hit_value(hit, "created_at")} for hit in hits]

    def retrieve_conversation_summaries(self, user_id: int, tenant_id: int, query_vector: list[float], top_k: int = 3) -> list[dict[str, Any]]:
        hits = self._search(self.summaries_collection, "summary_vector", query_vector, f"user_id == {user_id} and tenant_id == {tenant_id}", ["session_id", "summary_text", "turn_count", "key_intents", "created_at", "session_start", "session_end"], top_k)
        return [{"session_id": self._hit_value(hit, "session_id"), "summary": self._hit_value(hit, "summary_text"), "turn_count": self._hit_value(hit, "turn_count"), "intents": self._hit_value(hit, "key_intents") or [], "score": hit.score, "session_start": self._hit_value(hit, "session_start"), "session_end": self._hit_value(hit, "session_end"), "created_at": self._hit_value(hit, "created_at")} for hit in hits]

    def retrieve_important_facts(self, user_id: int, tenant_id: int, query_vector: list[float], fact_type: str | None = None, top_k: int = 2) -> list[dict[str, Any]]:
        if fact_type is not None and fact_type not in _FACT_TYPES:
            raise ValueError("unsupported fact type")
        expr = f"user_id == {user_id} and tenant_id == {tenant_id}"
        if fact_type:
            expr += f" and fact_type == '{fact_type}'"
        expr += f" and (expires_at == 0 or expires_at > {int(time.time())})"
        hits = self._search(self.facts_collection, "fact_vector", query_vector, expr, ["fact_type", "fact_text", "source_session_id", "confidence", "created_at"], top_k)
        return [{"type": self._hit_value(hit, "fact_type"), "text": self._hit_value(hit, "fact_text"), "source_session": self._hit_value(hit, "source_session_id"), "confidence": self._hit_value(hit, "confidence"), "score": hit.score, "created_at": self._hit_value(hit, "created_at")} for hit in hits]

    def retrieve_long_term_memory(self, user_id: int, tenant_id: int, query: str, query_vector: list[float] | None = None) -> dict[str, list[dict[str, Any]]]:
        try:
            vector = query_vector or self.embed(query)
            return {
                "preferences": self.retrieve_user_preferences(user_id, tenant_id, vector, top_k=2),
                "summaries": self.retrieve_conversation_summaries(user_id, tenant_id, vector, top_k=3),
                "facts": self.retrieve_important_facts(user_id, tenant_id, vector, top_k=2),
            }
        except Exception as exc:
            logger.error("Long-term memory retrieval failed: %s", exc, exc_info=True)
            return {"preferences": [], "summaries": [], "facts": []}
