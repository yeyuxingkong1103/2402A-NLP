from __future__ import annotations

import math
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from ..models import ModelGateway
from ..storage.mysql import LongTermMemoryStore, MemoryConflictStore

PROFILE_TYPES = ("user_profile", "conversation_summary", "fact", "preference", "skill", "project", "decision", "event")


@dataclass(frozen=True)
class MemoryCandidate:
    memory_type: str
    content: str
    importance: float = 0.5
    confidence: float = 0.7
    scope: str = "user"
    source_conversation_id: str | None = None
    source_message_id: int | None = None


class MemoryExtractor:
    def __init__(self, model: ModelGateway | None = None):
        self.model = model

    def extract(self, messages: list[dict]) -> list[MemoryCandidate]:
        candidates = []
        for row in messages:
            content = " ".join(str(row.get("content") or "").split())
            if len(content) < 8 or row.get("role") != "user":
                continue
            if any(marker in content for marker in ("我叫", "我是", "我希望", "我要求", "我决定", "我的职业")):
                candidates.append(MemoryCandidate("user_profile", f"用户画像：{content[:1200]}", source_conversation_id=str(row.get("conversation_id") or "") or None, source_message_id=int(row.get("id") or 0) or None))
        return candidates[:20]


def _recent_score(row: dict) -> float:
    value = row.get("last_used_at") or row.get("updated_at") or row.get("created_at")
    try:
        point = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        if point.tzinfo is None:
            point = point.replace(tzinfo=timezone.utc)
        return 1 / (1 + max(0, (datetime.now(timezone.utc) - point.astimezone(timezone.utc)).total_seconds() / 86400))
    except (TypeError, ValueError):
        return 0.0


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    size = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / size if size else 0.0


class LongMemoryRetriever:
    def __init__(self, model: ModelGateway | None, memories: LongTermMemoryStore, candidate_limit: int = 200):
        self.model, self.memories, self.candidate_limit = model, memories, max(20, int(candidate_limit or 200))

    def retrieve(self, user_id: str, query: str, top_k: int = 5, scope: str = "user", conversation_id: str | None = None, use_vector: bool = True) -> list[dict]:
        rows = self.memories.list_for_retrieval(user_id, scope=scope, limit=self.candidate_limit, conversation_id=conversation_id, memory_types=PROFILE_TYPES)
        if not rows or not str(query or '').strip():
            return []
        vector = None
        if use_vector and self.model:
            try:
                vector = self.model.embed([str(query).strip()])[0]
            except Exception:
                vector = None
        query_words = set(str(query).lower().split())
        scored = []
        for row in rows:
            content = str(row.get("content") or "")
            words = set(content.lower().split())
            text_score = len(query_words & words) / max(1, len(query_words | words))
            similarity = _cosine(vector, row.get("embedding") or []) if vector else 0.0
            score = similarity * 0.72 + float(row.get("importance") or 0) * 0.2 + _recent_score(row) * 0.08 if vector else text_score * 0.7 + float(row.get("importance") or 0) * 0.2 + _recent_score(row) * 0.1
            if score > 0:
                item = dict(row)
                item["memory_score"] = score
                scored.append(item)
        scored.sort(key=lambda item: item["memory_score"], reverse=True)
        selected = scored[:max(1, int(top_k or 5))]
        self.memories.mark_used(user_id, [int(item["id"]) for item in selected if item.get("id")])
        return selected

    def retrieve_by_text(self, user_id: str, query: str, top_k: int = 5, scope: str = "user") -> list[dict]:
        return self.retrieve(user_id, query, top_k, scope, use_vector=False)


class LongMemoryManager:
    def __init__(self, memories: LongTermMemoryStore, conflicts: MemoryConflictStore | None = None):
        self.memories, self.conflicts = memories, conflicts

    def list_memories(self, user_id: str, limit: int = 100, offset: int = 0, memory_type: Optional[str] = None) -> list[dict]:
        return self.memories.list_by_type(user_id, memory_type, limit, offset) if memory_type else self.memories.list_active(user_id, limit, offset)

    def list_conflicts(self, user_id: str, limit: int = 50) -> list[dict]:
        return self.conflicts.list_pending(user_id, limit=limit) if self.conflicts else []

    def add_memory(self, user_id: str, memory_type: str, content: str, importance: float = 0.5, confidence: float = 0.7, scope: str = "user", source_conversation_id: str | None = None, source_message_id: int | None = None, embedding: list[float] | None = None) -> int | None:
        value = " ".join(str(content or "").split())[:1200]
        if not user_id or not value:
            return None
        return self.memories.add({"user_id": str(user_id), "scope": scope, "memory_type": memory_type, "content": value, "embedding": embedding, "importance": importance, "confidence": confidence, "source_conversation_id": source_conversation_id, "source_message_id": source_message_id})

    def resolve_conflict(self, user_id: str, conflict_id: int, action: str) -> dict:
        return {"resolved": False, "conflict_id": int(conflict_id), "reason": "记忆采用用户隔离和来源校验，不自动合并冲突"}

    def explain(self, user_id: str, memory_id: int) -> dict:
        row = self.memories.get(user_id, memory_id)
        return {"found": bool(row), **(row or {})}

    def delete_memory(self, user_id: str, memory_id: int) -> bool:
        return self.memories.delete(user_id, memory_id)

    def delete_all_memories(self, user_id: str) -> int:
        return self.memories.delete_all(user_id)
