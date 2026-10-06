from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

from ..models import ModelGateway
from ..storage.redis import RedisStore

RECENT_MESSAGE_KEEP = 10
DEFAULT = {"user_id": "", "conversation_id": "", "last_history_id": 0, "messages": [], "summary": "", "current_topic": "", "current_goal": "", "entities": [], "decisions": [], "unresolved_questions": [], "updated_at": ""}


def _int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


class ShortMemoryStore:
    def __init__(self, redis: RedisStore, ttl: int = 60 * 60 * 24 * 30):
        self.redis, self.ttl = redis, max(60, int(ttl or 60 * 60 * 24 * 30))

    @staticmethod
    def key(user_id: str, conversation_id: str) -> str:
        return f"short_memory:{str(user_id).strip()}:{str(conversation_id).strip()}"

    @staticmethod
    def lock_key(user_id: str, conversation_id: str) -> str:
        return f"short_memory_lock:{str(user_id).strip()}:{str(conversation_id).strip()}"

    @staticmethod
    def normalize(value: dict | None) -> dict:
        result = deepcopy(DEFAULT)
        if isinstance(value, dict):
            result.update(value)
        result["messages"] = [{"role": str(item.get("role") or ""), "content": str(item.get("content") or ""), "history_id": _int(item.get("history_id"))} for item in result.get("messages") or [] if isinstance(item, dict) and str(item.get("content") or "").strip()]
        for key in ("entities", "decisions", "unresolved_questions"):
            result[key] = list(dict.fromkeys(str(item).strip() for item in result.get(key) or [] if str(item).strip()))
        for key in ("user_id", "conversation_id", "summary", "current_topic", "current_goal", "updated_at"):
            result[key] = str(result.get(key) or "")
        result["last_history_id"] = _int(result.get("last_history_id"))
        return result

    def load(self, user_id: str, conversation_id: str) -> dict:
        try:
            return self.normalize(self.redis.get_json(self.key(user_id, conversation_id)))
        except (TypeError, json.JSONDecodeError):
            return self.normalize(None)

    def save(self, user_id: str, conversation_id: str, memory: dict) -> dict:
        value = self.normalize(memory)
        value.update({"user_id": str(user_id), "conversation_id": str(conversation_id), "updated_at": datetime.now(timezone.utc).isoformat()})
        self.redis.set_json_value(self.key(user_id, conversation_id), value, self.ttl)
        return value

    def append_turn(self, user_id: str, conversation_id: str, question: str, answer: str, ids: tuple[int, int]) -> dict:
        memory = self.load(user_id, conversation_id)
        memory["messages"] += [{"role": "user", "content": str(question or ""), "history_id": _int(ids[0])}, {"role": "assistant", "content": str(answer or ""), "history_id": _int(ids[1])}]
        memory["messages"] = memory["messages"][-RECENT_MESSAGE_KEEP:]
        memory["last_history_id"] = _int(ids[1] or ids[0])
        return self.save(user_id, conversation_id, memory)

    def delete(self, user_id: str, conversation_id: str) -> None:
        self.redis.delete(self.key(user_id, conversation_id))

    def with_lock(self, user_id: str, conversation_id: str, ttl: int = 30):
        return self.redis.lock(self.lock_key(user_id, conversation_id), ttl)


ShortMemoryRedisStore = ShortMemoryStore


class ContextCompressor:
    def __init__(self, model: ModelGateway | None = None, token_limit: int = 8000):
        self.model, self.token_limit = model, max(1000, int(token_limit or 8000))

    def should_compress(self, memory: dict) -> bool:
        return len(json.dumps(memory.get("messages") or [], ensure_ascii=False)) // 2 > self.token_limit

    def maybe_compress(self, memory: dict) -> dict:
        messages = list(memory.get("messages") or [])
        if not self.should_compress(memory) or len(messages) <= RECENT_MESSAGE_KEEP:
            return memory
        old, recent = messages[:-RECENT_MESSAGE_KEEP], messages[-RECENT_MESSAGE_KEEP:]
        summary = "\n".join(f"{item.get('role')}: {str(item.get('content') or '').strip()}" for item in old)[-5000:]
        memory["summary"] = "\n".join(item for item in (str(memory.get("summary") or "").strip(), summary) if item)[-6000:]
        memory["messages"] = recent
        return memory


__all__ = ["ShortMemoryStore", "ShortMemoryRedisStore", "ContextCompressor"]
