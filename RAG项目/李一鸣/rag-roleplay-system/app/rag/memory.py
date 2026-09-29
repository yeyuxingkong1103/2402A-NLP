import json
import logging
from collections import defaultdict, deque
from threading import RLock

from app.core.config import Settings

logger = logging.getLogger(__name__)


class MemoryService:
    """Redis short-term memory with an in-process fallback."""

    def __init__(self, settings: Settings):
        self.enabled = settings.redis_enabled
        self.max_messages = settings.max_memory_messages
        self._redis = None
        self._lock = RLock()
        self._local: dict[str, deque[dict]] = defaultdict(
            lambda: deque(maxlen=self.max_messages)
        )
        if self.enabled:
            try:
                import redis

                self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("redis short-term memory connected")
            except Exception:
                logger.exception("redis unavailable; using in-process memory")
                self._redis = None

    def get_history(self, user_id: str, role_id: str, conversation_id: str) -> list[dict]:
        key = self._key(user_id, role_id, conversation_id)
        if self._redis is not None:
            try:
                values = self._redis.lrange(key, 0, self.max_messages - 1)
                return [json.loads(value) for value in values]
            except Exception:
                logger.exception("redis history read failed; using local memory")
        with self._lock:
            return list(self._local[key])

    def append_turn(
        self,
        user_id: str,
        role_id: str,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        key = self._key(user_id, role_id, conversation_id)
        items = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        if self._redis is not None:
            try:
                self._redis.rpush(key, *[json.dumps(item, ensure_ascii=False) for item in items])
                self._redis.ltrim(key, -self.max_messages, -1)
                return
            except Exception:
                logger.exception("redis history write failed; using local memory")
        with self._lock:
            self._local[key].extend(items)

    @staticmethod
    def _key(user_id: str, role_id: str, conversation_id: str) -> str:
        return f"rag:memory:{user_id}:{role_id}:{conversation_id}"
