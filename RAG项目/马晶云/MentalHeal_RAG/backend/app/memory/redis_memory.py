from collections.abc import Mapping
import hashlib
import json
import logging
from functools import lru_cache
from typing import Any

import redis

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class RedisChatMemory:
    def __init__(self, settings: Settings) -> None:
        self.max_rounds = settings.chat_history_max_rounds
        self.ttl_seconds = settings.chat_history_ttl_seconds
        self.cache_ttl_seconds = settings.rag_cache_ttl_seconds
        self.status_ttl_seconds = settings.document_status_ttl_seconds
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )

    def load(
        self,
        session_id: str,
        user_id: str | None = None,
        role_id: str | None = None,
    ) -> list[dict[str, str]]:
        try:
            raw = self.client.get(self._key(session_id, user_id, role_id))
            if not raw and user_id is not None and role_id is not None:
                raw = self.client.get(self._legacy_key(session_id))
            if not raw:
                return []
            history = json.loads(raw)
            if not isinstance(history, list):
                return []
            return history[-self.max_rounds * 2 :]
        except (redis.RedisError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Redis history read failed: %s", type(exc).__name__)
            return []

    def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        user_id: str | None = None,
        role_id: str | None = None,
    ) -> None:
        try:
            history = self.load(session_id, user_id, role_id)
            history.extend(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ]
            )
            history = history[-self.max_rounds * 2 :]
            key = self._key(session_id, user_id, role_id)
            self.client.setex(key, self.ttl_seconds, json.dumps(history, ensure_ascii=False))
        except redis.RedisError as exc:
            logger.warning("Redis history write failed: %s", type(exc).__name__)

    def get_query_cache(
        self,
        query: str,
        user_id: str,
        role_id: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]] | None:
        try:
            raw = self.client.get(self._cache_key(query, user_id, role_id, top_k))
            value = json.loads(raw) if raw else None
            return value if isinstance(value, list) else None
        except (redis.RedisError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Redis retrieval cache read failed: %s", type(exc).__name__)
            return None

    def set_query_cache(
        self,
        query: str,
        user_id: str,
        role_id: str,
        results: list[Mapping[str, Any]],
        top_k: int | None = None,
    ) -> None:
        try:
            self.client.setex(
                self._cache_key(query, user_id, role_id, top_k),
                self.cache_ttl_seconds,
                json.dumps(list(results), ensure_ascii=False),
            )
        except (redis.RedisError, TypeError) as exc:
            logger.warning("Redis retrieval cache write failed: %s", type(exc).__name__)

    def set_document_status(self, document_id: str, status: str) -> None:
        try:
            self.client.setex(
                f"doc:parse:status:{document_id}",
                self.status_ttl_seconds,
                status,
            )
        except redis.RedisError as exc:
            logger.warning("Redis document status write failed: %s", type(exc).__name__)

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except redis.RedisError:
            return False

    @staticmethod
    def _key(session_id: str, user_id: str | None = None, role_id: str | None = None) -> str:
        if user_id and role_id:
            return f"chat:history:{user_id}:{role_id}:{session_id}"
        return RedisChatMemory._legacy_key(session_id)

    @staticmethod
    def _legacy_key(session_id: str) -> str:
        return f"mentalheal:chat:history:{session_id}"

    @staticmethod
    def _cache_key(
        query: str,
        user_id: str,
        role_id: str,
        top_k: int | None = None,
    ) -> str:
        digest = hashlib.sha256(
            f"{query.strip().casefold()}\n{top_k or 0}".encode("utf-8")
        ).hexdigest()
        return f"rag:cache:{user_id}:{role_id}:{digest}"


@lru_cache(maxsize=1)
def get_chat_memory() -> RedisChatMemory:
    return RedisChatMemory(get_settings())
