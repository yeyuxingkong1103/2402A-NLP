from __future__ import annotations

"""Redis 会话记忆存储，保存会话、最近消息和长期记忆。"""

import json
from datetime import datetime, timezone
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from ..config.config import MemoryConfig


class MemoryStore:
    """使用 Redis 保存会话元数据、最近消息和长期记忆。"""
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.client = Redis.from_url(config.redis_url, decode_responses=True)

    def is_available(self) -> bool:
        try:
            return bool(self.client.ping())
        except RedisError:
            return False

    def _key(self, kind: str, session_id: str) -> str:
        """按统一命名规则生成 Redis Key。"""
        return f"{self.config.namespace}:{kind}:{session_id}"

    def _decode(self, values: list[str]) -> list[dict[str, Any]]:
        return [json.loads(value) for value in values]

    def list_sessions(self) -> list[dict[str, Any]]:
        pattern = f"{self.config.namespace}:session:*"
        sessions: list[dict[str, Any]] = []
        try:
            for key in self.client.scan_iter(match=pattern):
                value = self.client.hgetall(key)
                if value:
                    sessions.append(value)
        except RedisError:
            return []
        return sorted(sessions, key=lambda item: item.get("updated_at", ""), reverse=True)

    def create_session(self, session_id: str, title: str = "新对话") -> dict[str, str]:
        """创建会话 Hash，并设置会话过期时间。"""
        now = datetime.now(timezone.utc).isoformat()
        session = {"id": session_id, "title": title, "created_at": now, "updated_at": now}
        try:
            key = self._key("session", session_id)
            self.client.hset(key, mapping=session)
            self.client.expire(key, self.config.session_ttl_seconds)
        except RedisError:
            pass
        return session

    def get_session_messages(self, session_id: str) -> list[dict[str, Any]]:
        try:
            values = self.client.lrange(self._key("messages", session_id), 0, self.config.recent_message_limit - 1)
            return self._decode(values)
        except RedisError:
            return []

    def append_message(self, session_id: str, message: dict[str, Any]) -> None:
        """将消息放到列表头部，并裁剪到最近消息数量上限。"""
        key = self._key("messages", session_id)
        try:
            pipe = self.client.pipeline()
            pipe.lpush(key, json.dumps(message, ensure_ascii=False))
            pipe.ltrim(key, 0, self.config.recent_message_limit - 1)
            pipe.expire(key, self.config.session_ttl_seconds)
            pipe.execute()
        except RedisError:
            return

    def save_long_term_memory(self, session_id: str, content: str, memory_type: str = "conversation") -> None:
        """保存一条截断后的长期记忆。"""
        memory = {
            "session_id": session_id,
            "content": content[:1200],
            "memory_type": memory_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            key = self._key("long_term", session_id)
            pipe = self.client.pipeline()
            pipe.lpush(key, json.dumps(memory, ensure_ascii=False))
            pipe.ltrim(key, 0, min(self.config.long_term_limit, 20) - 1)
            pipe.execute()
        except RedisError:
            return

    def get_long_term_memories(self, session_id: str) -> list[dict[str, Any]]:
        try:
            values = self.client.lrange(self._key("long_term", session_id), 0, self.config.long_term_limit - 1)
            return self._decode(values)
        except RedisError:
            return []

    def delete_session(self, session_id: str) -> None:
        try:
            self.client.delete(
                self._key("session", session_id),
                self._key("messages", session_id),
                self._key("long_term", session_id),
            )
        except RedisError:
            return

    def clear_messages(self, session_id: str) -> None:
        try:
            self.client.delete(self._key("messages", session_id))
            self.touch_session(session_id, title="新对话")
        except RedisError:
            return

    def touch_session(self, session_id: str, title: str | None = None) -> None:
        try:
            key = self._key("session", session_id)
            mapping: dict[str, str] = {"updated_at": datetime.now(timezone.utc).isoformat()}
            if title:
                mapping["title"] = title
            self.client.hset(key, mapping=mapping)
            self.client.expire(key, self.config.session_ttl_seconds)
        except RedisError:
            return
