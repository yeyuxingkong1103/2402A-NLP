"""Redis 短期记忆：会话消息、元数据与活跃会话索引。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _decode(value: Any) -> Any:
    """Redis 未开启自动解码，统一兼容 bytes 与测试中的 str。"""
    return value.decode("utf-8") if isinstance(value, bytes) else value


class RedisMemory:
    """在 Redis 中维护有界、带 TTL 的短期会话状态。"""

    def __init__(self, redis_client: Any) -> None:
        self.client = redis_client
        self.default_ttl = 3600
        self.max_messages_per_session = 100

    @staticmethod
    def _messages_key(session_id: str) -> str:
        return f"session:{session_id}:messages"

    @staticmethod
    def _meta_key(session_id: str) -> str:
        return f"session:{session_id}:meta"

    def add_message(
        self, session_id: str, message: dict[str, Any], ttl: int | None = None
    ) -> None:
        """追加一条消息，并刷新消息列表与会话元数据的活跃时间。"""
        try:
            key = self._messages_key(session_id)
            if "timestamp" not in message:
                message["timestamp"] = datetime.now(timezone.utc).isoformat()
            self.client.rpush(key, json.dumps(message, ensure_ascii=False))
            self.client.ltrim(key, -self.max_messages_per_session, -1)
            self.client.expire(key, ttl or self.default_ttl)
            self._update_session_meta(session_id, increment_count=True)
            logger.debug("添加消息到会话 %s, role=%s", session_id, message.get("role"))
        except Exception as exc:
            logger.error("添加消息失败: %s", exc, exc_info=True)
            raise

    def _read_messages(self, session_id: str, start: int) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for raw_message in self.client.lrange(self._messages_key(session_id), start, -1):
            try:
                messages.append(json.loads(_decode(raw_message)))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("消息反序列化失败: %s", exc)
        return messages

    def get_messages(self, session_id: str, max_turns: int = 5) -> list[dict[str, Any]]:
        """读取最近 N 轮消息；一轮按一条 user 与一条 assistant 计算。"""
        try:
            messages = self._read_messages(session_id, -(max_turns * 2))
            logger.debug("从会话 %s 读取 %s 条消息", session_id, len(messages))
            return messages
        except Exception as exc:
            logger.error("读取消息失败: %s", exc, exc_info=True)
            return []

    def get_all_messages(self, session_id: str) -> list[dict[str, Any]]:
        """读取当前 TTL 窗口内的全部消息。"""
        try:
            return self._read_messages(session_id, 0)
        except Exception as exc:
            logger.error("读取所有消息失败: %s", exc, exc_info=True)
            return []

    def get_message_count(self, session_id: str) -> int:
        try:
            return int(self.client.llen(self._messages_key(session_id)))
        except Exception as exc:
            logger.error("获取消息数量失败: %s", exc)
            return 0

    def delete_session(self, session_id: str) -> None:
        """删除 Redis 中与会话相关的全部临时数据。"""
        try:
            self.client.delete(
                self._messages_key(session_id),
                self._meta_key(session_id),
                f"session:{session_id}:system_prompt_cache",
            )
            logger.info("已删除会话 %s 的 Redis 数据", session_id)
        except Exception as exc:
            logger.error("删除会话失败: %s", exc, exc_info=True)

    def set_session_meta(
        self, session_id: str, meta: dict[str, Any], ttl: int | None = None
    ) -> None:
        try:
            key = self._meta_key(session_id)
            self.client.hset(key, mapping={key: str(value) for key, value in meta.items()})
            self.client.expire(key, ttl or self.default_ttl)
        except Exception as exc:
            logger.error("设置会话元数据失败: %s", exc, exc_info=True)

    def get_session_meta(self, session_id: str, field: str | None = None) -> Any:
        """读取元数据；指定 field 时返回单值，否则返回字符串字典。"""
        try:
            key = self._meta_key(session_id)
            if field:
                value = self.client.hget(key, field)
                return _decode(value) if value is not None else None
            return {
                str(_decode(item_key)): _decode(item_value)
                for item_key, item_value in self.client.hgetall(key).items()
            }
        except Exception as exc:
            logger.error("获取会话元数据失败: %s", exc, exc_info=True)
            return None

    def _update_session_meta(self, session_id: str, increment_count: bool = False) -> None:
        """更新活跃状态；该写入可能创建残缺 hash，读取方必须检查完整性。"""
        try:
            key = self._meta_key(session_id)
            self.client.hset(key, "updated_at", datetime.now(timezone.utc).isoformat())
            if increment_count:
                self.client.hincrby(key, "message_count", 1)
            self.client.expire(key, self.default_ttl)
        except Exception as exc:
            logger.warning("更新会话元数据失败: %s", exc)

    def add_to_user_active_sessions(
        self, user_id: int, session_id: str, score: float | None = None
    ) -> None:
        try:
            key = f"user:{user_id}:active_sessions"
            ranking = score if score is not None else datetime.now(timezone.utc).timestamp()
            self.client.zadd(key, {session_id: ranking})
            self.client.expire(key, 86400)
        except Exception as exc:
            logger.error("添加活跃会话失败: %s", exc)

    def get_user_active_sessions(self, user_id: int, limit: int = 10) -> list[str]:
        try:
            values = self.client.zrevrange(f"user:{user_id}:active_sessions", 0, limit - 1)
            return [str(_decode(value)) for value in values]
        except Exception as exc:
            logger.error("获取活跃会话失败: %s", exc)
            return []

    def compress_messages(
        self,
        session_id: str,
        summary_message: dict[str, Any],
        keep_last_turns: int = 2,
    ) -> int:
        """以一条摘要替换旧消息，并保留最后若干完整对话轮次。"""
        try:
            all_messages = self.get_all_messages(session_id)
            keep_count = keep_last_turns * 2
            if len(all_messages) <= keep_count:
                return len(all_messages)
            compressed = [summary_message, *all_messages[-keep_count:]]
            key = self._messages_key(session_id)
            self.client.delete(key)
            for message in compressed:
                self.client.rpush(key, json.dumps(message, ensure_ascii=False))
            self.client.expire(key, self.default_ttl)
            logger.info("会话 %s 压缩完成: %s -> %s 条消息", session_id, len(all_messages), len(compressed))
            return len(compressed)
        except Exception as exc:
            logger.error("压缩消息失败: %s", exc, exc_info=True)
            return 0
