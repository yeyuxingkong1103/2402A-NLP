"""Redis 热缓存与 MySQL 持久层之间的会话生命周期协调器。"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _decode(value: Any) -> Any:
    return value.decode("utf-8") if isinstance(value, bytes) else value


class SessionManager:
    """创建、读取、更新和结束会话，并维护用户的活跃会话索引。"""

    def __init__(self, redis_client: Any, mysql_client: Any) -> None:
        self.redis = redis_client
        self.mysql = mysql_client

    @staticmethod
    def _meta_key(session_id: str) -> str:
        return f"session:{session_id}:meta"

    def _cache_meta(self, session_id: str, meta: dict[str, Any]) -> None:
        key = self._meta_key(session_id)
        self.redis.hset(
            key,
            mapping={name: str(value) for name, value in meta.items() if value is not None},
        )
        self.redis.expire(key, 3600)

    def create_session(
        self,
        user_id: int,
        tenant_id: int,
        role_id: int,
        channel: str = "web",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """同时创建数据库记录、Redis 元数据和用户活跃索引。"""
        try:
            session_id = f"sess_{uuid.uuid4().hex[:16]}"
            now = datetime.now(timezone.utc)
            self.mysql.execute(
                """INSERT INTO sessions (
                    session_id, tenant_id, user_id, role_id,
                    status, message_count, started_at, last_active_at, channel
                ) VALUES (%s, %s, %s, %s, 'active', 0, %s, %s, %s)""",
                (session_id, tenant_id, user_id, role_id, now, now, channel),
            )
            meta = {
                "session_id": session_id,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "role_id": role_id,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "message_count": 0,
                "status": "active",
                "channel": channel,
                "metadata": metadata or {},
            }
            self._cache_meta(session_id, meta)
            active_key = f"user:{user_id}:active_sessions"
            self.redis.zadd(active_key, {session_id: now.timestamp()})
            self.redis.expire(active_key, 86400)
            logger.info("创建会话成功: %s, user=%s, role=%s", session_id, user_id, role_id)
            return session_id
        except Exception as exc:
            logger.error("创建会话失败: %s", exc, exc_info=True)
            raise

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """优先读完整缓存；残缺缓存必须回落 MySQL 并重建。"""
        try:
            raw = self.redis.hgetall(self._meta_key(session_id))
            cached = {
                str(_decode(key)): _decode(value) for key, value in (raw or {}).items()
            }
            # RedisMemory 更新活跃时间时可能在 TTL 过期后创建仅含两个字段的 hash。
            # session_id 只有完整写入路径才有，因此作为缓存完整性标志。
            if cached.get("session_id"):
                return cached
            if cached:
                logger.info(
                    "会话 %s 的 Redis 元数据不完整（字段: %s），回落 MySQL 重建",
                    session_id,
                    ",".join(sorted(cached)),
                )

            row = self.mysql.fetchone(
                """SELECT session_id, tenant_id, user_id, role_id,
                          status, message_count, started_at, last_active_at,
                          ended_at, channel, metadata
                   FROM sessions WHERE session_id = %s""",
                (session_id,),
            )
            if not row:
                logger.warning("会话不存在: %s", session_id)
                return None
            meta = {
                "session_id": row["session_id"],
                "tenant_id": row["tenant_id"],
                "user_id": row["user_id"],
                "role_id": row["role_id"],
                "status": row["status"],
                "message_count": row["message_count"],
                "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                "last_active_at": row["last_active_at"].isoformat() if row["last_active_at"] else None,
                "ended_at": row["ended_at"].isoformat() if row["ended_at"] else None,
                "channel": row["channel"],
            }
            self._cache_meta(session_id, meta)
            return meta
        except Exception as exc:
            logger.error("获取会话失败: %s", exc, exc_info=True)
            return None

    def update_session(self, session_id: str, updates: dict[str, Any]) -> bool:
        """更新受信调用方给出的字段；路由层负责限制可写字段集合。"""
        if not updates:
            return True
        allowed_fields = {"role_id", "message_count", "status", "channel"}
        if not set(updates).issubset(allowed_fields):
            raise ValueError("unsupported session update field")
        try:
            self.redis.hset(
                self._meta_key(session_id),
                mapping={name: str(value) for name, value in updates.items()},
            )
            set_clause = ", ".join(f"{name} = %s" for name in updates)
            self.mysql.execute(
                f"UPDATE sessions SET {set_clause}, last_active_at = NOW() "
                "WHERE session_id = %s",
                (*updates.values(), session_id),
            )
            return True
        except Exception as exc:
            logger.error("更新会话失败: %s", exc, exc_info=True)
            return False

    def end_session(self, session_id: str) -> None:
        """将会话标记为已归档；长期记忆归档由上层编排。"""
        try:
            now = datetime.now(timezone.utc)
            self.mysql.execute(
                "UPDATE sessions SET status = 'archived', ended_at = %s "
                "WHERE session_id = %s",
                (now, session_id),
            )
            key = self._meta_key(session_id)
            self.redis.hset(key, "status", "archived")
            self.redis.hset(key, "ended_at", now.isoformat())
            logger.info("会话已结束: %s", session_id)
        except Exception as exc:
            logger.error("结束会话失败: %s", exc, exc_info=True)

    def get_user_active_sessions(self, user_id: int, limit: int = 10) -> list[dict[str, Any]]:
        try:
            session_ids = self.redis.zrevrange(
                f"user:{user_id}:active_sessions", 0, limit - 1
            )
            sessions = []
            for value in session_ids or []:
                session = self.get_session(str(_decode(value)))
                if session and session.get("status") == "active":
                    sessions.append(session)
            return sessions
        except Exception as exc:
            logger.error("获取活跃会话失败: %s", exc, exc_info=True)
            return []

    def increment_message_count(self, session_id: str) -> None:
        try:
            self.redis.hincrby(self._meta_key(session_id), "message_count", 1)
            self.mysql.execute(
                "UPDATE sessions SET message_count = message_count + 1, "
                "last_active_at = NOW() WHERE session_id = %s",
                (session_id,),
            )
        except Exception as exc:
            logger.warning("更新消息计数失败: %s", exc)

    def check_session_exists(self, session_id: str) -> bool:
        try:
            if self.redis.exists(self._meta_key(session_id)):
                return True
            return self.mysql.fetchone(
                "SELECT 1 FROM sessions WHERE session_id = %s", (session_id,)
            ) is not None
        except Exception as exc:
            logger.error("检查会话存在性失败: %s", exc)
            return False

    def delete_session(self, session_id: str) -> None:
        """物理删除仅供明确的管理场景使用，正常结束应调用 end_session。"""
        try:
            self.redis.delete(
                self._meta_key(session_id),
                f"session:{session_id}:messages",
                f"session:{session_id}:system_prompt_cache",
            )
            self.mysql.execute(
                "DELETE FROM sessions WHERE session_id = %s", (session_id,)
            )
            logger.warning("会话已删除: %s", session_id)
        except Exception as exc:
            logger.error("删除会话失败: %s", exc, exc_info=True)
