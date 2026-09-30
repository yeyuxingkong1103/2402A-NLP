"""角色配置缓存、角色切换与权限判断。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RoleConfig:
    """一个租户内可用于对话编排的完整角色配置。"""

    id: int
    tenant_id: int
    role_name: str
    role_key: str
    persona_name: str
    persona_description: str
    system_prompt: str
    greeting_message: str
    temperature: float
    max_tokens: int
    top_k: int
    enable_rerank: bool
    enable_memory: bool
    max_memory_turns: int
    allowed_intents: list[str]
    restricted_topics: list[str]
    escalation_rules: dict[str, str]


class RoleManager:
    """从 MySQL 加载角色并用 Redis 缓存，同时记录会话角色切换。"""

    def __init__(self, mysql_client: Any, redis_client: Any) -> None:
        self.mysql = mysql_client
        self.redis = redis_client
        self.cache_ttl = 86400

    @staticmethod
    def _cache_key(role_id: int, tenant_id: int) -> str:
        return f"role_config:{tenant_id}:{role_id}"

    @staticmethod
    def _config_from_row(row: dict[str, Any]) -> RoleConfig:
        return RoleConfig(
            id=row["id"],
            tenant_id=row["tenant_id"],
            role_name=row["role_name"],
            role_key=row["role_key"],
            persona_name=row["persona_name"],
            persona_description=row["persona_description"],
            system_prompt=row["system_prompt"],
            greeting_message=row["greeting_message"],
            temperature=float(row["temperature"]),
            max_tokens=row["max_tokens"],
            top_k=row["top_k"],
            enable_rerank=bool(row["enable_rerank"]),
            enable_memory=bool(row["enable_memory"]),
            max_memory_turns=row["max_memory_turns"],
            allowed_intents=json.loads(row["allowed_intents"] or "[]"),
            restricted_topics=json.loads(row["restricted_topics"] or "[]"),
            escalation_rules=json.loads(row["escalation_rules"] or "{}"),
        )

    def get_role_config(self, role_id: int, tenant_id: int) -> RoleConfig | None:
        """优先读取租户隔离的缓存，未命中时回源数据库。"""
        try:
            cache_key = self._cache_key(role_id, tenant_id)
            cached = self.redis.get(cache_key)
            if cached:
                return RoleConfig(**json.loads(cached))
            row = self.mysql.fetchone(
                """SELECT id, tenant_id, role_name, role_key,
                          persona_name, persona_description,
                          system_prompt, greeting_message,
                          temperature, max_tokens, top_k,
                          enable_rerank, enable_memory, max_memory_turns,
                          allowed_intents, restricted_topics, escalation_rules
                   FROM roles
                   WHERE id = %s AND tenant_id = %s AND status = 'active'""",
                (role_id, tenant_id),
            )
            if not row:
                logger.warning("角色不存在: role_id=%s, tenant_id=%s", role_id, tenant_id)
                return None
            config = self._config_from_row(row)
            self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(asdict(config), ensure_ascii=False),
            )
            logger.debug("加载角色配置: %s (%s)", config.persona_name, config.role_key)
            return config
        except Exception as exc:
            logger.error("获取角色配置失败: %s", exc, exc_info=True)
            return None

    def get_role_by_key(self, role_key: str, tenant_id: int) -> RoleConfig | None:
        try:
            row = self.mysql.fetchone(
                "SELECT id FROM roles WHERE role_key = %s AND tenant_id = %s "
                "AND status = 'active'",
                (role_key, tenant_id),
            )
            return self.get_role_config(row["id"], tenant_id) if row else None
        except Exception as exc:
            logger.error("通过 role_key 获取角色失败: %s", exc)
            return None

    def switch_role(
        self,
        session_id: str,
        user_id: int,
        tenant_id: int,
        new_role_id: int,
        reason: str | None = None,
        switch_type: str = "manual",
    ) -> bool:
        """切换角色并保留审计日志；目标角色必须属于当前租户。"""
        try:
            meta_key = f"session:{session_id}:meta"
            old_role_id = self.redis.hget(meta_key, "role_id")
            if isinstance(old_role_id, bytes):
                old_role_id = int(old_role_id.decode("utf-8"))
            new_role = self.get_role_config(new_role_id, tenant_id)
            if not new_role:
                logger.error("新角色不存在: role_id=%s", new_role_id)
                return False
            self.redis.hset(meta_key, "role_id", new_role_id)
            self.mysql.execute(
                "UPDATE sessions SET role_id = %s, last_active_at = NOW() "
                "WHERE session_id = %s",
                (new_role_id, session_id),
            )
            self.mysql.execute(
                """INSERT INTO role_switch_logs (
                    session_id, user_id, from_role_id, to_role_id, reason, switch_type
                ) VALUES (%s, %s, %s, %s, %s, %s)""",
                (session_id, user_id, old_role_id, new_role_id, reason, switch_type),
            )
            self.redis.delete(f"session:{session_id}:system_prompt_cache")
            logger.info(
                "角色切换成功: session=%s, %s -> %s (%s)",
                session_id,
                old_role_id,
                new_role_id,
                new_role.persona_name,
            )
            return True
        except Exception as exc:
            logger.error("角色切换失败: %s", exc, exc_info=True)
            return False

    def list_available_roles(self, tenant_id: int) -> list[dict[str, Any]]:
        try:
            rows = self.mysql.fetchall(
                """SELECT id, role_name, role_key, persona_name, persona_description
                   FROM roles WHERE tenant_id = %s AND status = 'active' ORDER BY id""",
                (tenant_id,),
            )
            return [
                {
                    "id": row["id"],
                    "role_name": row["role_name"],
                    "role_key": row["role_key"],
                    "persona_name": row["persona_name"],
                    "description": row["persona_description"],
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("列出角色失败: %s", exc)
            return []

    @staticmethod
    def check_intent_allowed(role_config: RoleConfig, intent: str) -> bool:
        """未配置白名单表示允许全部意图。"""
        return not role_config.allowed_intents or intent in role_config.allowed_intents

    @staticmethod
    def check_topic_restricted(role_config: RoleConfig, query: str) -> bool:
        query_lower = query.lower()
        return any(topic.lower() in query_lower for topic in role_config.restricted_topics)

    @staticmethod
    def get_escalation_target(role_config: RoleConfig, situation: str) -> str | None:
        return role_config.escalation_rules.get(situation)

    def invalidate_cache(self, role_id: int, tenant_id: int) -> None:
        try:
            cache_key = self._cache_key(role_id, tenant_id)
            self.redis.delete(cache_key)
            logger.info("角色缓存已失效: %s", cache_key)
        except Exception as exc:
            logger.warning("使缓存失效失败: %s", exc)
