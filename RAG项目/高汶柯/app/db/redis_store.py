"""Redis 存储：短期记忆 + 缓存 + 热问题统计。

键设计：
- ``chat:{user_id}:{role_id}``  List   最近对话（短期记忆）
- ``session:{user_id}:{role_id}`` Hash 会话元数据
- ``sessions:{user_id}``          Set   用户拥有的角色会话
- ``hot:queries``                 ZSet  热问题排行
- ``cache:query:{q}``             String 检索/答案缓存
- ``roles``                       Set   角色 ID 索引
"""
from __future__ import annotations

import json
from typing import Any, Optional

import redis

from app.config import settings
from app.logging_conf import log


class RedisStore:
    def __init__(self) -> None:
        self._client: Optional[redis.Redis] = None

    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password or None,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            self._client.ping()
            log.info("Redis 连接成功 %s:%s db=%s", settings.redis_host, settings.redis_port, settings.redis_db)
        return self._client

    @property
    def available(self) -> bool:
        try:
            self.client()
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis 不可用: %s", exc)
            return False

    # ===== 短期记忆 =====
    @staticmethod
    def _chat_key(user_id: str, role_id: int) -> str:
        return f"chat:{user_id}:{role_id}"

    def push_message(self, user_id: str, role_id: int, role: str, content: str) -> None:
        try:
            key = self._chat_key(user_id, role_id)
            c = self.client()
            c.rpush(key, json.dumps({"role": role, "content": content}, ensure_ascii=False))
            c.ltrim(key, -settings.memory_max_turns * 2, -1)
            c.expire(key, settings.memory_ttl)
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis push_message 失败: %s", exc)

    def get_messages(self, user_id: str, role_id: int) -> list[dict]:
        try:
            items = self.client().lrange(self._chat_key(user_id, role_id), 0, -1)
            return [json.loads(i) for i in items]
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis get_messages 失败: %s", exc)
            return []

    def clear_session(self, user_id: str, role_id: int) -> None:
        try:
            c = self.client()
            c.delete(self._chat_key(user_id, role_id), self._meta_key(user_id, role_id))
            c.srem(self._sessions_key(user_id), str(role_id))
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis clear_session 失败: %s", exc)

    # ===== 会话元数据 =====
    @staticmethod
    def _meta_key(user_id: str, role_id: int) -> str:
        return f"session:{user_id}:{role_id}"

    @staticmethod
    def _sessions_key(user_id: str) -> str:
        return f"sessions:{user_id}"

    def touch_session(self, user_id: str, role_id: int) -> None:
        try:
            c = self.client()
            meta_key = self._meta_key(user_id, role_id)
            c.hincrby(meta_key, "turns", 1)
            c.hset(meta_key, mapping={"user_id": user_id, "role_id": role_id})
            c.expire(meta_key, settings.memory_ttl)
            c.sadd(self._sessions_key(user_id), str(role_id))
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis touch_session 失败: %s", exc)

    def get_session_meta(self, user_id: str, role_id: int) -> dict:
        try:
            return self.client().hgetall(self._meta_key(user_id, role_id))
        except Exception:  # noqa: BLE001
            return {}

    def list_sessions(self, user_id: str) -> list[str]:
        try:
            return sorted(self.client().smembers(self._sessions_key(user_id)))
        except Exception:  # noqa: BLE001
            return []

    # ===== 热问题 ZSet =====
    def incr_hot_query(self, query: str) -> None:
        try:
            self.client().zincrby("hot:queries", 1, query)
        except Exception:  # noqa: BLE001
            pass

    def top_queries(self, n: int = 10) -> list[dict]:
        try:
            rows = self.client().zrevrange("hot:queries", 0, n - 1, withscores=True)
            return [{"query": q, "count": int(s)} for q, s in rows]
        except Exception:  # noqa: BLE001
            return []

    # ===== 缓存 =====
    def cache_get(self, key: str) -> Optional[str]:
        try:
            return self.client().get(f"cache:query:{key}")
        except Exception:  # noqa: BLE001
            return None

    def cache_set(self, key: str, value: str, ttl: int = 600) -> None:
        try:
            self.client().setex(f"cache:query:{key}", ttl, value)
        except Exception:  # noqa: BLE001
            pass

    # ===== 运维可视化 =====
    def scan_keys(self, pattern: str = "*", limit: int = 500) -> list[str]:
        try:
            keys: list[str] = []
            for k in self.client().scan_iter(match=pattern, count=200):
                keys.append(k)
                if len(keys) >= limit:
                    break
            return keys
        except Exception:  # noqa: BLE001
            return []

    def key_info(self, key: str) -> dict:
        try:
            c = self.client()
            ktype = c.type(key)
            ttl = c.ttl(key)
            size = 0
            preview: Any = ""
            if ktype == "string":
                value = c.get(key) or ""
                size, preview = len(value), value[:200]
            elif ktype == "list":
                size = c.llen(key)
                preview = c.lrange(key, -3, -1)
            elif ktype == "set":
                size = c.scard(key)
                preview = list(c.smembers(key))[:5]
            elif ktype == "zset":
                size = c.zcard(key)
                preview = c.zrevrange(key, 0, 4, withscores=True)
            elif ktype == "hash":
                size = c.hlen(key)
                preview = c.hgetall(key)
            return {"key": key, "type": ktype, "ttl": ttl, "size": size, "preview": json.dumps(preview, ensure_ascii=False)[:300]}
        except Exception as exc:  # noqa: BLE001
            return {"key": key, "type": "unknown", "ttl": -1, "size": 0, "preview": str(exc)}

    def stats(self) -> dict:
        try:
            c = self.client()
            info = c.info()
            hits = int(info.get("keyspace_hits", 0))
            misses = int(info.get("keyspace_misses", 0))
            total = hits + misses
            keys = self.scan_keys(limit=2000)
            dist: dict[str, int] = {}
            for k in keys:
                t = c.type(k)
                dist[t] = dist.get(t, 0) + 1
            return {
                "dbsize": c.dbsize(),
                "used_memory_human": info.get("used_memory_human", ""),
                "keyspace_hits": hits,
                "keyspace_misses": misses,
                "hit_rate": round(hits / total, 4) if total else 0.0,
                "type_distribution": dist,
                "top_queries": self.top_queries(10),
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis stats 失败: %s", exc)
            return {}


redis_store = RedisStore()
