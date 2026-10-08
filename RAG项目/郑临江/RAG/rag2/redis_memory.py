# -*- coding: utf-8 -*-
"""Redis 短期记忆层：只记忆用户「当前对话」数据，TTL 自动过期。

键设计（统一前缀 ``rag2:``，可用 ``redis.prefix`` 修改）
--------------------------------------------------------------
Hash   : ``users``                 全量用户（username → JSON）
String : ``auth:<token>``          登录令牌 → username（TTL = token_ttl）
Hash   : ``sess:<sid>:meta``       会话元信息（user_id/role_id/title/turns/时间）
List   : ``sess:<sid>:msgs``       会话消息（按时间顺序 RPUSH）
zSet   : ``user:<username>:sessions`` 用户会话列表（score = 最近活跃时间）

会话与消息在每次写入时都会 ``expire(history_ttl)``，实现「短期记忆」自动过期；
不做长期画像 / 事实抽取（超出本期范围）。

    from rag2 import RedisMemory

    mem = RedisMemory(prefix="rag2:")
    token = mem.create_token("alice")
    sid = mem.create_session("alice", "general")
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .logging_config import get_logger

logger = get_logger("redis_memory")


def _now() -> float:
    return time.time()


class RedisMemory:
    """Redis 封装（懒连接）。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        prefix: str = "rag2:",
        history_ttl: int = 86400,
        token_ttl: int = 43200,
        socket_timeout: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.password = password or None
        self.prefix = prefix
        self.history_ttl = history_ttl
        self.token_ttl = token_ttl
        self.socket_timeout = socket_timeout
        self._client: Any = None

    # ------------------------------------------------------------------ 连接
    @property
    def client(self) -> Any:
        if self._client is None:
            import redis  # 懒加载

            # decode_responses=True：读写字符串自动按 UTF-8 编解码，拿到的是 str 而非 bytes，
            # 省去到处 .decode() 的麻烦。Redis 是内存数据库，读写极快，适合做会话这类短期状态。
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_timeout,
                health_check_interval=30,
            )
            self._client.ping()
            logger.info("已连接 Redis：%s:%d（db=%d）", self.host, self.port, self.db)
        return self._client

    def ping(self) -> dict[str, Any]:
        info = self.client.info()
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "version": info.get("redis_version"),
            "keys": self.client.dbsize(),
        }

    def _key(self, *parts: str) -> str:
        return self.prefix + ":".join(str(part) for part in parts)

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    # ================================================================== 用户
    def set_user(self, username: str) -> dict[str, Any]:
        """登记用户（幂等，已存在则返回原记录）。"""
        key = self._key("users")
        raw = self.client.hget(key, username)
        if raw:
            return json.loads(raw)
        record = {"username": username, "display_name": username, "created_at": _now()}
        self.client.hset(key, username, self._json(record))
        return record

    def get_user(self, username: str) -> dict[str, Any] | None:
        raw = self.client.hget(self._key("users"), username)
        return json.loads(raw) if raw else None

    def list_users(self) -> list[dict[str, Any]]:
        items = self.client.hgetall(self._key("users"))
        users = [json.loads(raw) for raw in items.values()]
        return sorted(users, key=lambda item: str(item.get("username")))

    # ================================================================== 令牌
    def create_token(self, username: str) -> str:
        """为用户名签发令牌，返回 token。"""
        token = uuid.uuid4().hex
        self.client.set(self._key("auth", token), username, ex=self.token_ttl)
        return token

    def resolve_token(self, token: str) -> str | None:
        """解析令牌 → 用户名；不存在/过期返回 None。"""
        if not token:
            return None
        return self.client.get(self._key("auth", token))

    def delete_token(self, token: str) -> None:
        self.client.delete(self._key("auth", token))

    # ================================================================== 会话
    def create_session(self, user_id: str, role_id: str, title: str = "") -> str:
        session_id = uuid.uuid4().hex
        now = _now()
        meta_key = self._key("sess", session_id, "meta")
        msgs_key = self._key("sess", session_id, "msgs")
        pipe = self.client.pipeline()
        pipe.hset(
            meta_key,
            mapping={
                "session_id": session_id,
                "user_id": user_id,
                "role_id": role_id,
                "title": title or "新会话",
                "turns": "0",
                "created_at": str(now),
                "last_active": str(now),
            },
        )
        pipe.expire(meta_key, self.history_ttl)
        pipe.expire(msgs_key, self.history_ttl)
        pipe.zadd(self._key("user", user_id, "sessions"), {session_id: now})
        pipe.execute()
        return session_id

    def session_meta(self, session_id: str) -> dict[str, Any] | None:
        raw = self.client.hgetall(self._key("sess", session_id, "meta"))
        return raw or None

    def update_session(self, session_id: str, **fields: Any) -> None:
        if not fields:
            return
        key = self._key("sess", session_id, "meta")
        self.client.hset(key, mapping={name: str(value) for name, value in fields.items()})
        self.client.hset(key, "last_active", str(_now()))
        self.client.expire(key, self.history_ttl)

    def touch_session(self, session_id: str) -> None:
        meta = self.session_meta(session_id)
        now = _now()
        self.client.hset(self._key("sess", session_id, "meta"), "last_active", str(now))
        if meta:
            self.client.zadd(self._key("user", meta.get("user_id", ""), "sessions"), {session_id: now})

    def list_sessions(self, user_id: str, limit: int = 30) -> list[dict[str, Any]]:
        ids = self.client.zrevrange(self._key("user", user_id, "sessions"), 0, max(limit - 1, 0))
        sessions: list[dict[str, Any]] = []
        for session_id in ids:
            meta = self.session_meta(session_id)
            if not meta:
                self.client.zrem(self._key("user", user_id, "sessions"), session_id)
                continue
            meta["message_count"] = self.message_count(session_id)
            sessions.append(meta)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        meta = self.session_meta(session_id)
        keys = [
            self._key("sess", session_id, "meta"),
            self._key("sess", session_id, "msgs"),
        ]
        self.client.delete(*keys)
        if meta:
            self.client.zrem(self._key("user", meta.get("user_id", ""), "sessions"), session_id)
        return meta is not None

    # ================================================================== 消息
    def append_message(self, session_id: str, message: dict[str, Any]) -> int:
        """追加一条消息（List），返回当前消息总数。"""
        message = dict(message)
        message.setdefault("ts", _now())
        key = self._key("sess", session_id, "msgs")
        pipe = self.client.pipeline()
        pipe.rpush(key, self._json(message))
        pipe.expire(key, self.history_ttl)
        return int(pipe.execute()[0])

    def messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        key = self._key("sess", session_id, "msgs")
        start = -limit if limit else 0
        raw = self.client.lrange(key, start, -1)
        return [json.loads(item) for item in raw]

    def recent_context(self, session_id: str, turns: int) -> list[dict[str, Any]]:
        """取最近 ``turns`` 轮（1 轮 = 用户 + 助手）的消息，用于拼接提示词。"""
        if turns <= 0:
            return []
        items = self.messages(session_id, limit=turns * 2)
        return [item for item in items if item.get("role") in {"user", "assistant"}]

    def message_count(self, session_id: str) -> int:
        return int(self.client.llen(self._key("sess", session_id, "msgs")))
