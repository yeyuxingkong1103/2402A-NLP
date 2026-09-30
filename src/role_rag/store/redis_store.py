"""Redis 存储层：聊天记录、短期记忆、用户与权限、检索缓存、限流、统计。

键设计（统一前缀 ``rolerag:``，可用 ``redis.prefix`` 修改）
--------------------------------------------------------------
String : ``kb:version``            知识库版本号（入库后自增，用于内存索引失效）
         ``cache:retr:<hash>``     检索结果缓存（TTL = redis.cache_ttl）
         ``sess:<sid>:summary``    会话摘要（短期记忆的压缩形式）
         ``rl:<uid>:<分钟>``       每分钟限流计数器
Hash   : ``users``                 全量用户（username → JSON）
         ``user:<uid>:profile``    用户画像（昵称、偏好、累计提问数）
         ``sess:<sid>:meta``       会话元信息（owner/role/title/轮数/时间）
         ``stats:<日期>``          当日运行指标
List   : ``sess:<sid>:msgs``       会话原始消息（按时间顺序 RPUSH）
Set    : ``online``                在线用户集合
         ``sess:<sid>:cites``      本会话引用过的文档 id（去重）
zSet   : ``user:<uid>:sessions``   用户的会话列表（score = 最近活跃时间）
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from typing import Any, Iterable, Sequence

from ..config import Config, get_config
from ..errors import DependencyError
from ..logging_conf import get_logger

logger = get_logger(__name__)


def _now() -> float:
    return time.time()


def _today() -> str:
    return time.strftime("%Y-%m-%d")


class RedisStore:
    """Redis 封装（懒连接）。"""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        section = self.config.redis
        self.host = str(section.get("host", "127.0.0.1"))
        self.port = int(section.get("port", 6379))
        self.password = str(section.get("password", "") or "") or None
        self.db = int(section.get("db", 0))
        self.prefix = str(section.get("prefix", "rolerag:"))
        self.socket_timeout = float(section.get("socket_timeout", 5))
        self.history_ttl = int(section.get("history_ttl", 604800))
        self.summary_ttl = int(section.get("summary_ttl", 604800))
        self.cache_ttl = int(section.get("cache_ttl", 900))
        self.online_ttl = int(section.get("online_ttl", 300))
        self.rate_limit_per_min = int(section.get("rate_limit_per_min", 60))
        self._client: Any = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 连接
    @property
    def client(self) -> Any:
        if self._client is None:
            import redis

            try:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    db=self.db,
                    decode_responses=True,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_timeout,
                    health_check_interval=30,
                )
                self._client.ping()
            except Exception as exc:
                self._client = None
                raise DependencyError(
                    f"无法连接 Redis：{self.host}:{self.port}（{exc}）。"
                    "请在 WSL 中执行：sudo service redis-server start",
                    host=self.host,
                    port=self.port,
                ) from exc
        return self._client

    def ping(self) -> dict[str, Any]:
        info = self.client.info()
        return {"host": self.host, "port": self.port, "db": self.db,
                "version": info.get("redis_version"), "keys": self.client.dbsize()}

    def key(self, *parts: str) -> str:
        return self.prefix + ":".join(str(part) for part in parts)

    def _json(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    # ================================================================== 用户
    def seed_users(self, seeds: Iterable[dict[str, Any]], password_hasher) -> int:
        """首次启动写入内置用户（已存在则跳过，不覆盖）。"""

        created = 0
        for item in seeds or []:
            username = str(item.get("username", "")).strip()
            if not username or self.get_user(username):
                continue
            self.create_user(
                username=username,
                password=str(item.get("password", "")),
                display_name=str(item.get("display_name", username)),
                roles=[str(x) for x in (item.get("roles") or [])],
                is_admin=bool(item.get("is_admin", False)),
                password_hasher=password_hasher,
            )
            created += 1
        if created:
            logger.info("已写入内置用户 %d 个", created)
        return created

    def create_user(
        self,
        username: str,
        password: str,
        display_name: str,
        roles: Sequence[str],
        is_admin: bool = False,
        password_hasher=None,
    ) -> dict[str, Any]:
        if password_hasher is None:  # pragma: no cover - 正常路径都会传入
            raise DependencyError("create_user 需要 password_hasher")
        record = {
            "username": username,
            "display_name": display_name or username,
            "password_hash": password_hasher(password),
            "roles": list(roles) or ["*"],
            "is_admin": bool(is_admin),
            "created_at": _now(),
        }
        self.client.hset(self.key("users"), username, self._json(record))
        self.client.hset(
            self.key("user", username, "profile"),
            mapping={"display_name": record["display_name"], "created_at": str(record["created_at"]),
                     "questions": "0"},
        )
        return record

    def get_user(self, username: str) -> dict[str, Any] | None:
        raw = self.client.hget(self.key("users"), username)
        return json.loads(raw) if raw else None

    def list_users(self) -> list[dict[str, Any]]:
        items = self.client.hgetall(self.key("users"))
        users = []
        for raw in items.values():
            record = json.loads(raw)
            record.pop("password_hash", None)
            users.append(record)
        return sorted(users, key=lambda item: str(item.get("username")))

    # ================================================================== 会话
    def create_session(self, user_id: str, role_id: str, title: str = "") -> str:
        session_id = uuid.uuid4().hex
        now = _now()
        meta_key = self.key("sess", session_id, "meta")
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
        pipe.expire(self.key("sess", session_id, "msgs"), self.history_ttl)
        pipe.zadd(self.key("user", user_id, "sessions"), {session_id: now})
        pipe.execute()
        return session_id

    def session_meta(self, session_id: str) -> dict[str, Any] | None:
        raw = self.client.hgetall(self.key("sess", session_id, "meta"))
        return raw or None

    def session_owner(self, session_id: str) -> str | None:
        meta = self.session_meta(session_id)
        return str(meta.get("user_id")) if meta else None

    def update_session(self, session_id: str, **fields: Any) -> None:
        if not fields:
            return
        key = self.key("sess", session_id, "meta")
        self.client.hset(key, mapping={name: str(value) for name, value in fields.items()})
        self.client.hset(key, "last_active", str(_now()))
        self.client.expire(key, self.history_ttl)

    def touch_session(self, session_id: str) -> None:
        meta = self.session_meta(session_id)
        now = _now()
        self.client.hset(self.key("sess", session_id, "meta"), "last_active", str(now))
        if meta:
            self.client.zadd(self.key("user", meta.get("user_id", ""), "sessions"), {session_id: now})

    def append_message(self, session_id: str, message: dict[str, Any]) -> int:
        """追加一条消息（List），返回当前消息总数。"""

        message = dict(message)
        message.setdefault("ts", _now())
        key = self.key("sess", session_id, "msgs")
        pipe = self.client.pipeline()
        pipe.rpush(key, self._json(message))
        pipe.expire(key, self.history_ttl)
        length = int(pipe.execute()[0])
        return length

    def messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        key = self.key("sess", session_id, "msgs")
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
        return int(self.client.llen(self.key("sess", session_id, "msgs")))

    def get_summary(self, session_id: str) -> str:
        return self.client.get(self.key("sess", session_id, "summary")) or ""

    def set_summary(self, session_id: str, summary: str) -> None:
        self.client.set(self.key("sess", session_id, "summary"), summary, ex=self.summary_ttl)

    def list_sessions(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        ids = self.client.zrevrange(self.key("user", user_id, "sessions"), 0, max(limit - 1, 0))
        sessions: list[dict[str, Any]] = []
        for session_id in ids:
            meta = self.session_meta(session_id)
            if not meta:
                self.client.zrem(self.key("user", user_id, "sessions"), session_id)
                continue
            meta["message_count"] = self.message_count(session_id)
            sessions.append(meta)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        meta = self.session_meta(session_id)
        keys = [
            self.key("sess", session_id, "meta"),
            self.key("sess", session_id, "msgs"),
            self.key("sess", session_id, "summary"),
            self.key("sess", session_id, "cites"),
        ]
        self.client.delete(*keys)
        if meta:
            self.client.zrem(self.key("user", meta.get("user_id", ""), "sessions"), session_id)
        return meta is not None

    def add_citations(self, session_id: str, doc_ids: Iterable[str]) -> int:
        ids = [str(item) for item in doc_ids if item]
        if not ids:
            return 0
        key = self.key("sess", session_id, "cites")
        added = int(self.client.sadd(key, *ids))
        self.client.expire(key, self.history_ttl)
        return added

    def session_citations(self, session_id: str) -> list[str]:
        return sorted(self.client.smembers(self.key("sess", session_id, "cites")))

    # ================================================================== 记忆
    def profile(self, user_id: str) -> dict[str, Any]:
        return self.client.hgetall(self.key("user", user_id, "profile")) or {}

    def update_profile(self, user_id: str, **fields: Any) -> None:
        if fields:
            self.client.hset(
                self.key("user", user_id, "profile"),
                mapping={name: str(value) for name, value in fields.items()},
            )

    def incr_profile(self, user_id: str, field: str, amount: int = 1) -> int:
        return int(self.client.hincrby(self.key("user", user_id, "profile"), field, amount))

    def remember_facts(self, user_id: str, role_id: str, facts: Sequence[dict[str, Any]]) -> int:
        """Redis 侧保存近期事实（List，最多保留 50 条），Milvus 侧负责向量召回。"""

        if not facts:
            return 0
        key = self.key("user", user_id, "facts", role_id)
        pipe = self.client.pipeline()
        for fact in facts:
            item = dict(fact)
            item.setdefault("ts", _now())
            item.setdefault("role_id", role_id)
            pipe.lpush(key, self._json(item))
        pipe.ltrim(key, 0, 49)
        pipe.expire(key, self.history_ttl)
        pipe.execute()
        return len(facts)

    def recent_facts(self, user_id: str, role_id: str, limit: int = 10) -> list[dict[str, Any]]:
        raw = self.client.lrange(self.key("user", user_id, "facts", role_id), 0, max(limit - 1, 0))
        return [json.loads(item) for item in raw]

    # ================================================================== 缓存
    @staticmethod
    def retrieval_cache_key(role_id: str, mode: str, top_k: int, query: str, scope: str = "") -> str:
        digest = hashlib.sha1(f"{role_id}|{mode}|{top_k}|{scope}|{query}".encode("utf-8")).hexdigest()
        return f"cache:retr:{digest[:32]}"

    def cache_get(self, name: str) -> Any | None:
        raw = self.client.get(self.key(name))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def cache_set(self, name: str, value: Any, ttl: int | None = None) -> None:
        self.client.set(self.key(name), self._json(value), ex=ttl or self.cache_ttl)

    def cache_stats(self) -> dict[str, int]:
        hits = int(self.client.get(self.key("stats", "cache_hit")) or 0)
        misses = int(self.client.get(self.key("stats", "cache_miss")) or 0)
        return {"hit": hits, "miss": misses}

    def mark_cache(self, hit: bool) -> None:
        self.client.incr(self.key("stats", "cache_hit" if hit else "cache_miss"))

    # ================================================================== 限流
    def check_rate_limit(self, user_id: str, per_minute: int | None = None) -> tuple[bool, int]:
        limit = per_minute or self.rate_limit_per_min
        bucket = time.strftime("%Y%m%d%H%M")
        key = self.key("rl", user_id, bucket)
        count = int(self.client.incr(key))
        if count == 1:
            self.client.expire(key, 120)
        return count <= limit, max(limit - count, 0)

    # ================================================================== 统计
    def incr_stat(self, name: str, amount: int = 1, daily: bool = True) -> None:
        pipe = self.client.pipeline()
        pipe.hincrby(self.key("stats", "total"), name, amount)
        if daily:
            key = self.key("stats", "day", _today())
            pipe.hincrby(key, name, amount)
            pipe.expire(key, 60 * 60 * 24 * 30)
        pipe.execute()

    def stats(self) -> dict[str, Any]:
        total = self.client.hgetall(self.key("stats", "total")) or {}
        today = self.client.hgetall(self.key("stats", "day", _today())) or {}
        return {
            "total": {key: int(value) for key, value in total.items()},
            "today": {key: int(value) for key, value in today.items()},
            "cache": self.cache_stats(),
            "online": len(self.online_users()),
            "kb_version": self.kb_version(),
        }

    # ================================================================== 在线
    def mark_online(self, user_id: str) -> None:
        self.client.sadd(self.key("online"), user_id)
        self.client.set(self.key("online", user_id), str(_now()), ex=self.online_ttl)

    def online_users(self) -> list[str]:
        members = self.client.smembers(self.key("online")) or set()
        alive: list[str] = []
        for user_id in members:
            if self.client.exists(self.key("online", user_id)):
                alive.append(user_id)
            else:
                self.client.srem(self.key("online"), user_id)
        return sorted(alive)

    # ============================================================== 知识库版本
    def kb_version(self) -> int:
        return int(self.client.get(self.key("kb", "version")) or 0)

    def bump_kb_version(self) -> int:
        return int(self.client.incr(self.key("kb", "version")))

    def clear_cache(self) -> int:
        """清理检索缓存（入库后调用），返回删除的键数量。"""

        removed = 0
        pattern = self.key("cache", "*")
        for key in self.client.scan_iter(match=pattern, count=500):
            self.client.delete(key)
            removed += 1
        return removed

    def describe(self) -> dict[str, Any]:
        """列出本项目在 Redis 中的键结构（供前端「记忆面板」展示）。"""

        counts: dict[str, int] = {}
        pattern = self.key("*")
        for key in self.client.scan_iter(match=pattern, count=500):
            short = key[len(self.prefix):]
            group = short.split(":")[0]
            counts[group] = counts.get(group, 0) + 1
        return {"prefix": self.prefix, "groups": counts, "db": self.db}


_store: RedisStore | None = None
_store_lock = threading.Lock()


def get_redis(config: Config | None = None) -> RedisStore:
    """获取全局 RedisStore 单例。"""

    global _store
    with _store_lock:
        if _store is None:
            _store = RedisStore(config)
        return _store
