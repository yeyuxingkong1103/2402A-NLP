"""短期记忆（多轮对话上下文）。

MVP 用进程内内存（按 session_id 隔离）；生产切 Redis（List + TTL）。
大模型本身无状态，短期记忆在此层维护。
"""
from __future__ import annotations

import json
from collections import deque

from ..config import Settings
from ..logging_config import get_logger

log = get_logger("memory")


class MemoryStore:
    """短期记忆接口。"""

    def add(self, session_id: str, role: str, content: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_history(self, session_id: str, n: int | None = None) -> list[dict]:  # pragma: no cover
        """按时间顺序返回 [{"role", "content"}, ...]；n 为 None 时返回全部，否则只取**最近** n 条。

        两个后端都只承诺"最多" n 条，会话刚开始时返回不足 n 条是正常的——调用方
        不要按 n 的长度做假设。
        """
        raise NotImplementedError

    def clear(self, session_id: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def ping(self) -> bool:  # pragma: no cover
        """后端是否真的可用。"""
        raise NotImplementedError

    def describe(self) -> str:
        """健康检查用的状态串。

        比 ping() 多一档 degraded：后端不可达但已降级到进程内内存时，服务是能用的，
        却已经不是「ok」——不区分的话 /health 会把降级报成完全正常（这正是之前的 bug）。
        """
        return "ok" if self.ping() else "unavailable"


class InMemoryMemoryStore(MemoryStore):
    """进程内实现，按 session 存最近 max_turns 轮（一轮 = 一问一答 2 条）。"""

    # 进程内实现没有任何过期机制（Redis 版有 TTL），而 session_id 完全由客户端给，
    # 轮换 id 的客户端能让 _store 无限涨；Redis 挂掉降级到进程内时这层保护恰好消失。
    # 所以按"最多记多少个会话"兜底：超了就丢最久没被碰过的那个。
    MAX_SESSIONS = 1000

    def __init__(self, max_turns: int = 10):
        # max_turns <= 0 时两个后端语义相反：进程内 deque(maxlen=0) 一条不存，
        # Redis 算出 ltrim(0,-1) 即"不裁剪"、反而无界增长。统一夹到至少 1 轮。
        self.max_turns = max(1, int(max_turns))
        self._store: dict[str, deque] = {}

    def _touch(self, session_id: str) -> deque:
        """取该会话的双端队列，并把它标记成"最近用过"（dict 保序 = LRU 顺序）。"""
        dq = self._store.pop(session_id, None)
        if dq is None:
            if len(self._store) >= self.MAX_SESSIONS:
                self._store.pop(next(iter(self._store)), None)  # 队首 = 最久没碰过的
            dq = deque(maxlen=self.max_turns * 2)
        self._store[session_id] = dq
        return dq

    def add(self, session_id: str, role: str, content: str) -> None:
        self._touch(session_id).append({"role": role, "content": content})

    def get_history(self, session_id: str, n: int | None = None) -> list[dict]:
        dq = self._store.get(session_id)
        if dq is None:
            return []
        self._touch(session_id)  # 读也算用过
        items = list(dq)
        return items[-n:] if n is not None else items

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def ping(self) -> bool:
        # 进程内字典不会连不上，永远可用。
        return True


class RedisMemoryStore(MemoryStore):
    """Redis 实现：List 保存对话、LRANGE 取最近、EXPIRE 设 TTL。"""

    def __init__(self, redis_url: str, max_turns: int = 10):
        import redis

        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.max_turns = max(1, int(max_turns))  # 见 InMemoryMemoryStore：0 会让 ltrim 变成"不裁剪"

    def _key(self, session_id: str) -> str:
        # 统一加前缀：REDIS_URL 默认连的是 db 0（见 config.py 的默认值），那是 redis 的
        # 默认库、同一实例里可能有别的键。带前缀既能在 redis-cli 里一眼认出这批会话，
        # 也免得裸 session_id 和别人撞名——撞了会互相覆盖，且两边都看不出是谁写的。
        return f"memory:{session_id}"

    def add(self, session_id: str, role: str, content: str) -> None:
        try:
            key = self._key(session_id)
            self.client.rpush(key, json.dumps({"role": role, "content": content}, ensure_ascii=False))
            self.client.ltrim(key, -self.max_turns * 2, -1)
            self.client.expire(key, 24 * 3600)  # TTL 1 天
        except Exception as exc:  # noqa: BLE001
            # 启动时 ping 通过、运行中 Redis 才掉线（Docker 被关 / 容器被杀）会走到这。
            # 不重抛：记忆写不进去就写不进去，别让整轮对话 500。
            log.warning("Redis 写记忆失败（运行中掉线？），本轮上下文将丢失: %s", exc)

    def get_history(self, session_id: str, n: int | None = None) -> list[dict]:
        try:
            raw = self.client.lrange(self._key(session_id), 0, -1)
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis 读历史失败（运行中掉线？），本轮按无历史继续: %s", exc)
            return []

        # 逐条解析、跳过坏元素。以前是整个列表推导包在同一个 except 里：只要有一条
        # 非法 JSON（旧版本残留、被别的写入方污染、部分写入），该会话历史**整段静默消失**
        # 并返回 []，/health 还报 ok，而后续轮次继续往这个坏键里追加。
        hist: list[dict] = []
        bad = 0
        for x in raw:
            try:
                hist.append(json.loads(x))
            except (TypeError, ValueError):
                bad += 1
        if bad:
            log.warning("Redis 历史里有 %d 条非法 JSON，已跳过（会话 %s）", bad, session_id)
        return hist[-n:] if n is not None else hist

    def clear(self, session_id: str) -> None:
        try:
            self.client.delete(self._key(session_id))
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis 清历史失败（运行中掉线？）: %s", exc)

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis 不可达（%s）: %s", self.client.connection_pool.connection_kwargs.get("host"), exc)
            return False


class DegradedMemoryStore(InMemoryMemoryStore):
    """Redis 起不来时的降级实现：存储行为与进程内内存完全一致，只多一个 reason。

    为什么要单独一个类而不是直接返回 InMemoryMemoryStore：那样 /health 会报 ok，
    等于把降级伪装成正常。降级的代价是实打实的（重启后上下文全丢），必须能看见。
    """

    def __init__(self, max_turns: int, reason: str):
        super().__init__(max_turns)
        self.reason = reason

    def describe(self) -> str:
        return "degraded"

    def status_detail(self) -> str:
        # /health 用 getattr(p.memory, "status_detail", None) 探这个可选方法（health.py），
        # 有值就放进 details.memory。所以：只有需要解释"为什么不是 ok"的实现才该有这个方法，
        # 正常实现不要加一个返回空串的同名方法。
        return self.reason


def create_memory_store(settings: Settings) -> MemoryStore:
    if settings.memory_backend == "redis":
        store = RedisMemoryStore(settings.redis_url, settings.memory_max_turns)
        # 构造是不连的（redis.Redis.from_url 懒连接），所以 Redis 挂着时启动期一切正常，
        # 直到第一轮对话才 500。这里主动探一次，把它变成启动期就可见的降级。
        if store.ping():
            log.info("短期记忆使用 Redis: %s", settings.redis_url)
            return store
        log.warning(
            "短期记忆降级为进程内内存：Redis(%s) 不可达。服务可正常对话，"
            "但上下文只活在进程里——重启即丢（这正是 Redis 版要买到的东西）。"
            "检查 Docker Desktop 是否在运行、容器 redis-memory 是否起来。",
            settings.redis_url,
        )
        return DegradedMemoryStore(
            settings.memory_max_turns, f"Redis 不可达: {settings.redis_url}"
        )
    log.info("短期记忆使用进程内内存（memory）")
    return InMemoryMemoryStore(settings.memory_max_turns)
