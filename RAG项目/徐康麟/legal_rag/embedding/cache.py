# -*- coding: utf-8 -*-
"""嵌入缓存后端：内容 hash -> 向量。

**为什么必须有缓存**
----------------
VM 无 GPU，bge-m3 一次嵌入是 CPU 上几十毫秒到几百毫秒；重复入库、重复提问
（同一段法律条文被反复检索）会产生大量完全相同的嵌入请求。用「内容 hash」而不是
「文本本身」做 key，既省内存又不泄露原文。

**两种后端**
------------
* :class:`RedisEmbedCache` —— 优先。key 统一 ``legal_rag:embed:<sha256>``，
  **只读写 ``legal_rag:`` 前缀内的 key**，绝不 FLUSHDB/FLUSHALL/KEYS *；
* :class:`MemoryEmbedCache` —— Redis 不可用时的兜底（线程安全 LRU）。
  兜底是「按配置降级」而不是「静默失败」：降级时打 WARNING 并说明原因。

向量序列化用 ``struct.pack("<f")`` + base64：1024 维 = 4KB，比 JSON 文本
（约 20KB）紧凑，而且解析没有浮点精度损失（本来就是 float32）。
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import struct
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

#: 嵌入缓存 key 的前缀（相对 CACHE_PREFIX，即最终形如 legal_rag:embed:<hash>）
EMBED_KEY_PREFIX = "embed:"


def content_key(text: str, model: str = "", extra: str = "") -> str:
    """由「模型名 + 文本」派生稳定缓存 key（sha256 十六进制）。

    带上模型名：换模型（bge-m3 -> 别的）后旧缓存自然失效，不会串维度。
    """
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    if extra:
        digest.update(extra.encode("utf-8"))
        digest.update(b"\x00")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def pack_vector(vector: Sequence[float]) -> bytes:
    """float 向量 -> base64 文本（float32 紧凑编码）。"""
    packed = struct.pack(f"<{len(vector)}f", *[float(v) for v in vector])
    return base64.b64encode(packed)


def unpack_vector(raw: bytes | str) -> list[float]:
    """base64 文本 -> float 向量；长度非法时抛 ValueError（由调用方按未命中处理）。"""
    if isinstance(raw, str):
        raw = raw.encode("ascii", errors="strict")
    packed = base64.b64decode(raw, validate=True)
    if not packed or len(packed) % 4:
        raise ValueError(f"嵌入缓存内容长度非法: {len(packed)} 字节")
    count = len(packed) // 4
    return list(struct.unpack(f"<{count}f", packed))


class EmbedCache(ABC):
    """嵌入缓存接口：批量读、批量写、统计。"""

    name: str = "base"

    def __init__(self, ttl: int = 7 * 24 * 3600) -> None:
        self.ttl = int(ttl)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        #: 缓存自身故障（读写异常）留下的原因；非空表示「此刻缓存其实没在生效」，
        #: 于是每一条都会被判定为未命中、**真的去调 Ollama**。调用方（嵌入后端）
        #: 会把它打进汇总日志，避免「以为在用缓存、其实每次都走 HTTP」这种静默分叉。
        self.degraded_reason: str = ""

    # ---------- 子类实现 ----------
    @abstractmethod
    def _get_many(self, keys: list[str]) -> dict[str, list[float]]:
        ...

    @abstractmethod
    def _set_many(self, items: list[tuple[str, list[float]]]) -> int:
        ...

    # ---------- 公共入口（带命中统计） ----------
    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        """批量查询；返回 ``{key: vector}``，未命中的 key 不出现。

        缓存**故障**不会阻断主链路（按未命中处理），但会记一条 WARNING 并把原因留在
        :attr:`degraded_reason` 上 —— 这样汇总日志能区分「正常未命中」与「缓存坏了，
        所以条条都得走 Ollama」，这是定位「为什么这批全都在调 HTTP」的关键线索。
        """
        if not keys:
            return {}
        try:
            found = self._get_many(keys)
        except Exception as exc:  # noqa: BLE001 - 缓存故障绝不阻断主链路
            self.degraded_reason = f"读取失败 {type(exc).__name__}: {exc}"
            logger.warning("嵌入缓存读取失败（按未命中处理，本批将全部走 HTTP）：%s",
                           self.degraded_reason, exc_info=True)
            found = {}
        with self._lock:
            self.hits += len(found)
            self.misses += max(len(keys) - len(found), 0)
        return found

    def set_many(self, items: list[tuple[str, list[float]]]) -> int:
        """批量写入；返回成功写入条数（失败只记日志，不影响调用方）。"""
        if not items:
            return 0
        try:
            return int(self._set_many(items))
        except Exception as exc:  # noqa: BLE001
            self.degraded_reason = f"写入失败 {type(exc).__name__}: {exc}"
            logger.warning("嵌入缓存写入失败（跳过缓存，不影响本次返回）：%s",
                           self.degraded_reason, exc_info=True)
            return 0

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "backend": self.name,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "ttl": self.ttl,
            }

    def label(self) -> str:
        """给日志用的一行标签：``redis`` / ``memory（降级: ...）``。

        **不改变** :meth:`stats` 的返回结构（它已进入 ``OllamaEmbedder.health()``），
        日志需要的信息单独走本方法。
        """
        return f"{self.name}（降级：{self.degraded_reason}）" if self.degraded_reason else self.name

    def reset_stats(self) -> None:
        with self._lock:
            self.hits = 0
            self.misses = 0

    def close(self) -> None:
        """释放资源（Redis 连接等）；内存实现为空操作。"""


class MemoryEmbedCache(EmbedCache):
    """线程安全 LRU 内存缓存（Redis 缺席时的兜底）。"""

    name = "memory"

    def __init__(self, ttl: int = 7 * 24 * 3600, max_items: int = 2048) -> None:
        super().__init__(ttl)
        self.max_items = max(int(max_items), 1)
        self._store: OrderedDict[str, list[float]] = OrderedDict()

    def _get_many(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        with self._lock:
            for key in keys:
                vector = self._store.get(key)
                if vector is None:
                    continue
                self._store.move_to_end(key)
                found[key] = vector
        return found

    def _set_many(self, items: list[tuple[str, list[float]]]) -> int:
        with self._lock:
            for key, vector in items:
                self._store[key] = vector
                self._store.move_to_end(key)
            while len(self._store) > self.max_items:
                self._store.popitem(last=False)
        return len(items)

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class RedisEmbedCache(EmbedCache):
    """Redis 嵌入缓存；只用 ``legal_rag:embed:`` 前缀，绝不 flush 用户数据。"""

    name = "redis"

    def __init__(self, url: str, *, prefix: str = "legal_rag:", ttl: int = 7 * 24 * 3600,
                 connect_timeout: float = 3.0, socket_timeout: float = 5.0,
                 max_connections: int = 16) -> None:
        super().__init__(ttl)
        import redis  # type: ignore

        self.prefix = f"{prefix}{EMBED_KEY_PREFIX}"
        self._client = redis.from_url(
            url,
            decode_responses=False,
            socket_connect_timeout=float(connect_timeout),
            socket_timeout=float(socket_timeout),
            max_connections=max(int(max_connections), 1),
            health_check_interval=30,
        )
        self._client.ping()   # 连不上直接抛，由工厂决定降级
        logger.info("嵌入缓存使用 Redis: %s（key 前缀 %s）", url, self.prefix)

    # ---------- key ----------
    def key_for(self, digest: str) -> str:
        return f"{self.prefix}{digest}"

    def _get_many(self, keys: list[str]) -> dict[str, list[float]]:
        full_keys = [self.key_for(k) for k in keys]
        raw_values = self._client.mget(full_keys)
        found: dict[str, list[float]] = {}
        for key, raw in zip(keys, raw_values):
            if raw is None:
                continue
            try:
                found[key] = unpack_vector(raw)
            except (ValueError, binascii.Error):
                # 脏数据（旧格式/被截断）：删掉这一条并当作未命中
                logger.warning("嵌入缓存条目损坏，已删除: %s", self.key_for(key))
                try:
                    self._client.delete(self.key_for(key))
                except Exception:  # noqa: BLE001
                    pass
        return found

    def _set_many(self, items: list[tuple[str, list[float]]]) -> int:
        if not items:
            return 0
        pipe = self._client.pipeline(transaction=False)
        for key, vector in items:
            pipe.set(self.key_for(key), pack_vector(vector), ex=(self.ttl or None))
        pipe.execute()
        return len(items)

    def count_own_keys(self, limit: int = 10000) -> int:
        """只统计自己前缀下的 key 数量（用 SCAN，绝不 KEYS *）。"""
        total = 0
        for _ in self._client.scan_iter(match=f"{self.prefix}*", count=200):
            total += 1
            if total >= limit:
                break
        return total

    def health(self) -> dict:
        info: dict = {"provider": self.name, "prefix": self.prefix, "ttl": self.ttl}
        try:
            info["ping"] = bool(self._client.ping())
        except Exception as exc:  # noqa: BLE001
            info["ping"] = False
            info["error"] = str(exc)
        return info

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover
            logger.debug("关闭 Redis 嵌入缓存失败", exc_info=True)


def build_embed_cache(config, *, enabled: bool | None = None,
                      backend: str | None = None) -> EmbedCache | None:
    """按配置构建嵌入缓存；关闭时返回 ``None``。

    读的是 t1 冻结的 ``RagConfig.cache``（``CACHE_PREFIX`` 默认 ``legal_rag:``），
    最终 key 形如 ``legal_rag:embed:<sha256>``。
    """
    cache_cfg = getattr(config, "cache", None)
    if cache_cfg is None:
        return None

    is_enabled = getattr(cache_cfg, "enabled", True) if enabled is None else enabled
    if not is_enabled:
        logger.info("嵌入缓存已关闭（EMBED_CACHE_ENABLED=false）")
        return None

    chosen = str(backend if backend is not None else getattr(cache_cfg, "backend", "redis")).lower()
    prefix = str(getattr(cache_cfg, "prefix", "legal_rag:"))
    ttl = int(getattr(cache_cfg, "embed_ttl", 7 * 24 * 3600))
    max_items = int(getattr(cache_cfg, "memory_max_items", 2048))

    if chosen in ("none", "off", "disabled"):
        logger.info("嵌入缓存后端=none，跳过缓存")
        return None

    if chosen in ("redis", "auto"):
        url = str(getattr(config, "redis_url", "") or "")
        if url:
            try:
                return RedisEmbedCache(
                    url, prefix=prefix, ttl=ttl,
                    connect_timeout=float(getattr(cache_cfg, "connect_timeout", 3.0)),
                    socket_timeout=float(getattr(cache_cfg, "socket_timeout", 5.0)),
                )
            except Exception as exc:  # noqa: BLE001 - 降级是预期路径
                logger.warning(
                    "Redis 嵌入缓存不可用（%s: %s），降级为进程内 LRU 内存缓存"
                    "（重启后缓存失效，只影响速度不影响正确性）",
                    type(exc).__name__, exc)
                return MemoryEmbedCache(ttl=ttl, max_items=max_items)
        logger.warning("未配置 redis_url，嵌入缓存降级为进程内 LRU")

    return MemoryEmbedCache(ttl=ttl, max_items=max_items)


def iter_digests(texts: Iterable[str], model: str = "") -> list[str]:
    return [content_key(text, model) for text in texts]
