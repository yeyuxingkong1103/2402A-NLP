# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

F2 修复：线程安全懒加载单例 + 带 TTL 的探测结果缓存 —— ``研发/service/singletons.py``。

## 为什么需要它（问题定位）

工单原文「瓶颈一：每个请求可能都在独立初始化重量级组件」在本仓库的**真实等价形态**是：

``app/core/embedder.py:299``（``embed_texts``）每一次调用都执行
``resolve_embedding_backend(config, probe=True, ...)``，而它内部会发起一次
``GET http://127.0.0.1:11434/api/tags`` HTTP 探测。

也就是说：**每一次问答都多打一次 Ollama 探测接口**。20 并发 × 每秒若干次
= 每秒几十次无意义的 ``/api/tags`` 往返，既占连接又占服务端处理时间。

> 诚实性说明：本仓库**没有** DeepDoc、也没有「每个请求 new 一个解析器」的写法
> （``QAEngine`` 是进程级单例，索引走 ``retriever._BUNDLE_CACHE`` 模块级缓存，
> 见 ``优化/瓶颈排查与修复报告.md`` §2.1 的核实证据）。
> 因此这里修复的是**真实存在**的那一条：后端探测未缓存。

## 实现

* :func:`probe_cached` —— 键 → （过期时间, 值）的进程级缓存；双检锁；命中返回旧值时
  **不阻塞**；探测失败按短 TTL 缓存（避免故障时每请求都打超时）；
* :class:`Singleton` —— 通用懒加载单例（双检锁），用于索引 bundle / LLM 客户端 / 解析器；
* 全部提供 ``stats()`` 与 ``reset()``（测试要能重置）。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Generic, TypeVar

from .perf_config import PerfConfig, get_perf_config

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

T = TypeVar("T")


class _CachedError:
    """把「探测失败」也缓存起来，并在 TTL 内**重新抛出同一个异常**。

    为什么不能只缓存 ``None``：调用方（``resolve_embedding_backend`` 等）期望的是
    「拿到值」或「抛异常」二选一；静默返回 ``None`` 会把真正的失败信息丢掉
    （实测由 ``测试/test_01_单例唯一性.py`` 抓出）。
    """

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class _Entry:
    __slots__ = ("expires_at", "value")

    def __init__(self, expires_at: float, value: Any) -> None:
        self.expires_at = expires_at
        self.value = value


class ProbeCache:
    """带 TTL 的探测结果缓存（线程安全，双检锁）。"""

    def __init__(self, *, ttl_s: float, error_ttl_s: float | None = None) -> None:
        self.ttl_s = float(ttl_s)
        self.error_ttl_s = float(error_ttl_s if error_ttl_s is not None else max(1.0, ttl_s / 6.0))
        self._data: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.stores = 0

    def get(self, key: str) -> tuple[bool, Any]:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return False, None
            if item.expires_at <= now:
                self._data.pop(key, None)
                self.misses += 1
                return False, None
            self.hits += 1
            return True, item.value

    def put(self, key: str, value: Any, *, ttl_s: float | None = None) -> None:
        ttl = self.ttl_s if ttl_s is None else float(ttl_s)
        with self._lock:
            self._data[key] = _Entry(time.time() + ttl, value)
            self.stores += 1

    def call(self, key: str, factory: Callable[[], T], *, on_error: Callable[[Exception], None] | None = None) -> T:
        """命中直接返回；未命中才调 ``factory``（**锁外调用**，不阻塞其它键）。

        失败也会被缓存（短 TTL），且在 TTL 内**重新抛出同一个异常** ——
        避免故障期间每个请求都真探一次（等一次超时），也不丢失败原因。
        """
        hit, value = self.get(key)
        if hit:
            if isinstance(value, _CachedError):
                raise value.exc
            return value  # type: ignore[return-value]
        try:
            value = factory()
        except Exception as exc:
            self.errors += 1
            if on_error is not None:
                try:
                    on_error(exc)
                except Exception:  # noqa: BLE001
                    pass
            self.put(key, _CachedError(exc), ttl_s=self.error_ttl_s)
            raise
        self.put(key, value)
        return value

    def stats(self) -> dict[str, Any]:
        with self._lock:
            entries = len(self._data)
        total = self.hits + self.misses
        return {"entries": entries, "ttl_s": self.ttl_s, "error_ttl_s": self.error_ttl_s,
                "hits": self.hits, "misses": self.misses, "errors": self.errors,
                "stores": self.stores,
                "hit_rate": round(self.hits / total, 4) if total else 0.0}

    def reset(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = self.errors = self.stores = 0


class Singleton(Generic[T]):
    """通用懒加载单例（双检锁；``factory`` 只在首次调用时执行一次）。"""

    def __init__(self, name: str, factory: Callable[[], T]) -> None:
        self.name = name
        self.factory = factory
        self._value: T | None = None
        self._built = False
        self._lock = threading.Lock()
        self.builds = 0
        self.calls = 0

    def get(self) -> T:
        with self._lock:
            self.calls += 1
            if self._built:
                return self._value  # type: ignore[return-value]
            value = self.factory()
            self._value = value
            self._built = True
            self.builds += 1
            return value

    @property
    def built(self) -> bool:
        with self._lock:
            return self._built

    def peek(self) -> T | None:
        with self._lock:
            return self._value if self._built else None

    def reset(self) -> None:
        with self._lock:
            self._value = None
            self._built = False

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"name": self.name, "built": self._built, "builds": self.builds, "calls": self.calls}


# ====================================================================== #
# 进程级注册表
# ====================================================================== #
_PROBES: dict[str, ProbeCache] = {}
_SINGLETONS: dict[str, Singleton[Any]] = {}
_REG_LOCK = threading.Lock()


def get_probe_cache(name: str, cfg: PerfConfig | None = None) -> ProbeCache:
    """按名字取（或建）一个探测缓存。"""
    config = cfg or get_perf_config()
    with _REG_LOCK:
        cache = _PROBES.get(name)
        if cache is None:
            cache = ProbeCache(ttl_s=config.probe_cache_ttl_s)
            _PROBES[name] = cache
        return cache


def get_singleton(name: str, factory: Callable[[], T]) -> Singleton[T]:
    """按名字取（或建）一个懒加载单例。"""
    with _REG_LOCK:
        item = _SINGLETONS.get(name)
        if item is None:
            item = Singleton(name, factory)
            _SINGLETONS[name] = item
        return item  # type: ignore[return-value]


def registry_stats(cfg: PerfConfig | None = None) -> dict[str, Any]:
    """诊断快照（供 ``/api/v1/perf``）。"""
    config = cfg or get_perf_config()
    with _REG_LOCK:
        probes = {k: v.stats() for k, v in _PROBES.items()}
        singles = {k: v.stats() for k, v in _SINGLETONS.items()}
    return {"perf_enabled": config.enabled, "probe_cache_enabled": config.probe_cache_enabled,
            "probe_caches": probes, "singletons": singles}


def reset_all() -> None:
    """重置全部缓存与单例（测试用）。"""
    with _REG_LOCK:
        probes = list(_PROBES.values())
        singles = list(_SINGLETONS.values())
    for p in probes:
        p.reset()
    for s in singles:
        s.reset()


__all__ = ["WORK_ORDER", "ProbeCache", "Singleton", "get_probe_cache", "get_singleton",
           "registry_stats", "reset_all"]
