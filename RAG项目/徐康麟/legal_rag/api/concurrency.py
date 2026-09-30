# -*- coding: utf-8 -*-
"""并发原语：全局信号量限流、线程池卸载、每会话写锁。

为什么必须卸载到线程
--------------------
``pymilvus`` 的检索/写入、BM25 的建索引、Ollama 的 HTTP 调用**全是同步阻塞**。
如果直接在 ``async def`` 端点里调用，就会把事件循环堵死 —— 表现为「并发请求互相等待、
TTFT 长到离谱、连 /health 都不响应」。因此所有阻塞调用都经
:func:`run_blocking` 交给 anyio 的线程池，池大小取 ``config.concurrency.thread_pool_size``。

限流策略（已写进 docs/API.md）
------------------------------
* 全局 ``asyncio.Semaphore(config.concurrency.max_concurrent_requests)`` —— 同一时刻
  最多 N 个「重请求」（chat / ingest / upload）在跑，其余**排队**；
* 排队超过 :data:`QUEUE_WAIT_TIMEOUT_SECONDS` 就返回 **429**（附 ``Retry-After``），
  而不是无限堆积 —— CPU 上单请求 ≈25s，堆积只会让所有人一起超时（雪崩）；
* 排队耗时记进 ``queue_wait_seconds``，被拒次数记进 ``requests_rejected_total{reason}``；
* 每个请求在跑的时候 ``inflight_requests`` 会 +1/-1（配合 Prometheus 看瞬时并发）。
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, TypeVar

import anyio

from .. import metrics as M

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: 排队等待上限（秒）。超过即 429 —— 见模块 docstring 的雪崩说明。
#: 取值依据：CPU 上单请求 ≈25s，若前面已有 8 个在跑，最坏要等 8×25s=200s；
#: 但我们宁可快速拒绝也不让客户端干等，30s 是一个「客户端还能重试」的折中。
QUEUE_WAIT_TIMEOUT_SECONDS = 30.0

#: 429 响应的 Retry-After（秒）
RETRY_AFTER_SECONDS = 5

_REJECT_LABEL = "queue_full"


class QueueFull(RuntimeError):
    """排队超时（应由路由层转成 429）。"""

    def __init__(self, waited: float, waiters: int = 0) -> None:
        super().__init__(
            f"服务繁忙：排队 {waited:.1f}s 仍未获得执行槽位（当前排队 {waiters} 个请求）。"
            f"请稍后重试（Retry-After: {RETRY_AFTER_SECONDS}s）。")
        self.waited = waited
        self.waiters = waiters


class RequestLimiter:
    """全局并发闸门（信号量）+ inflight/排队指标。"""

    def __init__(self, max_concurrent: int, *,
                 queue_timeout: float | None = None,
                 name: str = "requests") -> None:
        self.max_concurrent = max(int(max_concurrent), 1)
        self.queue_timeout = (QUEUE_WAIT_TIMEOUT_SECONDS if queue_timeout is None
                              else float(queue_timeout))
        self.name = name
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: Any = None
        self._waiters = 0
        self.acquired_total = 0
        self.rejected_total = 0

    # 信号量必须在**运行中的事件循环**里创建，且换循环时要重建
    # （asyncio 原语绑定创建时的 loop；测试/多次 asyncio.run 会换 loop）
    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(self.max_concurrent)
            self._sem_loop = loop
        return self._sem

    @property
    def waiters(self) -> int:
        """当前排队中的请求数（仅本进程统计）。"""
        return self._waiters

    def snapshot(self) -> dict:
        """给 /health 的自描述（不含任何阻塞调用）。"""
        return {
            "max_concurrent": self.max_concurrent,
            "queue_timeout_seconds": self.queue_timeout,
            "waiters": self._waiters,
            "acquired_total": self.acquired_total,
            "rejected_total": self.rejected_total,
        }

    @asynccontextmanager
    async def slot(self, kind: str = "chat") -> AsyncIterator[None]:
        """获取一个执行槽位；排队超时抛 :class:`QueueFull`。"""
        sem = self._semaphore()
        started = time.perf_counter()
        self._waiters += 1
        acquired = False
        try:
            if self.queue_timeout > 0:
                try:
                    await asyncio.wait_for(sem.acquire(), timeout=self.queue_timeout)
                    acquired = True
                except asyncio.TimeoutError as exc:
                    self.rejected_total += 1
                    M.counter("requests_rejected_total",
                              "被限流拒绝的请求数（reason=queue_full）", "count").inc(
                        reason=_REJECT_LABEL)
                    raise QueueFull(time.perf_counter() - started, self._waiters) from exc
            else:  # pragma: no cover - 仅当显式关闭排队超时时走到
                await sem.acquire()
                acquired = True
        finally:
            self._waiters = max(self._waiters - 1, 0)
            M.observe("queue_wait_seconds", time.perf_counter() - started, kind=kind)
            if not acquired:
                logger.warning("限流拒绝：kind=%s 排队 %.2fs（max_concurrent=%d）",
                               kind, time.perf_counter() - started, self.max_concurrent)

        self.acquired_total += 1
        inflight = M.gauge("inflight_requests")
        inflight.inc(kind=kind)
        try:
            logger.debug("获得执行槽位：kind=%s（inflight=%s）",
                         kind, inflight.value(kind=kind))
            yield
        finally:
            inflight.dec(kind=kind)
            sem.release()


class SessionLockRegistry:
    """按 ``(user_id, role_id, session_id)`` 提供写锁，保证同一会话的历史追加串行。

    为什么需要：并发两个请求打到同一会话时，``读历史 -> 生成 -> 追加历史`` 会交错，
    导致历史错乱（后写的覆盖先写的、或同一轮被记两次）。同一会话串行、不同会话并行。
    """

    def __init__(self, max_entries: int = 4096) -> None:
        self.max_entries = max(int(max_entries), 1)
        self._guard: asyncio.Lock | None = None
        self._guard_loop: Any = None
        self._locks: dict[str, list[Any]] = {}      # key -> [asyncio.Lock, 引用计数]
        self.waits = 0
        self.timeouts = 0

    def _guard_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._guard is None or self._guard_loop is not loop:
            # 换了事件循环：旧锁表绑定在死循环上，整体重置
            self._guard = asyncio.Lock()
            self._guard_loop = loop
            self._locks = {}
        return self._guard

    @staticmethod
    def key_of(user_id: str, role_id: str, session_id: str) -> str:
        return f"{user_id}\x1f{role_id}\x1f{session_id}"

    @property
    def size(self) -> int:
        return len(self._locks)

    def snapshot(self) -> dict:
        return {"active_locks": len(self._locks), "waits": self.waits,
                "timeouts": self.timeouts, "max_entries": self.max_entries}

    @asynccontextmanager
    async def hold(self, key: str, *, timeout: float | None = None) -> AsyncIterator[None]:
        """持有该会话的写锁（可选等待上限，超时抛 :class:`TimeoutError`）。"""
        async with self._guard_lock():
            entry = self._locks.get(key)
            if entry is None:
                if len(self._locks) >= self.max_entries:
                    # 极端情况下不无界增长：直接放弃加锁（退化为无锁，仅记录）
                    logger.warning("会话锁表已满（%d），本次跳过加锁：key=%s…",
                                   self.max_entries, key[:24])
                    yield
                    return
                entry = [asyncio.Lock(), 0]
                self._locks[key] = entry
            entry[1] += 1

        lock: asyncio.Lock = entry[0]
        waited = lock.locked()
        if waited:
            self.waits += 1
        try:
            if timeout and timeout > 0:
                try:
                    await asyncio.wait_for(lock.acquire(), timeout=timeout)
                except asyncio.TimeoutError:
                    self.timeouts += 1
                    raise
            else:
                await lock.acquire()
            try:
                yield
            finally:
                lock.release()
        finally:
            async with self._guard_lock():
                entry[1] -= 1
                if entry[1] <= 0 and not lock.locked():
                    self._locks.pop(key, None)


async def run_blocking(func: Callable[..., T], *args: Any,
                       limiter: anyio.CapacityLimiter | None = None,
                       **kwargs: Any) -> T:
    """把同步阻塞函数放到 anyio 线程池执行（大小 = thread_pool_size）。

    ``anyio.to_thread.run_sync`` 会**复制当前 contextvars**（含 request_id），
    因此线程里的日志仍然带同一个 request_id，便于 grep 串联。
    """
    if kwargs:
        func = functools.partial(func, **kwargs)      # type: ignore[assignment]
    return await anyio.to_thread.run_sync(func, *args, limiter=limiter)


def make_thread_limiter(size: int) -> anyio.CapacityLimiter:
    """线程池容量上限（在事件循环里创建）。"""
    return anyio.CapacityLimiter(max(int(size), 1))


__all__ = [
    "QUEUE_WAIT_TIMEOUT_SECONDS", "RETRY_AFTER_SECONDS",
    "QueueFull", "RequestLimiter", "SessionLockRegistry",
    "run_blocking", "make_thread_limiter",
]
