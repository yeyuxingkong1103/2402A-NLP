# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

F3 修复：有界并发闸门（限流 + 快速失败 429）—— ``研发/service/limiter.py``。

## 为什么需要它（问题定位）

工单原文「瓶颈二：请求处理无队列/限流 …高并发时请求堆积，线程阻塞」在本仓库**完全成立且可核实**：

* ``service/config.py:119`` 定义了 ``max_concurrency: int = 4``；
* 但全仓库 ``grep -n "max_concurrency" 研发/`` **只命中定义那一行，零处引用**
  ⇒ 服务**没有任何入口限流**，20 个并发请求全部直接冲进业务逻辑；
* 而每个请求要跑「检索 + Ollama 生成」（实测 retrieve 197 ms + generate 1171 ms），
  在途请求只会越堆越多，表现为「平均响应从 2 s 陡增到 10 s+」——
  与工单原文描述的现象一致。

## 实现

:class:`BoundedGate` = ``threading.BoundedSemaphore(max_inflight)`` + 等待位计数：

* 在途 < ``max_inflight`` → 立即拿票；
* 否则：等待位 < ``queue_max`` → 排队等待（最多 ``acquire_timeout_s``）；
* 等待位也满 → **立即返回 False**，调用方回 ``429 + Retry-After``（**不堆积**）。

「排队」与「堆积」的区别：排队有**明确上限**且超限即拒，堆积没有上限。
这正是工单原文要求的「内存中的请求队列配合工作线程池进行限流」。
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .perf_config import PerfConfig, get_perf_config

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"


class GateRejected(Exception):
    """闸门拒绝（调用方应回 429）。"""

    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
        super().__init__(f"服务繁忙：在途 {snapshot['inflight']}/{snapshot['max_inflight']}，"
                         f"排队 {snapshot['queued']}/{snapshot['queue_max']}")


class GateTimeout(Exception):
    """排队超时（调用方应回 429 或 503）。"""

    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
        super().__init__(f"排队超时：在途 {snapshot['inflight']}，排队 {snapshot['queued']}，"
                         f"等待超过 {snapshot['waited_s']} s")


class BoundedGate:
    """有界并发闸门（线程安全）。"""

    def __init__(self, max_inflight: int = 8, queue_max: int = 32,
                 acquire_timeout_s: float = 30.0, *, enabled: bool = True) -> None:
        self.max_inflight = max(1, int(max_inflight))
        self.queue_max = max(0, int(queue_max))
        self.acquire_timeout_s = float(acquire_timeout_s)
        self.enabled = bool(enabled)
        self._sem = threading.BoundedSemaphore(self.max_inflight)
        # ⚠️ 必须是 RLock，不能是 Lock —— 实测踩坑（由 测试/test_03_限流与429.py 抓出）：
        #    ``acquire()`` 在 ``with self._lock:`` 内部抛 ``GateRejected``/``GateTimeout`` 时
        #    会调用 ``self.snapshot()``，而 snapshot 也要拿同一把锁 ⇒ **自死锁**，
        #    请求线程永久挂住（现象：客户端超时、服务端线程不回收）。
        #    该路径恰好只在「等待位满 / 排队超时」时进入 —— 也就是限流真正生效的那一刻。
        self._lock = threading.RLock()
        self._queued = 0
        self.inflight = 0
        self.inflight_peak = 0
        self.accepted = 0
        self.rejected = 0
        self.timeouts = 0
        self.waited_total_s = 0.0

    # -- 快照 ---------------------------------------------------------- #
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            queued = self._queued
            inflight = self.inflight
        return {"inflight": inflight, "max_inflight": self.max_inflight,
                "queued": queued, "queue_max": self.queue_max,
                "inflight_peak": self.inflight_peak, "accepted": self.accepted,
                "rejected": self.rejected, "timeouts": self.timeouts,
                "enabled": self.enabled,
                "waited_total_s": round(self.waited_total_s, 3)}

    def retry_after_s(self) -> int:
        """给客户端的 ``Retry-After``（保守估计：一个请求的处理时长量级）。"""
        return 2

    # -- 拿票/还票 ------------------------------------------------------ #
    def acquire(self, timeout_s: float | None = None) -> bool:
        """尝试进入。返回 ``True`` 表示拿到票（用完必须 :meth:`release`）。

        关闸（``enabled=False``）时永远返回 ``True`` 且不计数 —— 行为与基座一致。
        """
        if not self.enabled:
            return True
        budget = self.acquire_timeout_s if timeout_s is None else float(timeout_s)
        started = time.perf_counter()

        # ① 快路径：还有在途额度
        if self._sem.acquire(blocking=False):
            self._on_enter()
            return True

        # ② 慢路径：先占一个等待位（有界），占不到就直接拒
        with self._lock:
            if self._queued >= self.queue_max:
                self.rejected += 1
                raise GateRejected(self.snapshot())
            self._queued += 1
        try:
            got = self._sem.acquire(timeout=max(0.0, budget))
        finally:
            with self._lock:
                self._queued -= 1
        if not got:
            with self._lock:
                self.timeouts += 1
                snap = self.snapshot()
                snap["waited_s"] = round(time.perf_counter() - started, 3)
            raise GateTimeout(snap)
        with self._lock:
            self.waited_total_s += time.perf_counter() - started
        self._on_enter()
        return True

    def _on_enter(self) -> None:
        with self._lock:
            self.inflight += 1
            self.inflight_peak = max(self.inflight_peak, self.inflight)
            self.accepted += 1

    def release(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self.inflight > 0:
                self.inflight -= 1
        try:
            self._sem.release()
        except ValueError:  # pragma: no cover —— 多还票（调用方 bug），不崩
            pass

    # -- 上下文管理器 ---------------------------------------------------- #
    def __enter__(self) -> "BoundedGate":
        self.acquire()
        return self

    def __exit__(self, *_exc: Any) -> bool:
        self.release()
        return False


_GATE: BoundedGate | None = None
_GATE_LOCK = threading.Lock()


def get_gate(cfg: PerfConfig | None = None, *, refresh: bool = False) -> BoundedGate:
    """进程级单例闸门（参数取 ``PERF_*``）。"""
    global _GATE
    config = cfg or get_perf_config()
    with _GATE_LOCK:
        if _GATE is None or refresh:
            _GATE = BoundedGate(
                max_inflight=config.max_inflight, queue_max=config.queue_max,
                acquire_timeout_s=config.acquire_timeout_s, enabled=config.enabled,
            )
        return _GATE


__all__ = ["WORK_ORDER", "BoundedGate", "GateRejected", "GateTimeout", "get_gate"]
