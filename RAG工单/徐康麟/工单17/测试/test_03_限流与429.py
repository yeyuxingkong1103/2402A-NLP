# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

用例组 03：**有界并发闸门与 429 快速失败**（修复项 F3）。

对应工单原文「瓶颈二：请求处理无队列/限流 …高并发时请求堆积，线程阻塞」。
基座的 ``service/config.py`` 定义了 ``max_concurrency=4`` 却**零引用**（等于没有限流），
本组用例验证修复后的行为：

* 在途 ≤ ``PERF_MAX_INFLIGHT`` 时正常放行；
* 在途满 + 等待位满 ⇒ **立即** ``GateRejected``（上调方回 429），不是无限排队；
* 等待位未满时排队，超过 ``PERF_ACQUIRE_TIMEOUT_S`` ⇒ ``GateTimeout``；
* 关闸（``PERF_ENABLED=0``）时不计数、不拒绝 ＝ 基座行为；
* 上下文管理器异常路径也一定还票（不泄漏票据）。
"""

from __future__ import annotations

import threading
import time

import pytest

from conftest import reset_singletons


def test_在途未满时放行(perf_env):
    from service.limiter import BoundedGate

    gate = BoundedGate(max_inflight=2, queue_max=2, acquire_timeout_s=0.3, enabled=True)
    assert gate.acquire() is True
    assert gate.acquire() is True
    snap = gate.snapshot()
    assert snap["inflight"] == 2 and snap["max_inflight"] == 2
    gate.release()
    gate.release()
    assert gate.snapshot()["inflight"] == 0


def test_等待位满时立即拒绝(perf_env):
    """**核心断言**：在途满 + 等待位满 → 立即抛 ``GateRejected``（耗时远小于超时时间）。"""
    from service.limiter import BoundedGate, GateRejected

    gate = BoundedGate(max_inflight=1, queue_max=0, acquire_timeout_s=5.0, enabled=True)
    assert gate.acquire() is True
    started = time.perf_counter()
    with pytest.raises(GateRejected) as exc:
        gate.acquire()
    elapsed = time.perf_counter() - started
    assert elapsed < 0.2, f"应立即拒绝，实际等了 {elapsed:.3f}s"
    assert exc.value.snapshot["rejected"] >= 1
    gate.release()
    assert gate.snapshot()["rejected"] == 1


def test_超出等待位上限的并发只放行在途数(perf_env):
    """20 个线程抢 max_inflight=2 / queue_max=0 的闸门：只有 2 个能进，其余全部 429。"""
    from service.limiter import BoundedGate, GateRejected

    gate = BoundedGate(max_inflight=2, queue_max=0, acquire_timeout_s=1.0, enabled=True)
    entered: list[int] = []
    rejected: list[int] = []
    lock = threading.Lock()
    release_flag = threading.Event()

    def worker(wid: int) -> None:
        try:
            gate.acquire()
        except GateRejected:
            with lock:
                rejected.append(wid)
            return
        try:
            with lock:
                entered.append(wid)
            release_flag.wait(1.0)
        finally:
            gate.release()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    release_flag.set()
    for t in threads:
        t.join()

    assert len(entered) == 2, f"放行了 {len(entered)} 个（应为 2）"
    assert len(rejected) == 18, f"拒绝了 {len(rejected)} 个（应为 18）"
    assert gate.snapshot()["inflight"] == 0, "票据泄漏"


def test_排队超时抛GateTimeout(perf_env):
    from service.limiter import BoundedGate, GateTimeout

    gate = BoundedGate(max_inflight=1, queue_max=5, acquire_timeout_s=0.2, enabled=True)
    assert gate.acquire() is True
    started = time.perf_counter()
    with pytest.raises(GateTimeout):
        gate.acquire()
    elapsed = time.perf_counter() - started
    assert 0.15 <= elapsed < 1.5, f"超时等待时长异常：{elapsed:.3f}s"
    assert gate.snapshot()["timeouts"] == 1
    gate.release()


def test_排队后能拿到票(perf_env):
    """等待位够时，释放后排队者应立刻拿到票（限流 ≠ 拒绝服务）。"""
    from service.limiter import BoundedGate

    gate = BoundedGate(max_inflight=1, queue_max=4, acquire_timeout_s=3.0, enabled=True)
    assert gate.acquire() is True
    ok: list[bool] = []

    def waiter() -> None:
        gate.acquire()
        ok.append(True)

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.15)
    gate.release()
    t.join(timeout=3)
    assert ok == [True], "排队者未能拿到票"
    gate.release()


def test_关闸时行为与基座一致(base_env):
    """``PERF_ENABLED=0``：不计数、不拒绝、release 幂等 —— 完全等于没有限流。"""
    from service.limiter import BoundedGate

    gate = BoundedGate(max_inflight=1, queue_max=0, acquire_timeout_s=0.1, enabled=False)
    for _ in range(50):
        assert gate.acquire() is True
    snap = gate.snapshot()
    assert snap["enabled"] is False
    assert snap["inflight"] == 0 and snap["accepted"] == 0 and snap["rejected"] == 0
    gate.release()      # 未拿票就还票：不得抛
    gate.release()


def test_上下文管理器异常路径也还票(perf_env):
    from service.limiter import BoundedGate

    gate = BoundedGate(max_inflight=1, queue_max=0, acquire_timeout_s=0.5, enabled=True)
    with pytest.raises(ValueError):
        with gate:
            raise ValueError("业务异常")
    assert gate.snapshot()["inflight"] == 0, "异常路径泄漏了票据"
    assert gate.acquire() is True
    gate.release()


def test_闸门可观测指标齐全(perf_env):
    from service.limiter import BoundedGate

    gate = BoundedGate(max_inflight=3, queue_max=1, acquire_timeout_s=0.5, enabled=True)
    for _ in range(3):
        gate.acquire()
    snap = gate.snapshot()
    for key in ("inflight", "max_inflight", "queued", "queue_max", "inflight_peak",
                "accepted", "rejected", "timeouts", "enabled"):
        assert key in snap, f"快照缺少 {key}"
    assert snap["inflight_peak"] == 3
    assert gate.retry_after_s() >= 1
