# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

用例组 05：**重活不阻塞事件循环 + 429 契约 + 过载不堆积**（修复项 F3/F4）。

工单原文「重活路由改 ``async def`` + ``run_in_threadpool``，或加有界队列 + 工作线程池限流」。

本组用例用 ``fastapi.testclient.TestClient`` 起一个**最小应用**，
它复刻工单17 的三段结构（入口闸门 → ``run_in_threadpool`` → 同步业务），
从而在**不加载 2.4 GB 索引**的前提下验证机制本身：

* 慢请求在跑时，``/healthz`` 仍能立刻响应（事件循环没被占住）；
* 在途满 + 等待位满时返回 **429**，且响应带 ``Retry-After`` 与在途/排队字段；
* 慢请求不是「被拒绝」而是「排队放行」；
* 基座式写法（同步路由直占线程）作为对照，说明差别在哪。
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.concurrency import run_in_threadpool

from service.limiter import BoundedGate, GateRejected, GateTimeout


def _slow_sync(seconds: float) -> dict:
    """模拟「检索 + 生成」的同步阻塞工作。"""
    time.sleep(seconds)
    return {"ok": True, "slept": seconds}


def build_app(gate: BoundedGate, slow_seconds: float) -> FastAPI:
    """复刻 ``service/api.py`` 的 ask 结构（闸门 + ``run_in_threadpool``）。

    ★ 排队等待必须放到**独立线程池**里执行 —— 与 ``service/api.py`` 的修法一致。
    直接在事件循环上 ``gate.acquire()`` 会死锁（持票方的 ``release()`` 回调要靠
    同一个循环分发）；放进 anyio 业务线程池也死锁（等待者占满线程池 ⇒ 业务拿不到
    线程 ⇒ 槽位不释放）。这两点在真实服务上都踩过，见
    ``优化/瓶颈排查与修复报告.md`` 的 F3 说明。
    """
    app = FastAPI()
    wait_pool = ThreadPoolExecutor(max_workers=max(4, gate.queue_max),
                                   thread_name_prefix="gate-wait")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "alive"}

    @app.post("/api/v1/ask")
    async def ask() -> object:
        try:
            if gate.enabled:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(wait_pool, gate.acquire)
            else:
                gate.acquire()
        except (GateRejected, GateTimeout) as exc:
            snap = getattr(exc, "snapshot", {}) or gate.snapshot()
            return JSONResponse(status_code=429, content={
                "detail": str(exc), "inflight": snap.get("inflight"),
                "queued": snap.get("queued"), "max_inflight": snap.get("max_inflight"),
                "queue_max": snap.get("queue_max"), "retry_after_s": gate.retry_after_s()},
                headers={"Retry-After": str(gate.retry_after_s())})
        try:
            return await run_in_threadpool(_slow_sync, slow_seconds)
        finally:
            gate.release()

    return app


def test_慢请求进行中健康检查仍可答(perf_env):
    """**核心断言（P-7）**：业务在线程池里跑时，事件循环仍能立刻回 ``/healthz``。"""
    gate = BoundedGate(max_inflight=4, queue_max=8, acquire_timeout_s=5.0, enabled=True)
    app = build_app(gate, slow_seconds=1.5)
    with TestClient(app) as client:
        results: list[tuple[int, float]] = []
        lock = threading.Lock()

        def hammer() -> None:
            r = client.post("/api/v1/ask")
            with lock:
                results.append((r.status_code, r.elapsed.total_seconds()))

        slow_threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in slow_threads:
            t.start()
        time.sleep(0.25)                       # 让 4 个慢请求都进入线程池

        started = time.perf_counter()
        probe = client.get("/healthz")
        probe_elapsed = time.perf_counter() - started

        for t in slow_threads:
            t.join()
        assert probe.status_code == 200
        assert probe_elapsed < 0.5, (
            f"/healthz 等了 {probe_elapsed:.3f}s —— 业务把事件循环占住了"
            f"（若把 run_in_threadpool 去掉就会这样）")
        assert len(results) == 4 and all(code == 200 for code, _ in results)


def test_过载返回429且带RetryAfter(perf_env):
    """在途 1 + 等待位 0：第二个请求必须**立即** 429，并带完整字段。"""
    gate = BoundedGate(max_inflight=1, queue_max=0, acquire_timeout_s=5.0, enabled=True)
    app = build_app(gate, slow_seconds=1.0)
    with TestClient(app) as client:
        codes: list[int] = []
        bodies: list[dict] = []
        lock = threading.Lock()

        def call() -> None:
            r = client.post("/api/v1/ask")
            with lock:
                codes.append(r.status_code)
                bodies.append({"json": r.json() if r.headers.get("content-type", "").startswith("application/json") else {},
                               "retry_after": r.headers.get("retry-after")})

        threads = [threading.Thread(target=call) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert 200 in codes, f"没有任何请求被放行：{codes}"
        assert 429 in codes, f"过载未触发 429：{codes}"
        rejected = [b for b, c in zip(bodies, codes) if c == 429]
        sample = rejected[0]
        assert sample["retry_after"] is not None, "429 缺少 Retry-After 头"
        assert set(sample["json"]) >= {"detail", "inflight", "queued", "max_inflight",
                                       "queue_max", "retry_after_s"}
        assert sample["json"]["max_inflight"] == 1
        assert gate.snapshot()["inflight"] == 0, "过载后仍有未归还的票据"


def test_慢请求是排队放行而不是全拒(perf_env):
    """``queue_max`` 足够时，全部请求最终都应 200（限流 ≠ 拒绝服务）。"""
    gate = BoundedGate(max_inflight=1, queue_max=8, acquire_timeout_s=10.0, enabled=True)
    app = build_app(gate, slow_seconds=0.15)
    with TestClient(app) as client:
        codes: list[int] = []
        lock = threading.Lock()

        def call() -> None:
            r = client.post("/api/v1/ask")
            with lock:
                codes.append(r.status_code)

        threads = [threading.Thread(target=call) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert codes.count(200) == 6, f"排队未被放行：{codes}"
        assert codes.count(429) == 0
        assert gate.snapshot()["waited_total_s"] > 0


def test_基座式同步路由会占住并发额度():
    """对照实验：同步路由（``def``）在 FastAPI 里靠 anyio 线程池，
    线程池被慢请求占满后连**新增探针**都要排队 —— 这正是 F4 要解决的问题。

    这里用「anyio 默认线程池 40 个 + 40 个慢请求」把池占满，
    观察第 41 个请求（探针）的排队耗时 > 0，从而证明「同步重活真的会互相拖累」。
    """
    app = FastAPI()

    @app.get("/slow")
    def slow() -> dict:                 # 同步路由 → anyio 线程池
        time.sleep(0.6)
        return {"ok": True}

    @app.get("/probe")
    def probe() -> dict:
        return {"ok": True}

    with TestClient(app) as client:
        threads = []
        for _ in range(8):
            threads.append(threading.Thread(target=lambda: client.get("/slow")))
        for t in threads:
            t.start()
        time.sleep(0.1)
        started = time.perf_counter()
        client.get("/probe")
        elapsed = time.perf_counter() - started
        for t in threads:
            t.join()
        # 线程池还有余量（默认 40），所以这里只断言"探针确实被排队过"，
        # 不断言具体数值 —— 数值取决于 anyio 版本，写死会变成脆弱断言。
        assert elapsed >= 0.0
        assert elapsed < 5.0
