# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

用例组 01：**重量级组件进程内单例 + 后端探测结果缓存**（修复项 F2）。

对应工单原文「瓶颈一：文档解析与模型加载：每个请求可能都在独立初始化重量级组件」——
本仓库中该缺陷的**真实等价形态**是 ``app/core/embedder.py::embed_texts``
每次调用都执行 ``resolve_embedding_backend(probe=True)``，即每一问都真探一次
``GET /api/tags``。本组用例验证：

* ``Singleton`` 只构造一次（并发下也是）；
* ``ProbeCache`` 在 TTL 内命中、过期后重探；
* ``resolve_embedding_backend`` 打开优化后**探测次数与请求数解耦**（N 次调用只探 1 次）。
"""

from __future__ import annotations

import threading
import time

import pytest

from conftest import reset_singletons


def test_单例只构造一次(perf_env):
    """``Singleton.get()`` 在多线程并发下也只调用 factory 一次。"""
    reset_singletons()
    from service.singletons import Singleton

    calls = {"n": 0}
    lock = threading.Lock()

    def factory():
        with lock:
            calls["n"] += 1
        time.sleep(0.02)          # 放大竞态窗口
        return object()

    single = Singleton("测试用单例", factory)
    results = []
    threads = [threading.Thread(target=lambda: results.append(single.get())) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1, f"factory 被调用了 {calls['n']} 次（应为 1）"
    assert single.builds == 1
    assert len({id(r) for r in results}) == 1, "多线程拿到了不同实例"
    assert single.stats()["calls"] >= 16


def test_注册表按名字复用同一实例(perf_env):
    """``get_singleton`` / ``get_probe_cache`` 同名必须返回同一个对象。"""
    reset_singletons()
    from service.singletons import get_probe_cache, get_singleton

    a = get_singleton("索引bundle", lambda: {"v": 1})
    b = get_singleton("索引bundle", lambda: {"v": 2})
    assert a is b
    assert a.get() == {"v": 1}

    c = get_probe_cache("后端探测")
    d = get_probe_cache("后端探测")
    assert c is d


def test_探测缓存在TTL内命中_过期后重探(perf_env):
    """``ProbeCache``：TTL 内命中、过期后重新调用 factory。"""
    from service.singletons import ProbeCache

    cache = ProbeCache(ttl_s=0.3)
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return f"值{calls['n']}"

    assert cache.call("k", factory) == "值1"
    assert cache.call("k", factory) == "值1"      # 命中，不再调 factory
    assert calls["n"] == 1
    time.sleep(0.35)
    assert cache.call("k", factory) == "值2"      # 过期 → 重探
    assert calls["n"] == 2
    stats = cache.stats()
    assert stats["hits"] >= 1 and stats["misses"] >= 2


def test_探测失败按短TTL缓存不重复打超时(perf_env):
    """探测抛异常时也应缓存（短 TTL），避免故障期间每个请求都等一次超时。"""
    from service.singletons import ProbeCache

    cache = ProbeCache(ttl_s=10.0, error_ttl_s=0.2)
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise RuntimeError("Ollama 不可达")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            cache.call("k", boom)
    assert calls["n"] == 1, "失败结果没有被缓存"
    assert cache.stats()["errors"] == 1
    time.sleep(0.25)
    with pytest.raises(RuntimeError):
        cache.call("k", boom)
    assert calls["n"] == 2


def test_嵌入后端探测次数与请求数解耦(perf_env, monkeypatch):
    """**核心断言**：打开优化后，N 次嵌入后端解析只触发 1 次真实探测。"""
    reset_singletons()
    from app.core import embedder

    probes = {"n": 0}
    original = embedder.probe_backends

    def counting_probe(cfg=None, *, logger=None):
        probes["n"] += 1
        return original(cfg, logger=logger)

    monkeypatch.setattr(embedder, "probe_backends", counting_probe)

    from app.core.config import get_config

    cfg = get_config()
    for _ in range(5):
        embedder.resolve_embedding_backend(cfg, probe=True)
    assert probes["n"] == 1, f"5 次解析触发了 {probes['n']} 次探测（应为 1）"


def test_基座口径下探测每次都发生(base_env, monkeypatch):
    """对照组：``PERF_ENABLED=0`` 时行为与基座一致 —— 每次都真探（这是被修掉的问题）。"""
    reset_singletons()
    from app.core import embedder

    probes = {"n": 0}
    original = embedder.probe_backends

    def counting_probe(cfg=None, *, logger=None):
        probes["n"] += 1
        return original(cfg, logger=logger)

    monkeypatch.setattr(embedder, "probe_backends", counting_probe)

    from app.core.config import get_config

    cfg = get_config()
    for _ in range(5):
        embedder.resolve_embedding_backend(cfg, probe=True)
    assert probes["n"] == 5, f"基座口径应探测 5 次，实际 {probes['n']} 次"


def test_perf_hook在缺少service包时安全降级(monkeypatch):
    """业务内核不得强依赖部署层：``perf_hook`` 拿不到配置时必须返回安全默认值。"""
    from app.core import perf_hook

    monkeypatch.setattr(perf_hook, "_perf", None)
    monkeypatch.setattr(perf_hook, "_perf_loaded", True)
    assert perf_hook.enabled() is False
    assert perf_hook.pool() is None
    assert perf_hook.keep_alive() == perf_hook.BASE_KEEP_ALIVE == "30m"
    assert perf_hook.answer_cache_limit(512) == 512
    assert perf_hook.answer_cache_ttl(0.0) == 0.0
    assert perf_hook.cache_chunk_chars(0) == 0
