# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

用例组 04：**HTTP keep-alive 连接复用**（修复项 F1）。

对应工单原文泄漏点二「数据库/向量库连接池泄漏」在本仓库的真实形态：
基座所有外部 HTTP（``llm_client._http_json`` / ``_stream_ollama`` / ``_stream_openai`` /
``embedder._http_json``）都用 ``urllib.request.urlopen`` —— **每次调用新建 TCP、读完即关**。
一次问答至少 2 次 Ollama 往返（嵌入 + 生成），20 并发下就是几十次/秒的建连断连。

本组用例起一个**本机 stdlib HTTP 服务器并统计 TCP 连接数**，直接量化「复用 vs 不复用」：

* 打开优化：N 次请求只建少量连接，``reused_total > created_total``；
* 关闭优化：N 次请求 ≈ N 次建连（对照组）。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import reset_singletons


class _CountingHandler(BaseHTTPRequestHandler):
    """每接受一条 TCP 连接就记一次；支持 keep-alive（HTTP/1.1）。

    ⚠️ 实测踩坑：``HTTP/1.1`` + 默认的 ``block_on_close=True`` 会让
    ``server_close()`` **join 那些正阻塞在 keep-alive 读上的 handler 线程** ——
    测试进程直接挂死（pytest 跑 10 分钟无输出）。
    必须同时打开 ``daemon_threads`` 并关掉 ``block_on_close``。
    """

    protocol_version = "HTTP/1.1"
    daemon_threads = True
    block_on_close = False
    connections = 0
    requests = 0
    lock = threading.Lock()

    def log_message(self, *_a):   # 静音
        return

    def _respond(self) -> None:
        # ⚠️ 实测踩坑：HTTP/1.1 + keep-alive 下**必须读完请求体**，
        #    否则残留字节会被当成下一条请求行 ⇒ 服务端回 400
        #    （现象：同一个连接上的第二个请求开始就失败）。
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        with _CountingHandler.lock:
            _CountingHandler.requests += 1
        body = json.dumps({"ok": True, "n": _CountingHandler.requests}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _respond
    do_POST = _respond

    def setup(self) -> None:
        super().setup()
        with _CountingHandler.lock:
            _CountingHandler.connections += 1


@pytest.fixture()
def counting_server():
    """本机计数 HTTP 服务器（每个用例独立计数）。"""
    _CountingHandler.connections = 0
    _CountingHandler.requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CountingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    yield f"http://{host}:{port}"
    # 先关掉连接池里的空闲长连接，再停服务端，避免 handler 线程阻塞在 keep-alive 读上
    try:
        from service.http_pool import get_http_pool

        get_http_pool().reset()
    except Exception:  # noqa: BLE001
        pass
    server.shutdown()
    server.server_close()


def test_连接池复用连接(perf_env, counting_server):
    """**核心断言**：20 次请求，新建连接远小于 20，且复用次数 > 新建次数。"""
    reset_singletons()
    from service.http_pool import get_http_pool

    pool = get_http_pool(refresh=True)
    assert pool.enabled is True
    for _ in range(20):
        data = pool.request_json(f"{counting_server}/api/tags")
        assert data["ok"] is True

    stats = pool.stats()
    assert stats["created_total"] < 20, f"仍在逐请求建连：created={stats['created_total']}"
    assert stats["reused_total"] >= 10, f"复用次数不足：{stats['reused_total']}"
    assert stats["reuse_ratio"] > 0.5, f"复用率过低：{stats['reuse_ratio']}"
    # 服务端看到的 TCP 连接数应当很少（keep-alive 生效）
    assert _CountingHandler.connections <= 3, f"服务端接受了 {_CountingHandler.connections} 条连接"
    assert _CountingHandler.requests == 20


def test_基座口径逐请求建连(base_env, counting_server):
    """对照组：``PERF_ENABLED=0`` 时走 urlopen，服务端连接数 ≈ 请求数。"""
    reset_singletons()
    from service.http_pool import get_http_pool

    pool = get_http_pool(refresh=True)
    assert pool.enabled is False
    for _ in range(10):
        with urllib.request.urlopen(f"{counting_server}/api/tags", timeout=5) as resp:
            resp.read()
    assert _CountingHandler.connections >= 10, (
        f"基座口径下服务端只看到 {_CountingHandler.connections} 条连接（应 ≈ 请求数）")


def test_连接池按host分桶(perf_env, counting_server):
    reset_singletons()
    from service.http_pool import get_http_pool

    pool = get_http_pool(refresh=True)
    pool.request_json(f"{counting_server}/api/tags")
    pool.request_json(f"{counting_server}/api/embed")
    pool.request_json(f"{counting_server}/api/generate")
    stats = pool.stats()
    assert len(stats["hosts"]) == 1, f"应只有一个桶：{list(stats['hosts'])}"
    assert stats["created_total"] <= 2


def test_流式响应读完后归还连接(perf_env, counting_server):
    """流式（NDJSON）读取必须**读完**才归还，否则下一个请求会读到上一个的残留。"""
    reset_singletons()
    from service.http_pool import get_http_pool

    pool = get_http_pool(refresh=True)
    for _ in range(6):
        with pool.stream_lines(f"{counting_server}/api/generate", {"stream": True}) as resp:
            payload = resp.read()
            assert b"ok" in payload
    stats = pool.stats()
    assert stats["created_total"] <= 3, f"流式路径未复用：created={stats['created_total']}"
    assert stats["reused_total"] >= 3


def test_连接池统计口径完整(perf_env, counting_server):
    reset_singletons()
    from service.http_pool import get_http_pool

    pool = get_http_pool(refresh=True)
    pool.request_json(f"{counting_server}/api/tags")
    stats = pool.stats()
    for key in ("enabled", "max_idle", "hosts", "created_total", "reused_total", "reuse_ratio"):
        assert key in stats
    host_stats = next(iter(stats["hosts"].values()))
    for key in ("created", "reused", "discarded", "retries", "idle"):
        assert key in host_stats


def test_连接池关闭幂等(perf_env, counting_server):
    reset_singletons()
    from service.http_pool import get_http_pool

    pool = get_http_pool(refresh=True)
    pool.request_json(f"{counting_server}/api/tags")
    pool.reset()
    pool.reset()
    assert sum(h["idle"] for h in pool.stats()["hosts"].values()) == 0


def test_ollama请求统一带keep_alive(perf_env, monkeypatch):
    """F7：开启优化后，Ollama 请求的 ``keep_alive`` 必须来自配置（不再是写死的 30m）。"""
    reset_singletons()
    monkeypatch.setenv("PERF_OLLAMA_KEEP_ALIVE", "5m")
    from service import perf_config

    perf_config.get_perf_config(refresh=True)
    from app.core import perf_hook

    monkeypatch.setattr(perf_hook, "_perf", perf_config.get_perf_config())
    monkeypatch.setattr(perf_hook, "_perf_loaded", True)
    assert perf_hook.keep_alive() == "5m"


def test_关闭优化时keep_alive保持基座值(base_env, monkeypatch):
    from app.core import perf_hook

    monkeypatch.setattr(perf_hook, "_perf", None)
    monkeypatch.setattr(perf_hook, "_perf_loaded", True)
    assert perf_hook.keep_alive() == "30m", "基座口径必须保留 30m（逐位一致）"
