# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

F1 修复：标准库 HTTP keep-alive 连接池 —— ``研发/service/http_pool.py``。

## 为什么需要它（问题定位）

基座的全部外部 HTTP 调用都用 ``urllib.request.urlopen``：

* ``app/core/llm_client.py:92``   ``_http_json``（探测/预热）
* ``app/core/llm_client.py:239-245`` ``_stream_ollama``（**每次生成**）
* ``app/core/llm_client.py:278-283`` ``_stream_openai``
* ``app/core/embedder.py:84``     ``_http_json``（**每次嵌入**）

``urlopen`` **不复用连接**：每个调用都要走一次 socket 建连（三次握手），
响应体读完即关闭。一次问答至少 2 次 Ollama 往返（embed + generate），
20 并发下即 40 次/秒的建连/断连，产生大量 TIME_WAIT 并抬高尾延迟。

## 实现

* 按 ``(scheme, host, port)`` 分桶的 ``http.client.HTTPConnection`` 栈；
* ``lease()`` 上下文管理器：借出 → 用 → 归还（异常时丢弃该连接）；
* 归还时若连接已断（``sock is None``）则关闭丢弃，超过 ``max_idle`` 也丢弃；
* 对端提前关闭（``RemoteDisconnected`` / ``BrokenPipe`` / ``ConnectionReset``）
  **重建一次并重试**，与 ``session_store.MiniRedis.do()`` 同款策略；
* 全部操作线程安全（每桶一把锁），**零第三方依赖**。

## 开关

``PERF_ENABLED=1`` 且 ``PERF_HTTP_POOL_ENABLED=1`` 时才启用；否则
``request()`` 直接退回 ``urllib.request.urlopen``，行为与基座一致。
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import urlparse

from .perf_config import PerfConfig, get_perf_config

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

#: 判定「连接已被对端关闭」的异常类型
_RETRYABLE = (
    http.client.RemoteDisconnected,
    http.client.BadStatusLine,
    http.client.CannotSendRequest,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
    socket.timeout,
)


class _Bucket:
    """单个 ``host:port`` 的连接栈。"""

    def __init__(self, host: str, port: int, *, max_idle: int, timeout_s: float) -> None:
        self.host = host
        self.port = port
        self.max_idle = max(0, int(max_idle))
        self.timeout_s = float(timeout_s)
        self._idle: list[http.client.HTTPConnection] = []
        self._lock = threading.Lock()
        self.created = 0
        self.reused = 0
        self.discarded = 0
        self.retries = 0

    # -- 借还 ---------------------------------------------------------- #
    def _new(self) -> http.client.HTTPConnection:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        self.created += 1
        return conn

    def checkout(self) -> tuple[http.client.HTTPConnection, bool]:
        """返回 ``(连接, 是否复用)``。"""
        with self._lock:
            while self._idle:
                conn = self._idle.pop()
                if getattr(conn, "sock", None) is not None:
                    self.reused += 1
                    return conn, True
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            return self._new(), False

    def checkin(self, conn: http.client.HTTPConnection) -> None:
        alive = getattr(conn, "sock", None) is not None
        with self._lock:
            if alive and len(self._idle) < self.max_idle:
                self._idle.append(conn)
                return
        self.discarded += 1
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    def drop(self, conn: http.client.HTTPConnection) -> None:
        with self._lock:
            self.discarded += 1
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    def stats(self) -> dict[str, Any]:
        with self._lock:
            idle = len(self._idle)
        return {"created": self.created, "reused": self.reused,
                "discarded": self.discarded, "retries": self.retries, "idle": idle}


class HttpKeepAlive:
    """进程级 HTTP 连接池（单例；见 :func:`get_http_pool`）。"""

    def __init__(self, cfg: PerfConfig | None = None) -> None:
        self.cfg = cfg or get_perf_config()
        self._buckets: dict[tuple[str, str, int], _Bucket] = {}
        self._lock = threading.Lock()

    # -- 桶 ------------------------------------------------------------ #
    def _bucket(self, url: str) -> _Bucket:
        parsed = urlparse(url)
        scheme = parsed.scheme or "http"
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port or (443 if scheme == "https" else 80))
        key = (scheme, host, port)
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(host, port, max_idle=self.cfg.http_pool_max_idle,
                                 timeout_s=self.cfg.http_pool_timeout_s)
                self._buckets[key] = bucket
            return bucket

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.http_pool_enabled)

    # -- 核心：一次请求 ------------------------------------------------- #
    def request_bytes(self, url: str, *, method: str = "GET", body: bytes | None = None,
                      headers: dict[str, str] | None = None,
                      timeout: float | None = None) -> bytes:
        """发一次请求并**读完响应体**（读完才能安全归还连接）。"""
        if not self.enabled:
            req = urllib.request.Request(url, data=body, method=method,
                                         headers=headers or {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout or self.cfg.http_pool_timeout_s) as resp:
                return resp.read()

        parsed = urlparse(url)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        bucket = self._bucket(url)
        attempts = max(1, int(self.cfg.http_pool_retries) + 1)
        last: Exception | None = None
        for attempt in range(attempts):
            conn, _reused = bucket.checkout()
            try:
                conn.request(method, path, body=body, headers=headers or {"Content-Type": "application/json"})
                resp = conn.getresponse()
                if resp.status >= 400:
                    payload = resp.read()
                    # 4xx/5xx 也要读完（否则连接不可复用），然后让调用方看到错误
                    raise _HttpStatusError(resp.status, payload, dict(resp.getheaders()))
                data = resp.read()
                bucket.checkin(conn)
                return data
            except _HttpStatusError:
                bucket.drop(conn)
                raise
            except _RETRYABLE as exc:
                last = exc
                bucket.drop(conn)
                if attempt + 1 < attempts:
                    with bucket._lock:  # noqa: SLF001
                        bucket.retries += 1
                    continue
            except (OSError, http.client.HTTPException) as exc:
                last = exc
                bucket.drop(conn)
        raise last if last is not None else OSError("HTTP 请求失败（无异常信息）")

    def request_json(self, url: str, payload: dict[str, Any] | None = None, *,
                     timeout: float | None = None, method: str | None = None) -> Any:
        """JSON 请求/响应（``payload=None`` 时为 GET）。"""
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        body = self.request_bytes(
            url, method=(method or ("POST" if data is not None else "GET")), body=data,
            headers={"Content-Type": "application/json"}, timeout=timeout)
        return json.loads(body.decode("utf-8", errors="replace"))

    @contextmanager
    def stream_lines(self, url: str, payload: dict[str, Any], *,
                     timeout: float | None = None) -> Iterator[Iterator[bytes]]:
        """流式 NDJSON：借出连接 → 逐行 yield → **finally 必须归还/丢弃**。

        这是 Ollama ``/api/generate`` 的用法：响应是长连接流，
        只有读完后才能把连接还回池子里（否则下一个请求会读到上一个的残留）。
        """
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if not self.enabled:
            req = urllib.request.Request(url, data=body, method="POST", headers=headers)
            with urllib.request.urlopen(req, timeout=timeout or self.cfg.http_pool_timeout_s) as resp:
                yield resp
            return

        parsed = urlparse(url)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        bucket = self._bucket(url)
        conn, _ = bucket.checkout()
        ok = False
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            if resp.status >= 400:
                payload_bytes = resp.read()
                raise _HttpStatusError(resp.status, payload_bytes, dict(resp.getheaders()))
            yield resp
            try:
                resp.read()          # 读完剩余字节，保证连接可复用
            except Exception:        # noqa: BLE001
                pass
            ok = True
        except _HttpStatusError:
            bucket.drop(conn)
            raise
        except Exception:
            bucket.drop(conn)
            raise
        finally:
            if ok:
                bucket.checkin(conn)
            elif getattr(conn, "sock", None) is None:
                bucket.drop(conn)
            else:
                bucket.drop(conn)

    # -- 统计 ---------------------------------------------------------- #
    def stats(self) -> dict[str, Any]:
        with self._lock:
            items = {f"{k[1]}:{k[2]}": v.stats() for k, v in self._buckets.items()}
        total_created = sum(v["created"] for v in items.values())
        total_reused = sum(v["reused"] for v in items.values())
        return {"enabled": self.enabled, "max_idle": self.cfg.http_pool_max_idle,
                "hosts": items, "created_total": total_created, "reused_total": total_reused,
                "reuse_ratio": round(total_reused / max(total_created + total_reused, 1), 4)}

    def reset(self) -> None:
        """关闭全部空闲连接（测试/退出用）。"""
        with self._lock:
            buckets = list(self._buckets.values())
        for bucket in buckets:
            with bucket._lock:  # noqa: SLF001
                idle, bucket._idle = bucket._idle, []  # noqa: SLF001
            for conn in idle:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass


class _HttpStatusError(urllib.error.HTTPError):
    """HTTP 4xx/5xx（与 ``urllib.error.HTTPError`` 同族，便于上层统一 except）。"""

    def __init__(self, status: int, payload: bytes, headers: dict[str, str]) -> None:
        self._payload = payload or b""
        super().__init__(url="", code=int(status), msg=f"HTTP {status}", hdrs=None, fp=None)  # type: ignore[arg-type]
        self.headers = headers  # type: ignore[assignment]

    def read(self, *args: Any, **kwargs: Any) -> bytes:  # noqa: D102
        return self._payload


_POOL: HttpKeepAlive | None = None
_POOL_LOCK = threading.Lock()


def get_http_pool(cfg: PerfConfig | None = None, *, refresh: bool = False) -> HttpKeepAlive:
    """进程级单例连接池。"""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None or refresh:
            _POOL = HttpKeepAlive(cfg)
        return _POOL


def now_ms() -> float:
    """便捷：毫秒时间戳（供调用方计时）。"""
    return time.perf_counter() * 1000.0


__all__ = ["WORK_ORDER", "HttpKeepAlive", "get_http_pool", "now_ms"]
