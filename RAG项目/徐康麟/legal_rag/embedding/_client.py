# -*- coding: utf-8 -*-
"""Ollama HTTP 客户端（urllib3 会话复用 + 超时 + 指数退避重试）。

为什么要单独一层
----------------
1. **连接复用**：用 ``urllib3.PoolManager`` 做连接池，比每次 ``urllib.request``
   新建 TCP 连接更省；入库时会有成百上千次嵌入调用，这个开销不可忽略。
   同时 ``PoolManager`` 是线程安全的，可被 t4 的并发请求共用。
2. **超时可控**：``urllib3.Timeout(connect=..., read=...)`` —— 连接超时短
   （服务没起来就快速失败），读超时取 ``OLLAMA`` 配置里的 ``embed_timeout``
   （CPU 上 bge-m3 单批可能要几秒）。
3. **重试只针对「可重试」的失败**：连接被拒 / 读超时 / 5xx / 429 才退避重试；
   调用方自己导致的 4xx（400 参数错、404 模型不存在）立即失败，不浪费时间。
   POST 天然不可幂等安全，因此 **不依赖 urllib3 的自动 retry**，改为显式重试循环，
   每次重试都打 WARNING 日志（含第几次、退避多少秒、失败原因）。

约定：本模块**不吞异常**——重试耗尽后抛出 :class:`OllamaHTTPError`（带状态码、
URL、响应片段），由调用方决定是否降级。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..http_retry import RETRYABLE_STATUS as _RETRYABLE_STATUS
from ..http_retry import backoff_seconds, should_retry

logger = logging.getLogger(__name__)

#: 会被重试的 HTTP 状态码。
#:
#: **2026-09-29 迁移**：原先在本模块自定义，现与生成侧（``generate/openai_compat``）
#: 共用 :mod:`legal_rag.http_retry` 的同一份定义 —— 消除"同一类失败在嵌入侧可重试、
#: 在生成侧不可重试"的隐性不一致。保留本别名是为了不破坏既有 import。
RETRYABLE_STATUS = _RETRYABLE_STATUS


class OllamaHTTPError(RuntimeError):
    """Ollama HTTP 调用失败（重试耗尽后抛出），带定位所需的全部上下文。"""

    def __init__(
        self,
        message: str,
        *,
        url: str = "",
        status: int = 0,
        attempts: int = 1,
        body: str = "",
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status = status
        self.attempts = attempts
        self.body = body
        self.cause = cause

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "status": self.status,
            "attempts": self.attempts,
            "body": self.body[:500],
            "cause": f"{type(self.cause).__name__}: {self.cause}" if self.cause else None,
        }


class OllamaHTTPClient:
    """线程安全的 Ollama HTTP 客户端。

    :param base_url: 形如 ``http://127.0.0.1:11434``（**各机自用，不要硬编码 VM 地址**）；
    :param timeout: 读超时秒数（连接超时固定取 ``min(timeout, connect_timeout)``）；
    :param max_retries: 失败后的额外重试次数（总尝试次数 = max_retries + 1）；
    :param retry_backoff: 退避基数（秒），第 n 次重试等待 ``retry_backoff * 2**(n-1)``；
    :param api_key: 可选，填了才带 ``Authorization: Bearer``；
    :param connect_timeout: 连接超时秒数上限（默认 5s）。
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        api_key: str = "",
        connect_timeout: float = 5.0,
        pool_size: int = 16,
    ) -> None:
        self.base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = max(int(max_retries), 0)
        self.retry_backoff = max(float(retry_backoff), 0.0)
        self.api_key = api_key or ""
        self.connect_timeout = float(connect_timeout)
        self.pool_size = max(int(pool_size), 1)

        try:
            import urllib3  # type: ignore
        except ImportError as exc:  # pragma: no cover - requests/urllib3 是必装项
            raise RuntimeError(
                "缺少 urllib3，无法调用 Ollama。\n"
                "    .venv\\Scripts\\python.exe -m pip install -r requirements.txt"
            ) from exc

        self._urllib3 = urllib3
        #: 显式关闭自动重试：POST /api/embed 不保证幂等，重试策略由本类控制
        self._pool = urllib3.PoolManager(
            num_pools=4,
            maxsize=self.pool_size,
            retries=False,
            timeout=urllib3.Timeout(
                connect=min(self.connect_timeout, self.timeout or self.connect_timeout),
                read=self.timeout,
            ),
            # 允许对 localhost 之外的主机复用连接（VM 场景）
            block=False,
        )

    # ------------------------------------------------------------------
    # 底层
    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
        operation: str = "ollama.post",
    ) -> dict[str, Any]:
        """POST 一个 JSON 并解析 JSON 响应；失败按策略退避重试。

        失败一定抛 :class:`OllamaHTTPError`（**不吞异常**），并带：
        URL、HTTP 状态码、尝试次数、响应体片段、底层异常。
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        read_timeout = float(timeout if timeout is not None else self.timeout)
        attempts = self.max_retries + 1
        last_error: BaseException | None = None
        last_status = 0
        last_body = ""

        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                logger.debug("[BEFORE] %s url=%s 第 %d/%d 次，body=%dB",
                             operation, url, attempt, attempts, len(body))
                response = self._pool.request(
                    "POST", url, body=body, headers=self._headers(),
                    timeout=self._urllib3.Timeout(
                        connect=min(self.connect_timeout, read_timeout),
                        read=read_timeout,
                    ),
                    preload_content=True,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                raw = response.data.decode("utf-8", errors="replace")

                if response.status >= 400:
                    last_status = int(response.status)
                    last_body = raw
                    logger.warning(
                        "[AFTER] ✗ %s url=%s 第 %d/%d 次 HTTP %d，耗时 %.1fms，响应=%s",
                        operation, url, attempt, attempts, response.status, elapsed_ms,
                        raw[:300])
                    # 是否值得重试：交给公共策略判断（4xx 参数/模型错立即失败，省时间；
                    # 408/425/429/5xx 才退避重试）。判据与生成侧共用同一份实现。
                    if not should_retry(status=int(response.status)):
                        raise OllamaHTTPError(
                            f"{operation} 被拒绝（HTTP {response.status}）：{raw[:300]}\n"
                            f"提示：4xx 通常是请求参数或模型名不对，重试无意义。",
                            url=url, status=response.status, attempts=attempt, body=raw,
                        )
                    last_error = OllamaHTTPError(
                        f"HTTP {response.status}", url=url, status=response.status,
                        attempts=attempt, body=raw)
                else:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise OllamaHTTPError(
                            f"{operation} 返回的不是合法 JSON：{raw[:300]}",
                            url=url, status=response.status, attempts=attempt,
                            body=raw, cause=exc,
                        ) from exc
                    logger.debug("[AFTER] [OK] %s url=%s 第 %d/%d 次 HTTP %d，耗时 %.1fms",
                                 operation, url, attempt, attempts, response.status, elapsed_ms)
                    return data
            except OllamaHTTPError:
                raise
            except Exception as exc:  # noqa: BLE001 - 网络类异常都要退避重试
                last_error = exc
                logger.warning("[AFTER] [FAIL] %s url=%s 第 %d/%d 次调用异常：%s: %s",
                               operation, url, attempt, attempts, type(exc).__name__, exc)

            if attempt < attempts:
                # 退避公式也走公共策略（指数：backoff * 2**(n-1)），与迁移前逐位一致。
                wait = backoff_seconds(self, attempt)
                if wait > 0:
                    logger.info("%s 第 %d 次失败，%.2fs 后重试（共 %d 次机会）",
                                operation, attempt, wait, attempts)
                    time.sleep(wait)

        message = (
            f"{operation} 调用 Ollama 失败，已重试 {attempts} 次：{url}\n"
            f"最后一次失败：{type(last_error).__name__}: {last_error}\n"
            f"排查建议：\n"
            f"  1) 确认 Ollama 在本机可用：curl {self.base_url}/api/tags\n"
            f"  2) 确认 OLLAMA_BASE_URL 指向「本机」Ollama（Windows/VM 各用自己那份）\n"
            f"  3) CPU 场景模型冷启动很慢，可调大 EMBED_TIMEOUT（当前 {self.timeout:g}s）"
            f" 或拉长 OLLAMA_KEEP_ALIVE"
        )
        raise OllamaHTTPError(
            message, url=url, status=last_status, attempts=attempts,
            body=last_body, cause=last_error,
        ) from last_error

    def get_json(self, path: str, *, timeout: float = 10.0,
                 operation: str = "ollama.get") -> dict[str, Any]:
        """GET 一个 JSON（健康探针用；不重试）。"""
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self._pool.request(
                "GET", url, headers=self._headers(),
                timeout=self._urllib3.Timeout(connect=min(self.connect_timeout, timeout),
                                              read=timeout),
                preload_content=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise OllamaHTTPError(
                f"{operation} 失败：{type(exc).__name__}: {exc}", url=url, cause=exc) from exc
        raw = response.data.decode("utf-8", errors="replace")
        if response.status >= 400:
            raise OllamaHTTPError(
                f"{operation} 失败（HTTP {response.status}）：{raw[:200]}",
                url=url, status=int(response.status), body=raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaHTTPError(f"{operation} 返回的不是合法 JSON：{raw[:200]}",
                                  url=url, body=raw, cause=exc) from exc

    def close(self) -> None:
        try:
            self._pool.clear()
        except Exception:  # pragma: no cover
            logger.debug("清理 Ollama 连接池失败", exc_info=True)
