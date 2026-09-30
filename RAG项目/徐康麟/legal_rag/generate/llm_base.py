# -*- coding: utf-8 -*-
"""大模型客户端抽象与工厂。

统一抽象让「本地 vLLM/SGLang 上的 Qwen3.8-27B」与「在线 API」可以互换：
切换后端只改配置，不改调用代码。

失败原因口径（t121 F2）
----------------------
同一套**可定位**的原因分类，客户端 / 路由 / 指标 / 日志 / API 响应五处共用：

===============  ====================================================
``missing_api_key``  没配 key（``OPENAI_API_KEY`` 之类）——**配置**问题，重试无意义
``connect_failed``   连不上端点（拒绝连接 / DNS / 网络不可达）
``http_5xx``         端点回了 5xx（服务端故障，可重试）
``http_4xx``         端点回了 4xx（参数/路径问题，重试一般无意义）
``timeout``          超时（发起后没在 ``timeout`` 内拿到响应）
``model_missing``    404：端点上没有这个模型名
``auth``             401/403：鉴权失败
``rate_limit``       429：被限流
``bad_request``      400：请求体被拒
``bad_response``     响应不是合法 JSON / 缺 choices
``unknown``          其它（**不猜**，如实写 unknown）
===============  ====================================================

与**旧口径**（``LLMStats.error_reason`` / ``llm_error_total{reason}``）的关系：
旧口径的取值（``connect`` / ``http`` / ``timeout`` / …）已被既有测试钉住，**保持不动**；
新口径只多不少（``connect`` -> ``connect_failed``、``http`` 按状态码细分 4xx/5xx），
用于 ``llm_failure_total``、``[LLM-FAIL]`` 日志与 API 的 503 响应。
"""
from __future__ import annotations

import json
import logging
import re
import socket
import time
import urllib.error
from abc import ABC, abstractmethod
from collections.abc import Iterator

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """大模型调用失败。"""

    def __init__(self, message: str, *, reason: str = "") -> None:
        super().__init__(message)
        #: 可分类的失败原因（见模块 docstring）。**只能关键字传**，
        #: 因此既有的 ``except LLMError`` 与位置参数用法全部继续有效。
        self.reason = reason


#: 新口径的原因名（t121 F2）
FAILURE_MISSING_API_KEY = "missing_api_key"
FAILURE_CONNECT_FAILED = "connect_failed"
FAILURE_HTTP_5XX = "http_5xx"
FAILURE_HTTP_4XX = "http_4xx"
FAILURE_TIMEOUT = "timeout"

#: 旧口径 -> 新口径（只映射**名字**，语义不变）
_LEGACY_REASON_ALIASES = {"connect": FAILURE_CONNECT_FAILED}

#: **我们自己的**原因词表：只有取值落在这里的字符串 ``reason`` 才当分类词用。
#: 为什么必须这样：``urllib.error.HTTPError.reason`` 是 HTTP 短语（"Not Found"）、
#: ``URLError.reason`` 是**异常对象**（ConnectionRefusedError）—— 直接拿来当分类词
#: 会得到 "Not Found"/"refused" 这种没法聚合的标签。
_KNOWN_REASON_TOKENS = frozenset({
    FAILURE_MISSING_API_KEY, FAILURE_CONNECT_FAILED, FAILURE_HTTP_5XX, FAILURE_HTTP_4XX,
    FAILURE_TIMEOUT, "connect", "http", "model_missing", "auth", "rate_limit",
    "bad_request", "bad_response", "unknown",
})

#: 打印进日志/响应的异常摘要上限（截断，避免把整段 HTML 错误页糊上去）
SUMMARY_LIMIT = 300

#: 脱敏：Bearer token / api_key=xxx / sk-xxx 一律打码（错误信息可能带请求头回显）
_SECRET_RE = re.compile(
    r"(?i)(bearer\s+|api[_-]?key[\"'\s:=]+|\bsk-)[A-Za-z0-9._\-]{6,}")


def redact_secrets(text: str) -> str:
    """把疑似密钥打码（日志与响应都走这里，避免把 key 写进证据）。"""
    return _SECRET_RE.sub(lambda match: match.group(1) + "***", text or "")


def summarize_exception(exc: BaseException, limit: int = SUMMARY_LIMIT) -> str:
    """异常摘要：单行化 + 脱敏 + 截断（F2 要求的"截断的异常摘要"就是它）。"""
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ").replace("\r", " ")
    text = redact_secrets(" ".join(text.split()))
    if len(text) > limit:
        text = text[:limit] + "...(截断)"
    return text


def _http_status(exc: BaseException) -> int:
    cause = getattr(exc, "__cause__", None)
    for candidate in (exc, cause):
        status = getattr(candidate, "code", 0)
        if isinstance(status, int) and status:
            return status
    match = re.search(r"\bHTTP (\d{3})\b", str(exc) or "")
    return int(match.group(1)) if match else 0


def is_timeout_error(exc: BaseException) -> bool:
    """是否超时（含 ``URLError(reason=timeout)`` 这种被包一层的）。"""
    if isinstance(exc, LLMError) and getattr(exc, "reason", "") == FAILURE_TIMEOUT:
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, urllib.error.URLError):
        return (isinstance(exc.reason, (TimeoutError, socket.timeout))
                or "timed out" in str(exc))
    return False


def _reason_token(exc: BaseException) -> str:
    """取「我们自己写进去的」原因词；不是已知词就返回空串（见 ``_KNOWN_REASON_TOKENS``）。"""
    raw = getattr(exc, "reason", "")
    if not isinstance(raw, str):
        return ""
    token = raw.strip()
    return token if token in _KNOWN_REASON_TOKENS else ""


def failure_reason(exc: BaseException) -> str:
    """把异常归类成新口径的失败原因（见模块 docstring 的对照表）。

    **不猜**：认不出来就返回 ``unknown`` —— 宁可知其不可知，也不编一个像样的原因。
    包装过的异常（``LLMError`` 套 ``URLError``）会**顺着 ``__cause__`` 链**往下找一层层
    归因：上游只拿到"调用失败"那句话时，原因不能退化成 unknown（t120 F2）。
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        reason = _failure_reason_one(current)
        if reason != "unknown":
            return reason
        current = current.__cause__ or current.__context__
    return "unknown"


def _failure_reason_one(exc: BaseException) -> str:
    """单层归类（不加 cause 链）。"""
    status = _http_status(exc)
    if status:
        # 有状态码就以状态码为准（比字符串原因可靠）：
        # 400/401/403/404/408/429 有专名，其余 5xx -> http_5xx、4xx -> http_4xx
        by_status = {400: "bad_request", 401: "auth", 403: "auth", 404: "model_missing",
                     408: FAILURE_TIMEOUT, 429: "rate_limit"}.get(status)
        if by_status:
            return by_status
        if status >= 500:
            return FAILURE_HTTP_5XX
        if status >= 400:
            return FAILURE_HTTP_4XX
        return "http"
    token = _reason_token(exc)
    if token in _LEGACY_REASON_ALIASES:
        return _LEGACY_REASON_ALIASES[token]
    if token and token != "http":
        return token                       # timeout / model_missing / auth / bad_response …
    if token == "http":
        return "http"                      # 拿不到状态码时如实保留旧词，不猜 4xx/5xx
    if is_timeout_error(exc):
        return FAILURE_TIMEOUT
    if isinstance(exc, urllib.error.URLError):
        return FAILURE_CONNECT_FAILED
    if isinstance(exc, (ConnectionError, OSError)):
        return FAILURE_CONNECT_FAILED
    if isinstance(exc, json.JSONDecodeError):
        return "bad_response"
    if "缺少 API key" in str(exc) or "missing api key" in str(exc).lower():
        return FAILURE_MISSING_API_KEY
    return "unknown"


class LLMClient(ABC):
    name: str = "base"

    def __init__(self, model: str = "", temperature: float = 0.3,
                 max_tokens: int = 1024, timeout: float = 60.0) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @abstractmethod
    def _complete(self, messages: list[dict]) -> str:
        """子类实现：一次非流式补全。"""

    def chat(self, messages: list[dict]) -> str:
        logger.info("入口 %s.chat(model=%s, messages=%d)", self.name, self.model, len(messages))
        started = time.perf_counter()
        try:
            text = self._complete(messages)
        except Exception as exc:  # noqa: BLE001 - 统一包装成 LLMError
            logger.exception("%s 调用失败", self.name)
            # 包装**必须带上可分类原因**：否则上游（router/API）只看到一句中文描述，
            # 归因退化成 unknown —— t120 F2 说的"失败只给 reason=unknown"就是这么来的。
            raise LLMError(f"{self.name} 调用失败: {exc}",
                           reason=failure_reason(exc)) from exc
        logger.info("出口 %s.chat -> %d 字符，耗时 %.2fs",
                    self.name, len(text or ""), time.perf_counter() - started)
        return text

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """默认实现：一次性返回。真实后端可覆写为真正的流式。"""
        yield self.chat(messages)

    def health(self) -> dict:
        return {"provider": self.name, "model": self.model}


def build_llm_client(provider: str, model: str = "", temperature: float = 0.3,
                     max_tokens: int = 1024, timeout: float = 60.0,
                     deepseek_base_url: str = "", openai_base_url: str = "",
                     ollama_base_url: str = "", ollama_api_key: str = "",
                     ollama_top_p: float = 0.9, keep_alive: str = "10m",
                     num_gpu: int | None = None, num_thread: int | None = None,
                     max_retries: int = 2, retry_backoff: float = 1.0) -> LLMClient:
    """按 provider 名构造客户端。

    provider 取值：``mock`` / ``ollama`` / ``deepseek`` / ``openai_compat``
    （``openai`` / ``vllm`` / ``sglang`` / ``qwen`` 等为兼容别名）。

    注意：``ollama`` 分支的模型名默认 ``qwen2.5:3b``；``base_url`` 取调用方传入的
    ``config.ollama_base_url``，**各机自用 127.0.0.1:11434，不要硬编码 VM 地址**。
    """
    key = (provider or "mock").strip().lower()

    if key in ("mock", "offline", "fake"):
        from .mock import MockLLMClient

        return MockLLMClient(model=model or "mock-legal", temperature=temperature,
                             max_tokens=max_tokens, timeout=timeout)

    if key in ("ollama", "local", "ollama_local"):
        from .ollama import DEFAULT_OLLAMA_LLM_MODEL, OllamaClient

        return OllamaClient(
            model=model or DEFAULT_OLLAMA_LLM_MODEL,
            base_url=ollama_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            top_p=ollama_top_p,
            keep_alive=keep_alive,
            num_gpu=num_gpu,
            num_thread=num_thread,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            api_key=ollama_api_key,
        )

    if key == "deepseek":
        from .deepseek import DeepSeekClient

        return DeepSeekClient(model=model or "deepseek-chat", temperature=temperature,
                              max_tokens=max_tokens, timeout=timeout,
                              base_url=deepseek_base_url or "https://api.deepseek.com")

    if key in ("openai_compat", "openai-compat", "openai", "vllm", "sglang", "xinference",
               "doubao", "siliconflow", "qwen"):
        from .openai_compat import OpenAICompatClient

        return OpenAICompatClient(model=model or "qwen", temperature=temperature,
                                  max_tokens=max_tokens, timeout=timeout,
                                  base_url=openai_base_url)

    raise ValueError(
        f"未知的 llm_provider: {provider!r}"
        "（可选 mock | ollama | deepseek | openai_compat）"
    )
