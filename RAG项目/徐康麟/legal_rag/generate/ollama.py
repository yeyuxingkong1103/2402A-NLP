# -*- coding: utf-8 -*-
"""Ollama 原生 ``/api/chat`` 生成客户端（非流式 + 真流式），免密钥、免 torch。

为什么不用 OpenAI 兼容层
------------------------
Ollama 的 ``/api/chat`` 会带回官方统计字段（**纳秒**），本客户端把它们
逐一映射成指标，不需要客户端「猜」：

================================  ==========================================
返回字段                           映射到的指标
================================  ==========================================
``prompt_eval_count``              ``llm_prompt_tokens``
``prompt_eval_duration``           ``llm_prefill_seconds``（预填充）
``eval_count``                     ``llm_output_tokens``
``eval_duration``                  ``llm_decode_seconds``（纯解码）
``load_duration``                  ``llm_load_seconds``（**冷启动元凶**）
``total_duration``                 日志字段 ``ollama_total_seconds``
================================  ==========================================

走 OpenAI 兼容层就拿不到 ``load_duration``，也就无法区分
「冷启动加载模型」与「真的慢」——VM 上这一项实测能占一次 47s 请求的 87%。

双环境
------
``base_url`` 一律取 :attr:`RagConfig.ollama_base_url`（默认各机自用
``http://127.0.0.1:11434``），**不要硬编码 VM 地址**；跨机测试时用
``OLLAMA_BASE_URL=http://192.168.188.128:11434`` 覆盖。

Windows 本地有 RTX 2060（qwen2.5:3b 热态 84–92 tok/s），VM 无卡（约 10.8 tok/s），
同一套代码靠 Ollama 自动选择 GPU/CPU。

并发安全
--------
客户端实例只持有**不可变配置**（模型名、URL、采样参数）；
一次请求的全部可变状态都放在 :class:`LLMStats` 局部对象与生成器局部变量里，
因此「多请求共用一个 client」时流式响应不会互相串包。
"""
from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from .llm_base import LLMClient, LLMError, failure_reason
from .llm_metrics import (
    UNKNOWN, LLMStats, current_role_id, log_llm_start, record_llm_stats,
    take_degraded,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OllamaClient", "OllamaError", "DEFAULT_OLLAMA_LLM_MODEL",
    "resolve_ollama_model", "OLLAMA_PROVIDERS",
]

DEFAULT_OLLAMA_LLM_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

#: build_llm_client 里认这些 provider 名为「Ollama」
OLLAMA_PROVIDERS = frozenset({"ollama", "local", "ollama_local"})

#: ``config.llm_model`` 的历史占位默认值（默认 provider 曾是 deepseek）。
#: 用户没显式设 LLM_MODEL 时，ollama 分支应改用 ``config.ollama.llm_model``。
_GENERIC_MODEL_DEFAULTS = frozenset({"", "deepseek-chat", "deepseek-reasoner"})

_NS = 1_000_000_000.0


class OllamaError(LLMError):
    """Ollama 调用失败。消息里一定带 base_url 与模型名，便于一眼定位。"""

    def __init__(self, message: str, *, reason: str = "unknown") -> None:
        super().__init__(message)
        self.reason = reason


def resolve_ollama_model(config: Any) -> str:
    """取 Ollama 生成模型名。

    优先级：显式 ``LLM_MODEL``（即 ``config.llm_model``，非占位默认值）
    → ``config.ollama.llm_model``（默认 ``qwen2.5:3b``）。
    """
    flat = str(getattr(config, "llm_model", "") or "").strip()
    if flat and flat not in _GENERIC_MODEL_DEFAULTS:
        return flat
    ollama = getattr(config, "ollama", None)
    sub = str(getattr(ollama, "llm_model", "") or "").strip()
    return sub or DEFAULT_OLLAMA_LLM_MODEL


def _ns_to_seconds(value: Any) -> float:
    try:
        return max(0.0, float(value) / _NS)
    except (TypeError, ValueError):
        return 0.0


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, OllamaError):
        return exc.reason == "timeout"
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, (TimeoutError, socket.timeout))
    return False


def _classify(exc: BaseException) -> str:
    """把异常归类成指标标签 ``reason``。"""
    if isinstance(exc, OllamaError):
        return exc.reason
    if isinstance(exc, urllib.error.HTTPError):
        return "model_missing" if exc.code == 404 else "http"
    if _is_timeout(exc):
        return "timeout"
    if isinstance(exc, urllib.error.URLError):
        return "connect"
    if isinstance(exc, (ConnectionError, OSError)):
        return "connect"
    return "unknown"


class OllamaClient(LLMClient):
    """Ollama ``/api/chat`` 客户端。"""

    name = "ollama"

    def __init__(self, model: str = DEFAULT_OLLAMA_LLM_MODEL,
                 base_url: str = DEFAULT_OLLAMA_BASE_URL,
                 temperature: float = 0.3, max_tokens: int = 512,
                 timeout: float = 180.0, top_p: float = 0.9,
                 keep_alive: str = "10m", num_gpu: int | None = None,
                 num_thread: int | None = None, max_retries: int = 2,
                 retry_backoff: float = 1.0, api_key: str = "",
                 role_id: str = "", probe_ps: bool = True) -> None:
        super().__init__(model=model or DEFAULT_OLLAMA_LLM_MODEL,
                         temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        self.base_url = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self.top_p = top_p
        self.keep_alive = keep_alive or ""
        self.num_gpu = num_gpu
        self.num_thread = num_thread
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        #: 显式 role_id（测试用）；默认从 observability 的请求上下文取
        self.role_id = role_id
        #: 请求前探测 ``/api/ps``，用于区分「冷启动」与「热态」
        self.probe_ps = probe_ps
        self._api_key = api_key
        # 显式禁用代理：Ollama 一律是本地/内网端点，走系统代理会拿到空响应
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # ---------------------------------------------------------------- 构造
    @classmethod
    def from_config(cls, config: Any, **overrides: Any) -> "OllamaClient":
        """按 ``RagConfig`` 构造（base_url / 模型名 / 超时 / keep_alive 全部来自配置）。"""
        ollama = config.ollama
        params: dict[str, Any] = {
            "model": resolve_ollama_model(config),
            "base_url": ollama.base_url,
            "temperature": config.llm_temperature,
            "max_tokens": config.llm_max_tokens,
            "timeout": config.llm_timeout,
            "top_p": ollama.top_p,
            "keep_alive": ollama.keep_alive,
            "num_gpu": ollama.num_gpu,
            "num_thread": ollama.num_thread,
            "max_retries": ollama.max_retries,
            "retry_backoff": ollama.retry_backoff,
            "api_key": ollama.api_key,
        }
        params.update(overrides)
        return cls(**params)

    # ---------------------------------------------------------------- 地址
    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    @property
    def tags_url(self) -> str:
        return f"{self.base_url}/api/tags"

    @property
    def ps_url(self) -> str:
        return f"{self.base_url}/api/ps"

    # ---------------------------------------------------------------- 内部
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/x-ndjson"}
        if self._api_key:          # 值绝不进日志
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(self, messages: list[dict], stream: bool) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
        }
        # 显式种子（F-D）：判定类调用需要**可复现**——只把 temperature 设 0 不足以保证
        # 同一输入 3 次给同一结论（采样器/并发内核仍可能选到不同的等价分支）。
        # 设了种子，同一 prompt + 同一模型就应当逐字复现。
        seed = getattr(self, "seed", None)
        if seed is not None:
            options["seed"] = int(seed)
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        if self.num_thread is not None:
            options["num_thread"] = self.num_thread
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        return payload

    def _open(self, payload: dict, timeout: float | None = None):
        request = urllib.request.Request(
            self.chat_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        return self._opener.open(request, timeout=timeout or self.timeout)

    def _get_json(self, url: str, timeout: float = 3.0) -> dict:
        request = urllib.request.Request(url, headers=self._headers(), method="GET")
        with self._opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))

    def loaded_models(self) -> list[str] | None:
        """``/api/ps`` 里当前驻留内存的模型；探测失败返回 None。"""
        try:
            body = self._get_json(self.ps_url)
        except Exception:  # noqa: BLE001 - 探测失败不是错误，只是「不知道」
            return None
        return [str(item.get("name") or item.get("model") or "") for item in body.get("models") or []]

    def _retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, OllamaError):
            # 模型不存在 / 参数错误这类问题重试没有意义
            return exc.reason in ("connect", "timeout", "http")
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code >= 500
        if isinstance(exc, (urllib.error.URLError, OSError)):
            return True
        return False

    def _sleep(self, attempt: int) -> None:
        delay = self.retry_backoff * attempt
        if delay > 0:
            time.sleep(delay)

    def _error_message(self, exc: BaseException, stream: bool) -> str:
        return (
            f"Ollama 调用失败（base_url={self.base_url}, model={self.model}, "
            f"stream={str(stream).lower()}, timeout={self.timeout:g}s）："
            f"{type(exc).__name__}: {exc}\n"
            f"排查：1) Ollama 是否在跑：curl {self.tags_url}；"
            f"2) 模型是否已拉取：ollama pull {self.model}；"
            f"3) 连远端要设 OLLAMA_BASE_URL（当前 {self.base_url}）。"
        )

    # ---------------------------------------------------------------- 统计
    def _new_stats(self, messages: list[dict], stream: bool) -> LLMStats:
        stats = LLMStats(
            provider=self.name,
            model=self.model,
            role_id=self.role_id or current_role_id(),
            stream=stream,
            prompt_chars=sum(len(str(m.get("content") or "")) for m in messages),
            keep_alive=self.keep_alive,
        )
        degraded = take_degraded()
        if degraded is not None:
            stats.degraded = True
            stats.degraded_from, stats.degraded_to = degraded
        if self.probe_ps:
            loaded = self.loaded_models()
            if loaded is not None:
                stats.loaded_before = any(
                    name == self.model or name.split(":")[0] == self.model.split(":")[0]
                    for name in loaded
                )
        return stats

    @staticmethod
    def _apply_ollama_fields(stats: LLMStats, body: dict) -> None:
        """把 Ollama 的官方统计字段（纳秒）落到指标口径上。"""
        try:
            stats.prompt_tokens = int(body.get("prompt_eval_count") or 0)
            stats.output_tokens = int(body.get("eval_count") or 0)
        except (TypeError, ValueError):  # pragma: no cover - 服务端字段异常不致命
            pass
        stats.load_seconds = _ns_to_seconds(body.get("load_duration"))
        stats.prefill_seconds = _ns_to_seconds(body.get("prompt_eval_duration"))
        stats.decode_seconds = _ns_to_seconds(body.get("eval_duration"))
        stats.ollama_total_seconds = _ns_to_seconds(body.get("total_duration"))

    def _mark_error(self, stats: LLMStats, exc: BaseException) -> None:
        stats.ok = False
        stats.error = f"{type(exc).__name__}: {exc}"
        stats.error_reason = _classify(exc)
        # t121 F2：新口径（connect -> connect_failed、http 按状态码细分 4xx/5xx）；
        # 旧口径保持不变（有测试钉住），两者都进指标/日志。
        stats.failure_reason = failure_reason(exc)

    # ---------------------------------------------------------------- 非流式
    def _complete(self, messages: list[dict]) -> str:
        stats = self._new_stats(messages, stream=False)
        log_llm_start(stats, base_url=self.base_url, messages=len(messages), logger=logger)
        started = time.perf_counter()
        try:
            body = self._post_chat(messages, stats)
            message = body.get("message") or {}
            text = str(message.get("content") or "")
            stats.output_chars = len(text)
            self._apply_ollama_fields(stats, body)
            return text
        except Exception as exc:  # noqa: BLE001 - 统一转成带上下文的 OllamaError
            self._mark_error(stats, exc)
            raise OllamaError(self._error_message(exc, stream=False),
                              reason=_classify(exc)) from exc
        finally:
            stats.total_seconds = time.perf_counter() - started
            record_llm_stats(stats, logger=logger)

    def _post_chat(self, messages: list[dict], stats: LLMStats) -> dict:
        payload = self._payload(messages, stream=False)
        attempt = 0
        while True:
            try:
                with self._open(payload) as response:
                    return json.loads(response.read().decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                error: BaseException = OllamaError(
                    f"HTTP {exc.code}: {detail}", reason="model_missing" if exc.code == 404 else "http")
            except Exception as exc:  # noqa: BLE001 - 网络/超时统一处理
                error = exc
            attempt += 1
            stats.attempts = attempt + 1
            if attempt > self.max_retries or not self._retryable(error):
                raise error
            logger.warning("Ollama 非流式调用失败，第 %d/%d 次重试（%s：%s）",
                           attempt, self.max_retries, type(error).__name__, error)
            self._sleep(attempt)

    # ---------------------------------------------------------------- 真流式
    def stream(self, messages: list[dict]) -> Iterator[str]:
        """真流式：``stream=true``，逐块 yield 增量文本。

        生成器内的所有状态（stats / 累计字符 / 是否已产出）都是**调用级局部变量**，
        因此同一个 client 被多个请求并发使用也不会串包。
        """
        stats = self._new_stats(messages, stream=True)
        log_llm_start(stats, base_url=self.base_url, messages=len(messages), logger=logger)
        started = time.perf_counter()
        try:
            yield from self._stream_chunks(messages, stats, started)
        except GeneratorExit:          # 调用方提前停止（客户端断连）
            raise
        except OllamaError as exc:     # 已经是带上下文的错误：只补记指标
            self._mark_error(stats, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            self._mark_error(stats, exc)
            raise OllamaError(self._error_message(exc, stream=True),
                              reason=_classify(exc)) from exc
        finally:
            stats.total_seconds = time.perf_counter() - started
            record_llm_stats(stats, logger=logger)

    def _stream_chunks(self, messages: list[dict], stats: LLMStats, started: float) -> Iterator[str]:
        payload = self._payload(messages, stream=True)
        attempt = 0
        while True:
            try:
                yield from self._read_stream(payload, stats, started)
                return
            except Exception as exc:  # noqa: BLE001 - 网络/超时/HTTP 统一处理
                attempt += 1
                stats.attempts = attempt + 1
                # 已经吐过 token 就不能重试（会重复输出）；未产出时允许重试
                if stats.output_chars > 0 or attempt > self.max_retries or not self._retryable(exc):
                    raise
                logger.warning("Ollama 流式调用失败，第 %d/%d 次重试（尚未产出 token，%s：%s）",
                               attempt, self.max_retries, type(exc).__name__, exc)
                self._sleep(attempt)

    def _read_stream(self, payload: dict, stats: LLMStats, started: float) -> Iterator[str]:
        try:
            response = self._open(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise OllamaError(
                f"HTTP {exc.code}: {detail}",
                reason="model_missing" if exc.code == 404 else "http") from exc

        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:   # 忽略心跳/空行之类的噪声
                    continue
                if chunk.get("error"):
                    raise OllamaError(f"Ollama 返回错误：{str(chunk['error'])[:300]}",
                                      reason="model_missing"
                                      if "not found" in str(chunk["error"]) else "http")
                piece = str((chunk.get("message") or {}).get("content") or "")
                if piece:
                    if stats.output_chars == 0:
                        # 首 token 延迟：从「请求发出」到「第一个 token 到达」
                        stats.ttft_seconds = time.perf_counter() - started
                    stats.output_chars += len(piece)
                    yield piece
                if chunk.get("done"):
                    self._apply_ollama_fields(stats, chunk)
                    return

    # ---------------------------------------------------------------- 运维
    def health(self) -> dict:
        """探活：``/api/tags`` 是否可达 + 模型是否已拉取 + 是否驻留内存。"""
        info: dict[str, Any] = {
            "provider": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "keep_alive": self.keep_alive,
            "timeout": self.timeout,
            "available": False,
            "model_present": False,
            "models": [],
            "loaded": [],
        }
        try:
            body = self._get_json(self.tags_url, timeout=min(self.timeout, 5.0))
            names = [str(item.get("name") or item.get("model") or "")
                     for item in body.get("models") or []]
            info["models"] = names
            info["available"] = True
            info["model_present"] = self.model in names
            info["loaded"] = self.loaded_models() or []
        except Exception as exc:  # noqa: BLE001 - 探活失败要能返回给 /health
            info["error"] = f"{type(exc).__name__}: {exc}"
        return info
