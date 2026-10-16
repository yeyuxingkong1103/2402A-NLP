# -*- coding: utf-8 -*-
"""工单4 可插拔 LLM 客户端（移植自 工单3 同源实现，按工单4 契约适配）。

工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单
工单17 补丁（F1 连接复用 / F7 Ollama 驻留控制，均受 ``PERF_ENABLED`` 总开关控制）：
    * ``_http_json`` / ``_stream_ollama`` / ``_stream_openai`` 全部改走
      :mod:`service.http_pool` 的 keep-alive 连接池，消除「每请求新建 TCP」；
      优化关闭时**自动退回** ``urllib.request.urlopen``，行为与基座逐位一致；
    * ``keep_alive`` 由 ``PERF_OLLAMA_KEEP_ALIVE`` 决定（默认 ``5m``），
      优化关闭时保留基座硬编码的 ``30m``；
    * 新增 :func:`unload_models` / :meth:`LLMClient.unload` —— 主动卸载 Ollama 模型，
      对应原文「GPU 内存被大量占用且释放不及时」。

后端优先级：``ollama``（``POST /api/generate`` 流式）→ ``openai``（兼容 vLLM / SGLang 的
``POST /v1/chat/completions`` 流式）→ ``extractive``（无 LLM 时的确定性兜底）。

硬约束（工单4 任务书原文）：
    * 探测超时 ``0.5 s`` 且**不重试**（``RAG_LLM__PROBE_TIMEOUT_S``）；
    * **流式**逐块 yield，首字延迟在客户端内计时（收到首个非空 delta 时记 ``first_token_ms``）；
    * 全失败不得静默：抛 ``LLMProbeError`` / ``LLMResponseError``，或以 ``extractive`` 显式降级并留痕；
    * 生成模型固定 ``qwen2.5:3b``（**禁止** deepseek-r1:7b：工单2 实测首字 18.68 s，破坏 ≤3 s 硬指标）；
    * 嵌入统一走本地 Ollama ``bge-m3:latest``（1024 维）。

本模块只依赖标准库 ``urllib``：本机 pip 通道不通（见 设计/环境事实.md），不引入新依赖。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from .config import AppConfig, get_config
from .errors import LLMProbeError, LLMResponseError
from .perf_hook import keep_alive as _keep_alive_of, pool as _pool_of
from .text_utils import text_digest

WORK_ORDER = "人工智能NLP-RAG 项目-金融问答系统部署"

UNKNOWN_TEXT = "不清楚"


def _lazy_logger(logger: Any, module: str = "llm_client") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class LLMBackendInfo:
    """后端探测结果。"""

    name: str
    model: str
    base_url: str
    available: bool
    probe_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LLMDelta:
    """流式增量块。"""

    text: str
    is_first: bool
    index: int


@dataclass(slots=True)
class LLMResult:
    """一次完整生成的结果（含首字与总耗时）。"""

    text: str
    backend: str
    model: str
    first_token_ms: float
    total_ms: float
    prompt_chars: int
    output_chars: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    """极简 JSON HTTP（标准库；探测调用**不重试**）。

    工单17 F1：``PERF_ENABLED=1`` 时走进程级 keep-alive 连接池（复用 TCP），
    否则原样走 ``urllib.request.urlopen``（基座行为）。
    """
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    pool = _pool_of()
    if pool is not None:
        return pool.request_json(url, payload, timeout=timeout)
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def unload_models(base_url: str = "http://127.0.0.1:11434", *,
                  models: list[str] | None = None, timeout: float = 20.0) -> dict[str, Any]:
    """工单17 F7：主动卸载 Ollama 常驻模型（``keep_alive="0s"``）。

    对应工单原文「GPU 资源利用不当：仅文本问答时 GPU 内存也被大量占用且释放不及时」。
    ``models=None`` 时先 ``GET /api/ps`` 取当前驻留列表，再逐个卸载。
    """
    out: dict[str, Any] = {"ok": False, "unloaded": [], "errors": []}
    base = base_url.rstrip("/")
    targets = list(models or [])
    if not targets:
        try:
            ps = _http_json(f"{base}/api/ps", None, timeout)
            targets = [str(m.get("name") or "") for m in (ps.get("models") or [])]
        except Exception as exc:  # noqa: BLE001 —— 卸载失败绝不能影响主流程
            out["errors"].append(f"查询 /api/ps 失败：{type(exc).__name__}: {exc}")
            return out
    for name in [t for t in targets if t]:
        try:
            _http_json(f"{base}/api/generate",
                       {"model": name, "keep_alive": "0s", "prompt": ""}, timeout)
            out["unloaded"].append(name)
        except Exception as exc:  # noqa: BLE001
            out["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
    out["ok"] = bool(out["unloaded"]) or not targets
    out["keep_alive"] = "0s"
    return out


# ---------------------------------------------------------------------------
# 探测与后端选择
# ---------------------------------------------------------------------------
def probe_backends(cfg: AppConfig | None = None, *, logger: Any = None) -> list[LLMBackendInfo]:
    """探测可用生成后端（超时 0.5 s、**不重试**）。"""
    config = cfg or get_config()
    log = _lazy_logger(logger)
    with log.enter("probe_backends", {"ollama": config.llm.ollama_base_url,
                                      "probe_timeout_s": config.llm.probe_timeout_s,
                                      "backend_cfg": config.llm.backend}) as span:
        infos: list[LLMBackendInfo] = []
        # ① Ollama：/api/tags 列模型（0.5 s 超时，不重试）
        started = time.perf_counter()
        try:
            data = _http_json(f"{config.llm.ollama_base_url}/api/tags", None, config.llm.probe_timeout_s)
            probe_ms = round((time.perf_counter() - started) * 1000, 2)
            names = [str(m.get("name") or "") for m in (data.get("models") or [])]
            has_model = any(n.split(":")[0] == config.llm.ollama_gen_model.split(":")[0] for n in names)
            infos.append(LLMBackendInfo(
                name="ollama", model=config.llm.ollama_gen_model, base_url=config.llm.ollama_base_url,
                available=has_model, probe_ms=probe_ms,
                error=None if has_model else f"模型未安装：{config.llm.ollama_gen_model}",
            ))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            infos.append(LLMBackendInfo(
                name="ollama", model=config.llm.ollama_gen_model, base_url=config.llm.ollama_base_url,
                available=False, probe_ms=round((time.perf_counter() - started) * 1000, 2),
                error=f"{type(exc).__name__}: {exc}",
            ))
        # ② OpenAI 兼容（仅在配置了 base_url 时才探测）
        started = time.perf_counter()
        if config.llm.openai_base_url:
            try:
                _http_json(f"{config.llm.openai_base_url.rstrip('/')}/v1/models", None,
                           config.llm.probe_timeout_s)
                infos.append(LLMBackendInfo(
                    name="openai", model=config.llm.openai_model or "<未配置>",
                    base_url=config.llm.openai_base_url, available=True,
                    probe_ms=round((time.perf_counter() - started) * 1000, 2), error=None,
                ))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                infos.append(LLMBackendInfo(
                    name="openai", model=config.llm.openai_model or "<未配置>",
                    base_url=config.llm.openai_base_url, available=False,
                    probe_ms=round((time.perf_counter() - started) * 1000, 2),
                    error=f"{type(exc).__name__}: {exc}",
                ))
        else:
            infos.append(LLMBackendInfo(
                name="openai", model="<未配置>", base_url="", available=False, probe_ms=0.0,
                error="未配置 RAG_LLM__OPENAI_BASE_URL",
            ))
        # ③ 抽取式兜底永远可用（无外部依赖）
        infos.append(LLMBackendInfo(name="extractive", model="rule-based", base_url="local",
                                    available=True, probe_ms=0.0, error=None))
        for info in infos:
            log.log_event("llm.probe", name=info.name, model=info.model, base_url=info.base_url,
                          available=info.available, probe_ms=info.probe_ms, error=info.error)
        span.set_output({"backends": [i.to_dict() for i in infos]})
        return infos


def resolve_backend(cfg: AppConfig | None = None, *, logger: Any = None) -> LLMBackendInfo:
    """按配置与探测结果解析生成后端（``auto`` 时 ollama → openai → extractive）。"""
    config = cfg or get_config()
    log = _lazy_logger(logger)
    with log.enter("resolve_backend", {"backend_cfg": config.llm.backend}) as span:
        infos = {i.name: i for i in probe_backends(config, logger=log)}
        wanted = str(config.llm.backend or "auto").lower()
        if wanted not in {"auto", ""}:
            info = infos.get(wanted)
            if info is None or not info.available:
                raise LLMProbeError(f"指定的后端不可用：{wanted}（{getattr(info, 'error', '未知')}）",
                                    detail={"backend": wanted})
            span.set_output(info.to_dict())
            return info
        for name in ("ollama", "openai", "extractive"):
            info = infos.get(name)
            if info is not None and info.available:
                if name != "ollama":
                    log.log_event("llm.degrade", level="WARNING", **{"from": "ollama", "to": name},
                                  reason=(infos.get("ollama").error if infos.get("ollama") else "ollama 不可用"))
                span.set_output(info.to_dict())
                return info
        raise LLMProbeError("没有任何可用的生成后端（含 extractive 兜底）", detail={"probed": list(infos)})


def warmup_generate(cfg: AppConfig | None = None, *, logger: Any = None) -> dict[str, Any]:
    """预热 Ollama 生成模型（``keep_alive`` 常驻），消除冷启动首字 3.4 s 违约。

    环境事实（设计/环境事实.md §8）：qwen2.5:3b 冷启动首字 **3430.8 ms（必违约）**、
    预热后 23.5~25.2 ms。因此**服务启动与评测前必须预热**，且必须把预热测量值写进日志。
    """
    config = cfg or get_config()
    log = _lazy_logger(logger)
    with log.enter("warmup_generate", {"model": config.llm.ollama_gen_model,
                                       "base_url": config.llm.ollama_base_url}) as span:
        started = time.perf_counter()
        payload = {
            "model": config.llm.ollama_gen_model, "prompt": "你好", "stream": False,
            "options": {"num_predict": 4, "temperature": 0.0}, "keep_alive": _keep_alive_of(),
        }
        try:
            data = _http_json(f"{config.llm.ollama_base_url}/api/generate", payload,
                              max(float(config.llm.request_timeout_s), 60.0))
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            out = {"ok": True, "elapsed_ms": elapsed_ms, "model": config.llm.ollama_gen_model,
                   "response_head": str(data.get("response") or "")[:40]}
            log.log_event("llm.warmup", **out)
            span.set_output(out)
            return out
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            out = {"ok": False, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                   "model": config.llm.ollama_gen_model, "error": f"{type(exc).__name__}: {exc}"}
            log.log_event("llm.warmup_failed", level="WARNING", **out)
            span.set_output(out)
            return out


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------
class LLMClient:
    """流式 LLM 客户端（ollama / openai 兼容）。"""

    def __init__(self, *, cfg: AppConfig | None = None, backend: LLMBackendInfo | None = None,
                 logger: Any = None) -> None:
        self.cfg = cfg or get_config()
        self.log = _lazy_logger(logger)
        self.backend = backend or resolve_backend(self.cfg, logger=self.log)
        self.begin_think: bool = bool(getattr(self.cfg.llm, "enable_think", True))

    # -- 内部：两套流式实现 ------------------------------------------------
    @staticmethod
    def _iter_ollama_lines(lines: Any, log: Any, model: str) -> Iterator[str]:
        """把 Ollama ``/api/generate`` 的 NDJSON 行序列转成文本块。

        推理模型 ``thinking`` 单独计数、**不混入答案**（基座语义不变）。
        """
        thinking_chars = 0
        for raw_line in lines:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                log.log_event("llm.chunk_error", level="WARNING", backend="ollama",
                              exception=f"{type(exc).__name__}: {exc}", raw=line[:120])
                continue
            if obj.get("thinking"):
                thinking_chars += len(str(obj["thinking"]))
            piece = obj.get("response") or ""
            if piece:
                yield piece
            if obj.get("done"):
                if thinking_chars:
                    log.log_event("llm.thinking", backend="ollama", model=model,
                                  thinking_chars=thinking_chars)
                break

    @staticmethod
    def _iter_sse_lines(lines: Any, log: Any) -> Iterator[str]:
        """把 OpenAI 兼容 SSE ``data:`` 行序列转成文本块。"""
        for raw_line in lines:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError as exc:
                log.log_event("llm.chunk_error", level="WARNING", backend="openai",
                              exception=f"{type(exc).__name__}: {exc}", raw=body[:120])
                continue
            choices = obj.get("choices") or []
            if choices:
                piece = (choices[0].get("delta") or {}).get("content") or ""
                if piece:
                    yield piece

    def _stream_ollama(self, prompt: str, *, max_tokens: int, temperature: float,
                       trace_id: str | None, log: Any) -> Iterator[str]:
        """Ollama ``/api/generate`` 流式（取 ``response`` 字段；推理模型 ``thinking`` 单独计数不混入答案）。

        工单17 F1：优化开启时复用 keep-alive 连接（读完流后归还池子）；关闭时原样 ``urlopen``。
        """
        payload: dict[str, Any] = {
            "model": self.backend.model, "prompt": prompt, "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "keep_alive": _keep_alive_of(),
        }
        if self.begin_think is False:
            payload["think"] = False
        url = f"{self.backend.base_url}/api/generate"
        timeout = float(self.cfg.llm.request_timeout_s)
        pool = _pool_of()
        if pool is not None:
            with pool.stream_lines(url, payload, timeout=timeout) as response:
                yield from self._iter_ollama_lines(response, log, self.backend.model)
            return
        request = urllib.request.Request(
            url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            yield from self._iter_ollama_lines(response, log, self.backend.model)

    def _stream_openai(self, prompt: str, *, max_tokens: int, temperature: float,
                       trace_id: str | None, log: Any) -> Iterator[str]:
        """OpenAI 兼容 ``/v1/chat/completions`` 流式（SSE ``data:`` 行，取 ``delta.content``）。"""
        payload = {
            "model": self.backend.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True, "temperature": temperature, "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.cfg.llm.openai_api_key:
            headers["Authorization"] = f"Bearer {self.cfg.llm.openai_api_key}"
        url = f"{self.backend.base_url.rstrip('/')}/v1/chat/completions"
        timeout = float(self.cfg.llm.request_timeout_s)
        pool = _pool_of()
        if pool is not None:
            with pool.stream_lines(url, payload, timeout=timeout) as response:
                yield from self._iter_sse_lines(response, log)
            return
        request = urllib.request.Request(
            url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers, method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            yield from self._iter_sse_lines(response, log)

    # -- 对外 --------------------------------------------------------------
    def generate(self, prompt: str, *, stream: bool = True, max_tokens: int | None = None,
                 temperature: float | None = None, trace_id: str | None = None,
                 logger: Any = None) -> Iterator[LLMDelta]:
        """流式生成（逐个 ``LLMDelta`` 产出；``stream=False`` 时仍按块产出，调用方拼接即可）。"""
        log = logger or self.log
        tokens = int(max_tokens or self.cfg.llm.max_tokens)
        temp = float(self.cfg.llm.temperature if temperature is None else temperature)
        log.log_event("llm.request", backend=self.backend.name, model=self.backend.model,
                      prompt_digest=text_digest(prompt, limit=100), max_tokens=tokens, temperature=temp,
                      trace_id=trace_id)
        started = time.perf_counter()
        index = 0
        for piece in self._iter_pieces(prompt, tokens=tokens, temp=temp, trace_id=trace_id, log=log):
            yield LLMDelta(text=piece, is_first=(index == 0), index=index)
            index += 1
        if index == 0:
            log.log_event("llm.empty_output", level="ERROR", backend=self.backend.name,
                          elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
            raise LLMResponseError("LLM 未返回任何文本", detail={"backend": self.backend.name})

    def _iter_pieces(self, prompt: str, *, tokens: int, temp: float,
                     trace_id: str | None, log: Any) -> Iterator[str]:
        """按后端分派到具体流式实现。"""
        if self.backend.name == "ollama":
            yield from self._stream_ollama(prompt, max_tokens=tokens, temperature=temp,
                                           trace_id=trace_id, log=log)
            return
        if self.backend.name == "openai":
            yield from self._stream_openai(prompt, max_tokens=tokens, temperature=temp,
                                           trace_id=trace_id, log=log)
            return
        raise LLMProbeError(f"后端 {self.backend.name} 不支持流式生成（extractive 请走 qa_engine 的合成器）",
                            detail={"backend": self.backend.name})

    def generate_full(self, prompt: str, *, max_tokens: int | None = None, temperature: float | None = None,
                      trace_id: str | None = None, logger: Any = None) -> LLMResult:
        """一次性拿到完整结果（内部仍走流式，以便如实测量首字延迟）。"""
        log = logger or self.log
        parts: list[str] = []
        first_ms: float | None = None
        started = time.perf_counter()
        for delta in self.generate(prompt, max_tokens=max_tokens, temperature=temperature,
                                   trace_id=trace_id, logger=log):
            if delta.is_first:
                first_ms = round((time.perf_counter() - started) * 1000, 2)
                log.log_event("llm.first_token", trace_id=trace_id, first_token_ms=first_ms,
                              backend=self.backend.name)
            parts.append(delta.text)
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        text = "".join(parts)
        log.log_event("llm.response", backend=self.backend.name, model=self.backend.model,
                      output_digest=text_digest(text, limit=100), total_ms=total_ms, chars=len(text),
                      trace_id=trace_id)
        return LLMResult(text=text, backend=self.backend.name, model=self.backend.model,
                         first_token_ms=float(first_ms or 0.0), total_ms=total_ms,
                         prompt_chars=len(prompt), output_chars=len(text))

    def health(self) -> dict[str, Any]:
        """后端健康状态。"""
        info = {"backend": self.backend.name, "model": self.backend.model,
                "base_url": self.backend.base_url, "available": self.backend.available,
                "probe_ms": self.backend.probe_ms}
        self.log.log_event("llm.health", **info)
        return info

    def unload(self) -> dict[str, Any]:
        """工单17 F7：主动卸载本客户端对应的 Ollama 模型（``keep_alive="0s"``）。"""
        if self.backend.name != "ollama":
            return {"ok": True, "skipped": True, "reason": f"后端 {self.backend.name} 非 ollama"}
        return unload_models(self.backend.base_url, models=[self.backend.model])
