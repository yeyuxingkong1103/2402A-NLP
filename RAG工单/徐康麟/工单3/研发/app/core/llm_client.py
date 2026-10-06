# -*- coding: utf-8 -*-
"""工单3 可插拔 LLM 客户端（设计/接口设计.md §2.4、§3.16 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

后端优先级：``ollama``（``POST /api/generate`` 流式）→ ``openai``（兼容 vLLM/SGLang 的
``POST /v1/chat/completions`` 流式）→ ``extractive``（无 LLM 时的确定性兜底）。

硬约束：
    * 探测超时 ``0.5 s`` 且**不重试**（``RAG_LLM__PROBE_TIMEOUT_S``）；
    * **流式**逐块 yield，首字延迟在客户端内计时（收到首个非空 delta 时记 ``first_token_ms``）；
    * 全失败不得静默：抛 ``LLMProbeError`` / ``LLMResponseError``，或以 ``extractive`` 显式降级并留痕。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence

from .config import AppConfig, get_config
from .errors import LLMProbeError, LLMResponseError
from .text_utils import keyword_coverage, split_sentences, text_digest, tokenize

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

UNKNOWN_TEXT = "不清楚"


def _lazy_logger(logger: Any, module: str = "llm_client") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class LLMBackendInfo:
    """后端探测结果（设计 §2.4 冻结字段）。"""

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
    """极简 JSON HTTP（标准库；探测调用不重试）。"""
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# 探测与后端选择
# ---------------------------------------------------------------------------
def probe_backends(cfg: AppConfig | None = None, *, logger: Any = None) -> list[LLMBackendInfo]:
    """探测可用生成后端（超时 0.5 s、不重试）。"""
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
                data = _http_json(f"{config.llm.openai_base_url.rstrip('/')}/v1/models", None,
                                  config.llm.probe_timeout_s)
                probe_ms = round((time.perf_counter() - started) * 1000, 2)
                infos.append(LLMBackendInfo(
                    name="openai", model=config.llm.openai_model or "<未配置>",
                    base_url=config.llm.openai_base_url, available=True, probe_ms=probe_ms,
                    error=None if data is not None else "空响应",
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
    def _stream_ollama(self, prompt: str, *, max_tokens: int, temperature: float,
                       trace_id: str | None, log: Any) -> Iterator[str]:
        """Ollama ``/api/generate`` 流式（取 ``response`` 字段；推理模型的 ``thinking`` 单独计数不混入答案）。"""
        payload: dict[str, Any] = {
            "model": self.backend.model, "prompt": prompt, "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        # 推理模型（deepseek-r1 等）默认会先输出思维链，占满 num_predict 导致答案为空；
        # 需要「直接作答」时显式关闭思考（Ollama ≥0.5 支持 think 参数）
        if self.begin_think is False:
            payload["think"] = False
        request = urllib.request.Request(
            f"{self.backend.base_url}/api/generate",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        thinking_chars = 0
        with urllib.request.urlopen(request, timeout=float(self.cfg.llm.request_timeout_s)) as response:
            for raw_line in response:
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
                    thinking_chars += len(str(obj["thinking"]))       # 思维链不进答案，只记长度
                piece = obj.get("response") or ""
                if piece:
                    yield piece
                if obj.get("done"):
                    if thinking_chars:
                        log.log_event("llm.thinking", backend="ollama", model=self.backend.model,
                                      thinking_chars=thinking_chars)
                    break

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
        request = urllib.request.Request(
            f"{self.backend.base_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers, method="POST",
        )
        with urllib.request.urlopen(request, timeout=float(self.cfg.llm.request_timeout_s)) as response:
            for raw_line in response:
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

    # -- 对外 --------------------------------------------------------------
    def generate(self, prompt: str, *, stream: bool = True, max_tokens: int | None = None,
                 temperature: float | None = None, trace_id: str | None = None,
                 logger: Any = None) -> Iterator[LLMDelta]:
        """流式生成（逐个 ``LLMDelta`` 产出；``stream=False`` 时一次性产出整段）。"""
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
        raise LLMProbeError(f"后端 {self.backend.name} 不支持流式生成（extractive 请走 ExtractiveGenerator）",
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


# ---------------------------------------------------------------------------
# 抽取式兜底
# ---------------------------------------------------------------------------
class ExtractiveGenerator:
    """无 LLM 时的确定性兜底：从召回片段中抽取与问题最相关的句子作答。"""

    def __init__(self, *, cfg: AppConfig | None = None, logger: Any = None) -> None:
        self.cfg = cfg or get_config()
        self.log = _lazy_logger(logger)

    def generate(self, question: str, chunks: Sequence[Any], *, language: str = "zh",
                 logger: Any = None) -> LLMResult:
        """按关键词覆盖度挑最佳句子；无可用句子则回「不清楚」。"""
        log = logger or self.log
        from .citation import format_citation  # 延迟导入避免循环

        with log.enter("ExtractiveGenerator.generate",
                       {"question": question[:80], "chunks": len(chunks), "language": language}) as span:
            started = time.perf_counter()
            query_tokens = tokenize(question)
            best_score = -1.0
            best_sentence = ""
            best_chunk = None
            for chunk in chunks:
                for sentence in split_sentences(str(getattr(chunk, "content", ""))):
                    if len(sentence) < 12:
                        continue
                    score = keyword_coverage(query_tokens, tokenize(sentence))
                    if score > best_score:
                        best_score, best_sentence, best_chunk = score, sentence, chunk
            if best_chunk is None or not best_sentence:
                text = UNKNOWN_TEXT
            else:
                cite = format_citation(str(getattr(best_chunk, "file_name", "")),
                                       int(getattr(best_chunk, "page", 0)))
                text = f"{best_sentence.strip()}\n引用：{cite}"
            total_ms = round((time.perf_counter() - started) * 1000, 2)
            result = LLMResult(text=text, backend="extractive", model="rule-based", first_token_ms=0.0,
                               total_ms=total_ms, prompt_chars=len(question), output_chars=len(text))
            log.log_event("llm.response", backend="extractive", output_digest=text_digest(text, limit=80),
                          total_ms=total_ms, chars=len(text), coverage=round(best_score, 4))
            span.set_output({"chars": len(text), "coverage": round(best_score, 4)})
            return result
