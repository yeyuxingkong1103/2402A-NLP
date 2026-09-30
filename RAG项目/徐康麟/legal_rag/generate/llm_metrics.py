# -*- coding: utf-8 -*-
"""LLM 请求级指标与日志（t3 采集，底座是 t1 的 ``metrics`` / ``observability``）。

一次生成请求要回答的问题
------------------------
用户原话：「在日志里面加入指标 比如首Token生成速度 一秒多少Token啊 就是加上很多的指标」

本模块把一次 LLM 调用拆成可比较的口径，**每个指标都带
``provider`` / ``model`` / ``role_id`` / ``stream`` 四个标签**：

=============================  ==================================================
指标                            含义
=============================  ==================================================
``llm_ttft_seconds``           首 token 延迟；非流式时 = 总耗时（标 ``stream=false``）
``llm_tokens_per_second``      生成速度，**两种口径都记**（见下）
``llm_prompt_tokens``          提示词 token 数（``prompt_eval_count``）
``llm_output_tokens``          生成 token 数（``eval_count``）
``llm_prefill_seconds``        预填充耗时（``prompt_eval_duration``）
``llm_decode_seconds``         解码耗时（``eval_duration``）
``llm_load_seconds``           模型加载耗时（``load_duration``）—— 单列，区分冷启动
``llm_total_seconds``          一次生成的总耗时（墙钟）
``llm_degraded_total``         降级次数（标签 ``from`` / ``to``，由 router 记）
``llm_timeout_total``          超时次数
``llm_error_total``            失败次数（标签 ``reason``）
=============================  ==================================================

> **同名指标的来源与 provider 有关**：上表的括号里写的是 **Ollama 原生字段**。
> OpenAI 兼容路径（``generate/openai_compat.py``）向同名 histogram 写样本时，
> 来源分别是 ``usage.prompt_tokens`` / ``usage.completion_tokens``（或客户端
> 内容 chunk 计数）与 ``total - ttft`` 导出量，**没有任何一项来自 ``eval_*`` /
> ``prompt_eval_count``**；该路径的结构性缺失项见其模块 docstring。
> ``METRIC_CATALOG`` 的 HELP 文本因此写成**不绑定单一后端**的表述。

两种 tok/s 口径（``llm_tokens_per_second`` 的 ``mode`` 标签）
------------------------------------------------------------
* ``mode="official"`` —— ``eval_count / eval_duration``：纯解码速度，不含排队与网络；
* ``mode="wall"``     —— ``output_tokens / (总耗时 - 首 token 时间)``：含排队/网络，更接近体感。

**``mode="wall"`` 的分母按 provider 刻意不同（不是 bug，勿「统一」）**：

* **Ollama 原生** ``/api/chat``（本模块的规范口径）：``output_tokens / (total - ttft)``，
  只扣掉「等首 token」的那一段；
* **OpenAI 兼容端点**（``generate/openai_compat.py::_throughput``）：``output_tokens / total``。
  该协议**拿不到** ``prompt_eval_duration``（预填充时长），无处可扣；而且该路径的
  ``decode`` 本身就是 ``total - ttft`` 的导出量，若 wall 也用 ``total - ttft``，
  两条 mode 会退化成同一个数，白白丢掉「整段体感」这一维。

因此**跨 provider 横向比较 ``mode="wall"`` 会得到不同口径的数字**：请按
``provider`` 标签分开看，或只比较同一 provider 内的趋势。同样的说明见
``docs/OBSERVABILITY.md``（指标表）与 ``docs/CLOUD.md`` §8 #4。

两者差得多，就说明「等待」而不是「解码」慢（Ollama 路径上最常见的元凶是
``load_duration`` 冷启动，实测能占一次请求的 87%）；对 OpenAI 兼容路径，
``wall`` 的分母已含预填充与网络，这条排障结论在该路径上**不成立**。

输出
----
1. 每个请求一条 INFO 汇总日志 ``[LLM_DONE] ...``（绝不含 prompt 全文与任何密钥）；
2. 每个请求一条 ``METRIC {json}`` 行（复用 t1 的 :func:`metrics.emit_request_metrics`），
   ``extra`` 里带本次请求的全部原始字段，``metrics`` 里带累计分布（P50/P95/P99）。
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from .. import metrics as M
from ..observability import current_request, log_event
from .llm_base import redact_secrets

logger = logging.getLogger(__name__)

__all__ = [
    "LLMStats", "record_llm_stats", "log_llm_start",
    "mark_degraded", "take_degraded", "last_stats",
    "mark_route_degraded", "take_route_degraded",
    "mark_context_trimmed", "take_context_trimmed",
    "current_role_id", "UNKNOWN",
]

#: 标签里未知值的占位（空字符串在 Prometheus 里难辨认）
UNKNOWN = "-"

#: 降级链上下文：router 在切换后端前标记，客户端建立 stats 时消费掉。
#: 用 ContextVar 而不是实例属性，保证「多个并发请求共用同一 client」时互不串包。
_DEGRADED: ContextVar[tuple[str, str] | None] = ContextVar(
    "legal_rag_llm_degraded", default=None
)

#: **路由层**降级记录 ``(from, to, reason)``（t121 F1/F2）。
#:
#: 与 ``_DEGRADED`` 的分工：``_DEGRADED`` 会被**客户端**在 ``_new_stats`` 里消费掉
#: （用于 LLM_DONE 日志里的 degrade_path）；而引擎要在**自己这一层**拿到"本轮到底
#: 降级了没、降到了谁"，用来如实填 ``Answer.degraded`` 并在答案里放可见提示。
#: 这里单独存一份、由引擎消费，两者互不干扰（同一个线程/上下文内）。
_ROUTE_DEGRADED: ContextVar[tuple[str, str, str] | None] = ContextVar(
    "legal_rag_llm_route_degraded", default=None
)

#: 最近一次 LLM 请求的完整指标字典（CLI 的 --metrics 摘要用）
_LAST: ContextVar[dict[str, Any] | None] = ContextVar(
    "legal_rag_llm_last_stats", default=None
)


# --------------------------------------------------------------------------
# 降级标记
# --------------------------------------------------------------------------

def mark_degraded(from_provider: str, to_provider: str) -> None:
    """标记「下一次调用是从 from_provider 降级到 to_provider 的」。"""
    _DEGRADED.set((from_provider or UNKNOWN, to_provider or UNKNOWN))


def take_degraded() -> tuple[str, str] | None:
    """取出并清空降级标记（一次性语义，避免污染后续请求）。"""
    value = _DEGRADED.get()
    if value is not None:
        _DEGRADED.set(None)
    return value


def mark_route_degraded(from_provider: str, to_provider: str, reason: str) -> None:
    """记录一次**路由级**降级（含可分类原因），供引擎填进 Answer / 响应。"""
    _ROUTE_DEGRADED.set((from_provider or UNKNOWN, to_provider or UNKNOWN,
                         reason or "unknown"))


def take_route_degraded() -> tuple[str, str, str] | None:
    """取出并清空路由级降级记录（一次性语义）。"""
    value = _ROUTE_DEGRADED.get()
    if value is not None:
        _ROUTE_DEGRADED.set(None)
    return value


#: **上下文裁剪**记录（§D7）：提示词超出服务窗口时客户端会裁掉部分证据再重试，
#: 这件事必须让用户看得见（"不许静默降级"），所以单独存一份给引擎写进答案。
_CONTEXT_TRIMMED: ContextVar[dict[str, Any] | None] = ContextVar(
    "legal_rag_llm_context_trimmed", default=None
)


def mark_context_trimmed(info: dict[str, Any]) -> None:
    """记录一次"因为超出上下文窗口而裁掉了部分证据"。"""
    _CONTEXT_TRIMMED.set(dict(info or {}))


def take_context_trimmed() -> dict[str, Any] | None:
    """取出并清空上下文裁剪记录（一次性语义）。"""
    value = _CONTEXT_TRIMMED.get()
    if value is not None:
        _CONTEXT_TRIMMED.set(None)
    return value


def last_stats() -> dict[str, Any] | None:
    """最近一次 LLM 请求的指标（无则返回 None）。"""
    return _LAST.get()


def current_role_id() -> str:
    """当前请求上下文里的 role_id（observability.bind_request 绑定），缺省 ``-``。"""
    try:
        return current_request().role_id or UNKNOWN
    except Exception:  # pragma: no cover - 上下文异常绝不影响业务
        return UNKNOWN


def summarize_error_text(text: str, limit: int = 300) -> str:
    """把**已经存下来的错误文本**单行化 + 脱敏 + 截断（t121 F2 的「截断的异常摘要」）。

    ``LLMStats.error`` 是字符串（不是异常对象），所以不能直接用
    :func:`legal_rag.generate.llm_base.summarize_exception`；这里复用它的脱敏规则。
    """
    single = " ".join((text or "").split())
    single = redact_secrets(single)
    return single if len(single) <= limit else single[:limit] + "...(截断)"


# --------------------------------------------------------------------------
# 请求级统计对象
# --------------------------------------------------------------------------

@dataclass
class LLMStats:
    """一次 LLM 请求的原始统计（**必须**是请求级局部对象，不得放到 client 实例上）。"""

    provider: str = ""
    model: str = ""
    role_id: str = ""
    stream: bool = False

    ok: bool = True
    degraded: bool = False
    degraded_from: str = ""
    degraded_to: str = ""
    error: str = ""
    #: 新口径失败原因（t121 F2，见 ``llm_base.failure_reason``）；
    #: ``error_reason`` 是旧口径，两者都保留，旧值有测试钉住。
    failure_reason: str = ""
    error_reason: str = ""
    attempts: int = 1

    prompt_chars: int = 0
    output_chars: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0

    ttft_seconds: float = 0.0
    total_seconds: float = 0.0
    load_seconds: float = 0.0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    ollama_total_seconds: float = 0.0

    tokens_per_second_official: float = 0.0
    tokens_per_second_wall: float = 0.0

    #: 请求发出前模型是否已在 ``/api/ps`` 中（True/False/None=探测失败）
    loaded_before: bool | None = None
    keep_alive: str = ""

    #: 内部：保证 record 只记一次
    recorded: bool = field(default=False, repr=False)

    # ---------- 派生 ----------

    @property
    def cold_start(self) -> bool | None:
        """是否冷启动：``/api/ps`` 说没加载 + ``load_duration`` 有值 → 冷启动。"""
        if self.loaded_before is not None:
            return not self.loaded_before
        if self.load_seconds > 1.0:
            return True
        return None

    def finalize(self) -> "LLMStats":
        """补齐两种口径的 tok/s 与非流式的 TTFT 语义。"""
        # 非流式：没有「第一个 token」这个概念，按约定等于总耗时
        if not self.stream:
            self.ttft_seconds = self.total_seconds
        if self.tokens_per_second_official <= 0.0 and self.decode_seconds > 0:
            self.tokens_per_second_official = self.output_tokens / self.decode_seconds
        if self.tokens_per_second_wall <= 0.0:
            window = self.total_seconds - self.ttft_seconds
            if window <= 1e-6:      # 非流式（ttft == total）退化为整段耗时
                window = self.total_seconds
            if window > 0 and self.output_tokens > 0:
                self.tokens_per_second_wall = self.output_tokens / window
        return self

    @property
    def labels(self) -> dict[str, Any]:
        return {
            "provider": self.provider or UNKNOWN,
            "model": self.model or UNKNOWN,
            "role_id": self.role_id or UNKNOWN,
            "stream": "true" if self.stream else "false",
        }

    def to_dict(self) -> dict[str, Any]:
        """扁平字典：既进 METRIC JSON 汇总行，也供 CLI 摘要打印。"""
        self.finalize()
        return {
            "provider": self.provider or UNKNOWN,
            "model": self.model or UNKNOWN,
            "role_id": self.role_id or UNKNOWN,
            "stream": self.stream,
            "ok": self.ok,
            "degraded": self.degraded,
            "degraded_from": self.degraded_from,
            "degraded_to": self.degraded_to,
            "error": self.error or None,
            "error_reason": self.error_reason or None,
            "failure_reason": self.failure_reason or None,
            "attempts": self.attempts,
            "prompt_chars": self.prompt_chars,
            "output_chars": self.output_chars,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "ttft_seconds": round(self.ttft_seconds, 6),
            "total_seconds": round(self.total_seconds, 6),
            "load_seconds": round(self.load_seconds, 6),
            "prefill_seconds": round(self.prefill_seconds, 6),
            "decode_seconds": round(self.decode_seconds, 6),
            "ollama_total_seconds": round(self.ollama_total_seconds, 6),
            "tokens_per_second_official": round(self.tokens_per_second_official, 4),
            "tokens_per_second_wall": round(self.tokens_per_second_wall, 4),
            "cold_start": self.cold_start,
            "loaded_before": self.loaded_before,
            "keep_alive": self.keep_alive,
        }


# --------------------------------------------------------------------------
# 采集与输出
# --------------------------------------------------------------------------

def _observe(stats: LLMStats, *, skip: Iterable[str] = ()) -> None:
    """把一次请求的原始统计登记进 histogram。

    :param skip: 本次请求中**协议上根本取不到**的指标名：跳过登记而不是写 0。
        写 0 会被 Prometheus 当成「一次 0 秒的加载/预填充」，把 P50/P95 拉成假值；
        跳过则让样本数只统计真实可观测的请求（默认空集合 = 该机制对既有调用方
        不生效）。

    注意：下面 tok/s 的 ``> 0`` 守卫是**全局**的（不在 ``skip`` 机制里），
    因此 ``ollama`` 路径在「算不出 tok/s」的边界场景（如 ``output_tokens=0``）
    也会从「登记一个 0」变成「不登记样本」——这是同一类 bug 的有意修复，
    不是「行为完全不变」。
    """
    labels = stats.labels
    histogram = M.histogram
    skip = frozenset(skip)

    # 失败的请求只记总耗时，避免 0 值污染 ttft / tok/s 的分位数
    histogram("llm_total_seconds").observe(stats.total_seconds, **labels)
    if stats.ok:
        for name, value in (
            ("llm_ttft_seconds", stats.ttft_seconds),
            ("llm_prompt_tokens", stats.prompt_tokens),
            ("llm_output_tokens", stats.output_tokens),
            ("llm_prefill_seconds", stats.prefill_seconds),
            ("llm_decode_seconds", stats.decode_seconds),
            ("llm_load_seconds", stats.load_seconds),
        ):
            if name not in skip:
                histogram(name).observe(value, **labels)
        if "llm_tokens_per_second" not in skip:
            # tok/s 只在「真的有可测窗口」时才登记：official 需要真实解码时长（decode>0），
            # wall 需要非零产出。算不出来时 **不写样本**，而不是写 0 ——
            # 写 0 会被 Prometheus 当成「一次真实的 0 token/s」，把 P50/P95 拉成假值。
            # 典型场景：OpenAI 兼容路径的**非流式**请求，ttft ≡ total ⇒ decode ≡ 0，official 导不出。
            if stats.tokens_per_second_official > 0:
                histogram("llm_tokens_per_second").observe(
                    stats.tokens_per_second_official, **{**labels, "mode": "official"})
            if stats.tokens_per_second_wall > 0:
                histogram("llm_tokens_per_second").observe(
                    stats.tokens_per_second_wall, **{**labels, "mode": "wall"})

    if not stats.ok:
        reason = stats.error_reason or "unknown"
        if reason == "timeout":
            M.counter("llm_timeout_total").inc(1, **labels)
        M.counter("llm_error_total").inc(1, **{**labels, "reason": reason})
        # t121 F2：新口径的失败原因（missing_api_key / connect_failed / http_5xx /
        # http_4xx / timeout …）+ 截断的异常摘要，一条日志就把"哪一类、哪一句"钉住。
        failed_reason = stats.failure_reason or "unknown"
        M.counter("llm_failure_total").inc(1, **{**labels, "reason": failed_reason})
        logger.warning("[LLM-FAIL] provider=%s model=%s reason=%s summary=%s",
                       labels.get("provider", UNKNOWN), labels.get("model", UNKNOWN),
                       failed_reason, summarize_error_text(stats.error))
    # 注：llm_degraded_total 由 ModelRouter 记录（只有它知道 from/to 全貌）


def _log_done(stats: LLMStats, log: logging.Logger,
              extra: dict[str, Any] | None = None) -> None:
    if stats.degraded:
        path = f"{stats.degraded_from}->{stats.degraded_to}"
    else:
        path = None
    log_event(
        "LLM_DONE",
        logger=log,
        provider=stats.provider or UNKNOWN,
        model=stats.model or UNKNOWN,
        role_id=stats.role_id or UNKNOWN,
        stream=stats.stream,
        ok=stats.ok,
        degraded=stats.degraded,
        degrade_path=path,
        prompt_chars=stats.prompt_chars,
        output_chars=stats.output_chars,
        elapsed=f"{stats.total_seconds:.2f}s",
        server_total=f"{stats.ollama_total_seconds:.2f}s",   # Ollama total_duration
        ttft=f"{stats.ttft_seconds:.2f}s",
        tok_s_official=round(stats.tokens_per_second_official, 2),
        tok_s_wall=round(stats.tokens_per_second_wall, 2),
        prompt_tokens=stats.prompt_tokens,
        output_tokens=stats.output_tokens,
        # 冷启动三件套：单列 load，才能区分「加载模型」与「真的慢」
        load=f"{stats.load_seconds:.2f}s",
        cold_start=stats.cold_start,
        keep_alive=stats.keep_alive or None,
        prefill=f"{stats.prefill_seconds:.2f}s",
        decode=f"{stats.decode_seconds:.2f}s",
        attempts=stats.attempts,
        error=stats.error or None,
        reason=stats.error_reason or None,
        # 协议侧本身取不到的指标（如 OpenAI 协议没有 load/prefill）由调用方显式声明，
        # 让「缺样本」在日志里也看得见，而不是安静地什么都不产出。
        metrics_unavailable=(extra or {}).get("metrics_unavailable") or None,
        metrics_derived=(extra or {}).get("metrics_derived") or None,
    )


def record_llm_stats(stats: LLMStats, logger: logging.Logger | None = None,
                     emit: bool = True, *,
                     skip: Iterable[str] = (),
                     extra: dict[str, Any] | None = None,
                     force: bool = False) -> dict[str, Any]:
    """一次 LLM 请求结束时调用：登记指标 + 打汇总日志 + 打 METRIC JSON 行。

    幂等（``stats.recorded``）；**绝不抛异常**——指标系统自身的故障
    不能影响业务链路的返回。

    :param skip: 本次请求中**协议上取不到**的指标名（跳过登记，不写 0 值）。
        默认空集合，既有调用方行为不变。
    :param extra: **可选**的附加只读信息，只影响日志/METRIC 行的可读性，
        不改变任何既有字段与指标口径。当前只有 ``openai_compat`` 用它声明
        「本协议取不到哪些量」：``metrics_unavailable`` / ``metrics_derived`` /
        ``token_source``。
    :param force: 已登记过的 stats 也重新输出日志与 METRIC 行（**不重复登记**）。
        给「先登记、拿到 payload 后再补元信息输出」的调用方用
        （``openai_compat`` 需要先把不可得字段改成 ``None`` 再输出）。
        注意：带 ``extra`` 再次调用时同样会重新输出（否则日志里看不到附加信息），
        但指标**绝不重复登记**。
    :param emit: 是否输出 ``[LLM_DONE]`` 汇总日志 + METRIC JSON 行（默认 True）。
        **``[LLM_DONE]`` 与 METRIC 行同生同灭**：二者都只在 ``emit`` 为真时写出。
        历史坑：``_log_done`` 曾在 ``if emit:`` **之外**被无条件调用，于是
        ``openai_compat._finish`` 的「先 ``emit=False`` 登记、再 ``force=True`` 输出」
        两遍调用让每个请求都打 **2 条** ``[LLM_DONE]``（断连时更多）——
        ELK/Loki 按行计数会因此翻倍。``ollama`` 路径用的是默认 ``emit=True``，
        行为与修复前逐字一致。
    """
    log = logger or globals()["logger"]
    stats.finalize()
    payload = stats.to_dict()
    if extra:
        # 附加只读字段与原始字段同层（dashboard 读 extra.<字段> 的习惯不变）
        payload = {**payload, **extra}
    if not stats.recorded:
        stats.recorded = True
        try:
            _observe(stats, skip=skip)
        except Exception:  # pragma: no cover - 指标登记失败不影响返回
            log.warning("LLM 指标登记失败（业务继续）", exc_info=True)
        emit = emit or force
    elif not force and extra is None:
        return payload

    if emit:
        try:
            _log_done(stats, log, extra=extra)
        except Exception:  # pragma: no cover
            log.warning("LLM 汇总日志写出失败", exc_info=True)
    try:
        _LAST.set(payload)
    except Exception:  # pragma: no cover
        pass
    if emit:
        try:
            # 一条 METRIC JSON 行 = 本次请求的全部字段（extra）+ 累计分布（metrics）
            M.emit_request_metrics(logger=log, extra=payload, only="llm_")
        except Exception:  # pragma: no cover
            log.warning("LLM METRIC 汇总行写出失败", exc_info=True)
    return payload


def log_llm_start(stats: LLMStats, *, base_url: str = "", messages: int = 0,
                  logger: logging.Logger | None = None) -> None:
    """调用前的入口日志：只记元信息，**绝不记 prompt 全文**。"""
    log_event(
        "LLM_CALL",
        logger=logger or globals()["logger"],
        provider=stats.provider or UNKNOWN,
        model=stats.model or UNKNOWN,
        role_id=stats.role_id or UNKNOWN,
        stream=stats.stream,
        prompt_chars=stats.prompt_chars,
        messages=messages,
        base_url=base_url or None,
        keep_alive=stats.keep_alive or None,
        loaded_before=stats.loaded_before,
    )


@contextmanager
def timed_stats(stats: LLMStats) -> Iterator[LLMStats]:
    """with 形式包裹一次请求：退出时自动写 ``total_seconds`` 并记录指标。"""
    import time
    started = time.perf_counter()
    try:
        yield stats
    finally:
        stats.total_seconds = time.perf_counter() - started
        record_llm_stats(stats)
