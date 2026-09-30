# -*- coding: utf-8 -*-
"""OpenAI 兼容协议客户端（纯标准库 urllib 实现，无需 openai SDK）。

可用于豆包 / 硅基流动 / 千问，也可指向本地 vLLM / SGLang / Xinference 的
OpenAI 兼容端点 —— 后续把模型换成 Qwen3.8-27B 时只需改 base_url 与 model。

安全约定：API key 只从环境变量读取，绝不写死在代码里，也绝不写进日志。

LLM 埋点：哪些量取得到、哪些取不到
----------------------------------
本协议（``POST /chat/completions``）**只上报 token 用量，不上报服务端耗时**，
因此这里的口径与 :mod:`legal_rag.generate.ollama` 不同，必须如实区分：

=================================  ===================================  ==========
指标                                本协议的来源                          可得性
=================================  ===================================  ==========
``llm_total_seconds``              客户端墙钟（``time.perf_counter``）     ✅ 可得
``llm_ttft_seconds``               流式：首个内容 chunk 到达时刻           ✅ 可得
                                   （非流式：约定为总耗时，标               （流式）
                                   ``stream=false``）                      ⚠️ 退化
``llm_prompt_tokens``              ``usage.prompt_tokens``                ⚠️ 仅当有 usage
``llm_output_tokens``              ``usage.completion_tokens``            ⚠️ 同上；
                                   （流式无 usage 时退化为「内容 chunk     降级=客户端计数
                                   数」，见下）                            （已标注）
``llm_decode_seconds``             **导出量**：``total - ttft``            ⚠️ 导出，非上报
``llm_tokens_per_second``          ``output_tokens / decode``             ⚠️ 导出
``llm_prefill_seconds``            协议**没有** ``prompt_eval_duration``     ❌ 不可得
``llm_load_seconds``               协议**没有** ``load_duration``           ❌ 不可得
=================================  ===================================  ==========

「取不到」怎么表达
------------------
本模块**绝不写假数字**：``llm_prefill_seconds`` / ``llm_load_seconds``
（以及非流式下的 ``llm_decode_seconds``）在本协议下不可得，于是

* **不往这两个 histogram 里写样本** —— 写入 ``0.0`` 会被 Prometheus 当成
  「一次 0 秒的预填充/加载」，把 P50/P95 拉低成假值。样本数（``_count``）
  因此只统计**真实可观测**的请求，这是刻意的；
* METRIC JSON 行 / LLM_DONE 日志里带上
  ``metrics_unavailable`` / ``metrics_derived`` 清单，让「缺样本」一眼可见，
  而不是安静地什么都不产出；
* 客户端首次调用时打一条 WARNING（每个实例一次），明确说出缺哪几项、
  要这些量请改用 ``LLM_PROVIDER=ollama`` 的原生 ``/api/chat``。

注意：``LLMStats`` 的字段本身**没有**「不可得」的表示法（默认 0.0），
所以上面的区分由本模块自己维护（:func:`unavailable_fields`）；**没有**改动
:mod:`legal_rag.generate.llm_metrics` 的既有字段名与对外语义，只给
:func:`~legal_rag.generate.llm_metrics.record_llm_stats` 增加了三个**默认关闭**的
可选通道（``skip`` / ``extra`` / ``force``，默认空集合/None/False）。

对 ``ollama`` 路径的影响（据实说明，勿写成「完全不变」）：``ollama`` 的两个调用点
（``_complete`` / ``stream``）**一行未改**，三个新通道默认不生效；此外
``llm_metrics._observe`` 里 tok/s 的 ``> 0`` 守卫是**全局**的 —— 它会让
ollama 路径在「算不出 tok/s」的边界场景（``output_tokens=0`` 等）从「登记一个 0」
变成「不登记样本」。这是**有意的修复**（0 会污染 P50/P95），不是回归，但
「行为完全不变」的说法与代码不符。

流式 token 用量
---------------
OpenAI 协议里 ``stream=true`` 默认**不回** ``usage``（vLLM/SGLang 需要
``stream_options={"include_usage": true}``，由 ``request_usage=True`` 默认发出）。
因此：

* 拿到 usage → 用**真实** ``completion_tokens`` / ``prompt_tokens``；
* 拿不到 usage → ``output_tokens`` 退化为「内容 chunk 数」（协议约定一个
  内容 chunk 携带一个 token），并在日志/指标里标注为**客户端计数**；
  而 ``prompt_tokens`` 无从得知，保持「不可得」而不是编一个数。

环境变量口径（**云上部署必读**，t121 起生效）
----------------------------------------------
云上把生成后端换成 vLLM / SGLang 时，本项目侧只认这三个变量（见
``docs/CLOUD-DEPLOY.md`` 与本模块的 ``_payload`` / ``_key``）：

====================  ============================================================
``LLM_PROVIDER``      必须写 ``openai_compat``（写 ``vllm``/``sglang`` 也走本模块，
                      但**配置里不要另写一套模型名**）
``LLM_MODEL``         **必填**，且必须与端点上的 ``--served-model-name`` 完全一致，
                      否则端点会回 404 -> 归一成 ``model_missing``
``OPENAI_COMPAT_BASE_URL``  例 ``http://<host>:30000/v1``（``/v1`` 不能省）
``OPENAI_API_KEY``    **必填**：本地 vLLM/SGLang 不校验它的值，但本客户端**没 key
                      会直接抛错**（归为 ``missing_api_key``）
``OPENAI_COMPAT_DISABLE_THINKING``  默认 ``1``（关思考）。Qwen3.x 这类带思考模式的模型
                      不关思考时，把「自述推理」直接当正文流出来（真机实测：``content``
                      开头是「用户问：…需要一句话回答…」），用户看到的就是草稿。
                      个别网关不认 ``chat_template_kwargs`` 字段时设 ``0`` 退回旧行为。
====================  ============================================================

失败与兜底口径（t120/t121）
---------------------------
* 缺 key / 连不上 / 5xx / 超时 ⇒ 分别归成 ``missing_api_key`` / ``connect_failed`` /
  ``http_5xx`` / ``http_4xx`` / ``timeout``（见
  :func:`legal_rag.generate.llm_base.failure_reason`），并写进
  ``llm_failure_total{provider,model,reason}`` 与 ``[LLM-FAIL]`` 日志（带截断摘要）；
* **不再静默兜底到 mock**：``LLM_PROVIDER=openai_compat`` 失败时，默认兜底链**不含 mock**，
  调用会抛 :class:`~legal_rag.generate.llm_base.LLMError`（带 ``reason``），由 API 层映射成
  **503 + 可读原因 + Retry-After**；
* 若要兜底，必须**显式**设 ``LLM_FALLBACKS=mock`` —— 此时响应会带
  ``provider=mock`` / ``degraded=true`` / ``degraded_reason``，答案里也会加一句用户可见提示。
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

# 公共 HTTP 重试与失败分类策略（与嵌入侧 embedding/_client 共用同一份）。
# 别名保留原名，既不破坏既有 import，也让"取值是契约"这件事在 import 处就能看见。
from ..http_retry import NON_RETRYABLE_REASONS as _NON_RETRYABLE_REASONS
from ..http_retry import STATUS_REASON as _REASON_BY_STATUS

from .llm_base import (
    FAILURE_MISSING_API_KEY, LLMClient, LLMError, failure_reason,
)
from .budget import is_context_overflow, trim_ladder
from .llm_metrics import (
    UNKNOWN, LLMStats, current_role_id, log_llm_start, mark_context_trimmed,
    record_llm_stats, take_degraded,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OpenAICompatClient",
    "OpenAICompatError",
    "unavailable_fields",
    "derived_fields",
    "OPENAI_COMPAT_UNAVAILABLE_FIELDS",
]

#: 本协议**结构上就取不到**的指标（OpenAI 没有 load_duration / prompt_eval_duration）。
#:
#: 之所以要显式列出来：这两个量在 ollama 原生 ``/api/chat`` 下是有的，
#: 用户切到 vLLM/SGLang 后它们会「安静地消失」——不报错、不告警。
#: 与其写 0 冒充样本，不如如实声明不可得（见模块 docstring）。
OPENAI_COMPAT_UNAVAILABLE_FIELDS: tuple[str, ...] = (
    "llm_load_seconds",
    "llm_prefill_seconds",
)

# --------------------------------------------------------------------------
# 失败分类与可重试性：**共用公共策略**，本模块不再自带一份
# --------------------------------------------------------------------------
# 2026-09-29 迁移：原先这里自定义了 `_REASON_BY_STATUS` 与 `_NON_RETRYABLE_REASONS`，
# 而 `embedding/_client.py` 另有一份语义重复的重试逻辑（含各自的退避公式）——
# 导致"同一类失败在嵌入侧可重试、在生成侧不可重试"这类隐性不一致。
# 现在两者都从 `legal_rag.http_retry` 取，别名见文件顶部 import。
#
# ⚠️ `_REASON_BY_STATUS` 的取值是**冻结契约**：它们是指标标签
# `llm_error_total{reason=...}` 的取值，改名会打断既有告警与看板。


class OpenAICompatError(LLMError):
    """OpenAI 兼容端点调用失败。

    带 ``reason`` 属性（与 :class:`~legal_rag.generate.ollama.OllamaError` 同形），
    供 :func:`_classify` 映射成指标标签 ``llm_error_total{reason=...}``。
    ``reason`` 只能是关键字参数，因此既有的 ``except LLMError`` 全部继续有效。
    """

    def __init__(self, message: str, *, reason: str = "unknown") -> None:
        super().__init__(message)
        self.reason = reason


def unavailable_fields(*, usage_present: bool,
                       decode_derivable: bool,
                       counted_by_chunks: bool = False) -> tuple[str, ...]:
    """本协议下**取不到**的指标名（不写假样本，只在日志/METRIC 行里声明）。

    :param usage_present: 响应里是否带回 ``usage``（token 用量的唯一来源）；
    :param decode_derivable: ``decode = total - ttft`` 是否成立
        （非流式下 ttft 恒等于 total，导出量恒为 0，因此视为不可得）；
    :param counted_by_chunks: 流式且已用「内容 chunk 数」做了客户端计数时，
        ``llm_output_tokens`` **是有值的**（精度低，但已标注 derived），
        这时不得再把它列为「不可得」。
    """
    fields = list(OPENAI_COMPAT_UNAVAILABLE_FIELDS)
    if not usage_present:
        # 没有 usage 就没有 token 用量的唯一来源：prompt_tokens 无从得知
        fields.append("llm_prompt_tokens")
        if not counted_by_chunks:
            # 「用量未知」≠「用量为 0」：既没有服务端上报、也没有做 chunk 计数时，
            # output_tokens 是**未知**的；写 0 会被 Prometheus 当成
            # 「这一次真的生成了 0 个 token」，把 P50/P95 往下拉（曾漏掉这一条）。
            fields.append("llm_output_tokens")
    if not decode_derivable:
        fields.append("llm_decode_seconds")
    return tuple(fields)


def derived_fields(*, decode_derivable: bool, usage_present: bool,
                   counted_by_chunks: bool) -> tuple[str, ...]:
    """本协议下「不是服务端上报、而是客户端导出/计数」的指标名。

    显式标注它们，避免被误读成 vLLM 上报的官方吞吐。

    本清单与 :func:`unavailable_fields` 的清单**互斥**：同一项不得既被声明
    「不可得」，又被标成「客户端有值」。特别注意 ``llm_prompt_tokens``：无 usage
    时它**根本没有值**，客户端也推算不出来，因此只应出现在 ``unavailable`` 里 ——
    早期版本在这里又把它标成 derived，导致同一条 METRIC 行自相矛盾。

    ``usage_present`` 仍留在签名里：调用方按「用量来源」显式传参，读起来自解释；
    但它**不再决定**任何 derived 项（derived 只由导出/计数决定）。
    """
    fields: list[str] = []
    if decode_derivable:
        fields.append("llm_decode_seconds")
        fields.append("llm_tokens_per_second")
    if counted_by_chunks:
        fields.append("llm_output_tokens")
    return tuple(fields)


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, LLMError) and getattr(exc, "reason", "") == "timeout":
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, urllib.error.URLError):
        # urllib 会把 ConnectionRefusedError 之类包成 URLError(reason=...)，
        # 必须继续看 reason，否则「连不上」会被误判成「超时」（与 ollama 路径同规则）
        return isinstance(exc.reason, (TimeoutError, socket.timeout)) or "timed out" in str(exc)
    return False


def _classify(exc: BaseException) -> str:
    """把异常归类成指标标签 ``reason``（与 ollama 路径同名同义）。"""
    reason = getattr(exc, "reason", "")
    if isinstance(exc, LLMError) and reason:
        return str(reason)
    if isinstance(exc, urllib.error.HTTPError):
        return _REASON_BY_STATUS.get(exc.code, "http")
    if _is_timeout(exc):
        return "timeout"
    if isinstance(exc, urllib.error.URLError):
        return "connect"
    if isinstance(exc, (ConnectionError, OSError)):
        return "connect"
    if isinstance(exc, json.JSONDecodeError):
        return "bad_response"
    return "unknown"


def _int_or_zero(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


@dataclass
class _PostStats:
    """非流式请求的**用量来源**（请求级局部对象，不挂在 client 上）。

    必须显式跟踪而不能靠「``output_tokens > 0``」反推：两者语义不同
    （服务端上报 vs 客户端计数），混用会把「没有 usage」误判成「有 usage」。
    """

    usage_present: bool = False


def _env_bool(name: str, default: bool) -> bool:
    """读布尔型环境变量：未设/空 → 默认；``1/true/yes/on`` → True；其余 → False。"""
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class _StreamStats:
    """流式请求的用量来源 + 内容 chunk 数（请求级局部对象）。

    记 ``content_chunks`` 而不是只记一个布尔量，是为了**客户端提前断开**时
    仍能在 ``finally`` 里如实登记「已经产出多少个 token」——生成器被 close 时
    正常的 ``return`` 不会执行，只有这个可变容器会被读到最后状态。

    ``finished`` 是**收尾去重**的兜底开关：一次请求只允许输出一次
    ``[LLM_DONE]`` + 一条 METRIC 行，重复调用 ``_finish_stream`` 会直接返回。
    """

    usage_present: bool = False
    counted_by_chunks: bool = False
    content_chunks: int = 0
    finished: bool = False


class OpenAICompatClient(LLMClient):
    name = "openai_compat"

    def __init__(self, model: str = "qwen", base_url: str = "",
                 api_key_env: str = "OPENAI_API_KEY", api_key: str | None = None,
                 temperature: float = 0.3, max_tokens: int = 1024,
                 timeout: float = 60.0, role_id: str = "",
                 request_usage: bool = True, count_chunks_as_tokens: bool = True,
                 max_retries: int = 0, retry_backoff: float = 0.0,
                 disable_thinking: bool | None = None) -> None:
        super().__init__(model=model, temperature=temperature,
                         max_tokens=max_tokens, timeout=timeout)
        self.base_url = (base_url or "http://127.0.0.1:8000/v1").rstrip("/")
        self.api_key_env = api_key_env
        self._api_key = api_key
        #: 显式 role_id（测试用）；默认从 observability 的请求上下文取
        self.role_id = role_id
        #: 流式时是否显式请求 token 用量（``stream_options.include_usage``）。
        #: vLLM / SGLang 支持；个别严格网关可能拒绝这个字段，置 False 即退回旧行为。
        self.request_usage = bool(request_usage)
        #: 流式拿不到 usage 时，是否用「内容 chunk 数」作为 output_tokens 的客户端计数
        self.count_chunks_as_tokens = bool(count_chunks_as_tokens)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        #: 是否关掉模型的「思考模式」（``chat_template_kwargs.enable_thinking=False``）。
        #: 真机实测（cloud 2026-09-21，Qwen3.5-27B-FP8 @ vLLM 0.29）：不关思考时模型把
        #: 自述推理直接放进 ``content`` 流出来（``reasoning_content`` 为空），用户看到的
        #: 是「用户问：…需要一句话回答…」这种草稿；关掉后正文才干净。
        #: 显式传参优先；否则读 ``OPENAI_COMPAT_DISABLE_THINKING``（默认开）。
        self.disable_thinking = (
            _env_bool("OPENAI_COMPAT_DISABLE_THINKING", True)
            if disable_thinking is None else bool(disable_thinking)
        )
        #: 一次性告警开关：每个实例只喊一次「本协议缺哪些指标」
        self._gap_warned = False
        # 显式禁用代理：本地 vLLM/SGLang 常挂在 127.0.0.1，走系统代理会拿到空响应
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # ---------- 内部 ----------
    def _key(self) -> str:
        key = self._api_key or os.environ.get(self.api_key_env, "")
        if not key:
            # t121 F2：**配置**缺失要能和"网络坏了"区分开 —— 归到 missing_api_key，
            # 路由器据此**不再往 mock 兜底**（否则用户会拿到 200 的假答案，见 t120 F1）。
            raise OpenAICompatError(
                f"缺少 API key：请设置环境变量 {self.api_key_env}。\n"
                f"（当前 base_url={self.base_url}；"
                f"若使用本地 vLLM/SGLang，可把 {self.api_key_env} 设为任意非空字符串）",
                reason=FAILURE_MISSING_API_KEY,
            )
        return key

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _payload(self, messages: list[dict], stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if stream and self.request_usage:
            # vLLM / SGLang / OpenAI 官方都认这个开关；开启后最后一个 chunk 会带 usage。
            # 没有它，流式路径的 token 用量只能靠客户端数 chunk（精度更低）。
            payload["stream_options"] = {"include_usage": True}
        if self.disable_thinking:
            # Qwen3.x 的思考模式：不关它，模型会把自述推理当正文流出来（真机实测）。
            # 这是 vLLM/SGLang 的扩展字段；不认它的网关用 OPENAI_COMPAT_DISABLE_THINKING=0 关掉。
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _request(self, payload: dict) -> urllib.request.Request:
        return urllib.request.Request(
            self._endpoint(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key()}",
            },
            method="POST",
        )

    def _open(self, payload: dict):
        """发起请求。

        用实例自带的 opener（禁用代理），**不是** ``urllib.request.urlopen``；
        单元测试请 patch ``client._opener.open``。
        """
        return self._opener.open(self._request(payload), timeout=self.timeout)

    def _error_message(self, exc: BaseException, stream: bool) -> str:
        return (
            f"OpenAI 兼容端点调用失败（base_url={self.base_url}, model={self.model}, "
            f"stream={str(stream).lower()}, timeout={self.timeout:g}s）："
            f"{type(exc).__name__}: {exc}\n"
            f"排查：1) 端点是否在跑：curl -s {self.base_url}/models "
            f"-H \"Authorization: Bearer ${self.api_key_env}\"；"
            f"2) model 是否是端点上的模型名（当前 {self.model}）；"
            f"3) 本地 vLLM/SGLang 需要 {self.api_key_env} 为任意非空值；"
            f"4) 连远端要设 OPENAI_COMPAT_BASE_URL（当前 {self.base_url}）。"
        )

    # ---------- 埋点 ----------
    def _new_stats(self, messages: list[dict], stream: bool) -> LLMStats:
        """构造请求级 stats（可变状态一律局部，绝不放 client 实例上）。"""
        stats = LLMStats(
            provider=self.name,
            model=self.model,
            role_id=self.role_id or current_role_id(),
            stream=stream,
            prompt_chars=sum(len(str(m.get("content") or "")) for m in messages),
        )
        degraded = take_degraded()
        if degraded is not None:
            stats.degraded = True
            stats.degraded_from, stats.degraded_to = degraded
        return stats

    @staticmethod
    def _apply_usage(stats: LLMStats, body: dict) -> bool:
        """从 ``usage`` 取 token 用量；返回是否**真的取到**（只有 >0 才算取到）。

        字段名以 OpenAI 协议为准：``usage.prompt_tokens`` / ``usage.completion_tokens``
        （``input_tokens`` / ``output_tokens`` 是 Responses API 的叫法，这里不做猜测）。
        """
        usage = body.get("usage")
        if not isinstance(usage, dict):
            return False
        prompt = _int_or_zero(usage.get("prompt_tokens"))
        completion = _int_or_zero(usage.get("completion_tokens"))
        if prompt <= 0 and completion <= 0:
            return False
        stats.prompt_tokens = prompt
        stats.output_tokens = completion
        return True

    @staticmethod
    def _record_decode(stats: LLMStats, ttft: float, total: float) -> bool:
        """把 ``decode = total - ttft`` 落到指标上；返回是否可导出。

        **这是客户端导出量，不是服务端上报量**（OpenAI 协议没有 ``eval_duration``）。
        非流式下 ``ttft`` 按约定等于总耗时，导出量恒为 0 —— 那样没有信息量，
        因此判定为「不可得」而不是记一个 0。
        """
        if not stats.stream or ttft <= 0 or total <= ttft:
            return False
        stats.decode_seconds = total - ttft
        return True

    @staticmethod
    def _throughput(stats: LLMStats, decode_derivable: bool) -> None:
        """两种口径都写进 stats（``finalize()`` 只在为空时才补，故这里先写）。

        * ``official`` —— ``output_tokens / decode_seconds``：导出量，纯解码速度；
        * ``wall``     —— ``output_tokens / total_seconds``：整次请求吞吐（含预填充/网络）。
          本协议拿不到 ``prompt_eval_duration``，所以这里刻意用**整段耗时**做分母，
          避免与 ``official`` 变成同一个数（ollama 那边两者真不一样，有信息量）。
        """
        if decode_derivable and stats.decode_seconds > 0 and stats.output_tokens > 0:
            stats.tokens_per_second_official = stats.output_tokens / stats.decode_seconds
        if stats.total_seconds > 0 and stats.output_tokens > 0:
            stats.tokens_per_second_wall = stats.output_tokens / stats.total_seconds

    def _mark_error(self, stats: LLMStats, exc: BaseException) -> None:
        """与 ollama 路径一致的错误埋点语义：ok=False + error + error_reason。

        ``error_reason`` 是**旧口径**（取值被既有测试钉住，不动）；
        ``failure_reason`` 是 t121 F2 的**新口径**（区分 missing_api_key / connect_failed /
        http_5xx / http_4xx / timeout），供 ``llm_failure_total`` 与日志使用。
        """
        stats.ok = False
        stats.error = f"{type(exc).__name__}: {exc}"
        stats.error_reason = _classify(exc)
        stats.failure_reason = failure_reason(getattr(exc, "__cause__", None) or exc)

    def _finish(self, stats: LLMStats, started: float, *,
                usage_present: bool = False, counted_by_chunks: bool = False) -> None:
        """统一收尾：补 total/导出量 → 记录指标（不可得的量显式声明）。

        关键点：``record_llm_stats`` 默认会把 ``llm_load_seconds`` /
        ``llm_prefill_seconds`` 也 observe 一遍（本协议下它们恒为 0），
        因此这里用 ``skip`` 把**结构性不可得**的项从观测集合里摘掉：
        不写 0 值假样本，也不改动 ``llm_metrics`` 的既有字段与对外语义。

        ``usage_present`` / ``counted_by_chunks`` 由调用方显式传入（不能靠
        「output_tokens > 0」反推：chunk 计数也会让它是正数，那是客户端计数而不是
        服务端上报的用量 —— 曾因此把「没有 usage」误判成「有 usage」）。

        **每个请求只允许调用一次**：指标登记本身有 ``stats.recorded`` 幂等保护，
        但日志/METRIC 行的输出（``force=True`` 那遍）每次调用都会重打一行 ——
        重复调用会让一次请求出现 2 条 ``[LLM_DONE]`` 与 2 条 METRIC 行。
        """
        stats.total_seconds = time.perf_counter() - started
        decode_derivable = self._record_decode(stats, stats.ttft_seconds,
                                               stats.total_seconds)
        if not stats.stream and stats.ttft_seconds <= 0:
            # 非流式：没有「首个 token」的时间点，按约定 TTFT = 总耗时（标 stream=false）
            stats.ttft_seconds = stats.total_seconds
        self._throughput(stats, decode_derivable)

        missing = unavailable_fields(usage_present=usage_present,
                                     decode_derivable=decode_derivable,
                                     counted_by_chunks=counted_by_chunks)
        if stats.stream and stats.ttft_seconds <= 0:
            # 流式**成功**但一个内容 chunk 都没到（模型空答案 / 端点只回 [DONE] 或
            # role delta）：不存在「首个 token 到达」这个时刻。此时 ttft 仍是 0.0，
            # 直接登记会被当成「一次瞬时的首 token」假样本 —— 与 load/prefill 同理，
            # 判定为不可得而不是写 0。
            missing = (*missing, "llm_ttft_seconds")
        derived = derived_fields(decode_derivable=decode_derivable,
                                 usage_present=usage_present,
                                 counted_by_chunks=counted_by_chunks)
        if not stats.ok:
            # 失败的请求只记 llm_total_seconds（llm_metrics 的既有约定），
            # 这里把结构性缺失项一并列出，便于对照日志排查
            missing = tuple(dict.fromkeys((*missing, *OPENAI_COMPAT_UNAVAILABLE_FIELDS)))
        skip = frozenset(missing) if stats.ok else frozenset()

        # 第一遍：登记指标（emit=False，避免用未清洗的 payload 输出 METRIC 行）
        payload = record_llm_stats(stats, logger=logger, emit=False, skip=skip)
        # 不可得的量在 METRIC JSON 行里显式写 None（而不是伪造的 0）：
        # 与样本侧一致，看板作者一眼能区分「0」与「没有这个量」。
        payload = dict(payload)
        for field in ("load_seconds", "prefill_seconds", "decode_seconds",
                      "ttft_seconds", "prompt_tokens", "output_tokens"):
            if f"llm_{field}" in missing:
                payload[field] = None
        if not decode_derivable:
            # official 吞吐需要真实解码窗口；非流式导不出 ⇒ 显式写 None，而不是伪造的 0。
            # （对应样本也不会登记，见 llm_metrics._observe 里的同名守卫。）
            payload["tokens_per_second_official"] = None
        payload["metrics_unavailable"] = list(missing)
        payload["metrics_derived"] = list(derived)
        # token 用量的来源：usage=服务端上报 / chunks=客户端数 chunk / none=未知
        payload["token_source"] = ("usage" if usage_present
                                   else ("chunks" if counted_by_chunks else "none"))
        # 第二遍：force=True 只输出日志 + METRIC 行（**不会重复登记指标**）
        record_llm_stats(stats, logger=logger, emit=True, extra=payload, force=True)
        self._warn_gap_once(stats, missing)

    def _warn_gap_once(self, stats: LLMStats, missing: tuple[str, ...]) -> None:
        """每个客户端实例只喊一次：本协议下哪几项指标**结构性**取不到。"""
        structural = [name for name in missing
                      if name in OPENAI_COMPAT_UNAVAILABLE_FIELDS]
        if self._gap_warned or not structural:
            return
        self._gap_warned = True
        logger.warning(
            "OpenAI 兼容协议（provider=%s）不提供 %s：这几项指标不会有样本，"
            "不是采集失败。需要区分「冷启动加载」与「真的慢」请改用 "
            "LLM_PROVIDER=ollama 的原生 /api/chat（它有 load_duration/prompt_eval_duration）。",
            stats.provider, "、".join(structural),
        )

    def _sleep(self, attempt: int) -> None:
        delay = self.retry_backoff * attempt
        if delay > 0:
            time.sleep(delay)

    def _retryable(self, exc: BaseException) -> bool:
        reason = _classify(exc)
        if reason in _NON_RETRYABLE_REASONS:
            # 模型不存在 / 参数错误 / 鉴权失败，重试没有意义
            return False
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code >= 500 or exc.code == 429
        return reason in ("connect", "timeout", "http", "rate_limit")

    # ---------- 上下文预算守卫（§D7）----------
    def _note_context_trim(self, ratio: float, info: dict) -> None:
        """裁剪要**留痕**：日志 + 指标 + 给引擎的可见提示（绝不静默）。"""
        logger.warning(
            "提示词超出服务窗口 ⇒ 按 %.0f%% 裁证据后重试：证据块 %d→%d 字（丢 %d 字），"
            "问题/记忆/历史一字不动（model=%s）",
            ratio * 100, info.get("block_chars_before"), info.get("block_chars_after"),
            info.get("dropped_chars"), self.model)
        self._metrics_counter(ratio)
        mark_context_trimmed(info)

    def _metrics_counter(self, ratio: float) -> None:
        try:
            from .. import metrics as M  # noqa: PLC0415 - 只在这里用，避免模块级循环导入

            M.counter("llm_context_trim_total").inc(stage=f"{int(ratio * 100)}")
        except Exception:  # noqa: BLE001 - 指标失败不该影响回答
            logger.debug("llm_context_trim_total 计数失败（忽略）", exc_info=True)

    # ---------- LLMClient 接口 ----------
    def _complete(self, messages: list[dict]) -> str:
        try:
            return self._complete_once(messages)
        except OpenAICompatError as exc:
            # 超窗 ⇒ **逐档裁证据重试**，而不是把 400 抛给用户（G10 的成因）
            for ratio, candidate, info in trim_ladder(messages):
                if not is_context_overflow(str(exc)):
                    raise
                self._note_context_trim(ratio, info)
                try:
                    return self._complete_once(candidate)
                except OpenAICompatError as retry_exc:
                    exc = retry_exc
                    continue
            raise

    def _complete_once(self, messages: list[dict]) -> str:
        stats = self._new_stats(messages, stream=False)
        log_llm_start(stats, base_url=self.base_url, messages=len(messages), logger=logger)
        started = time.perf_counter()
        post = _PostStats()
        try:
            body = self._post_chat(messages, stats)
            choices = body.get("choices") or []
            if not choices:
                raise OpenAICompatError(
                    f"响应中没有 choices：{str(body)[:300]}", reason="bad_response")
            text = str(choices[0].get("message", {}).get("content", ""))
            stats.output_chars = len(text)
            post.usage_present = self._apply_usage(stats, body)
            return text
        except Exception as exc:  # noqa: BLE001 - 统一转成带上下文的错误
            self._mark_error(stats, exc)
            if isinstance(exc, OpenAICompatError):
                raise
            raise OpenAICompatError(self._error_message(exc, stream=False),
                                    reason=_classify(exc)) from exc
        finally:
            self._finish(stats, started, usage_present=post.usage_present)

    def _post_chat(self, messages: list[dict], stats: LLMStats) -> dict:
        payload = self._payload(messages, stream=False)
        attempt = 0
        while True:
            try:
                with self._open(payload) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise OpenAICompatError(
                        f"响应不是合法 JSON（前 300 字符）：{raw[:300]}",
                        reason="bad_response") from exc
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                error: BaseException = OpenAICompatError(
                    f"HTTP {exc.code}: {detail}", reason=_classify(exc))
            except OpenAICompatError as exc:
                error = exc
            except Exception as exc:  # noqa: BLE001 - 网络/超时统一处理
                error = exc
            attempt += 1
            stats.attempts = attempt + 1
            if attempt > self.max_retries or not self._retryable(error):
                raise error
            logger.warning("OpenAI 兼容端点非流式调用失败，第 %d/%d 次重试（%s：%s）",
                           attempt, self.max_retries, type(error).__name__, error)
            self._sleep(attempt)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """真流式：``stream=true`` + SSE，逐块 yield 增量文本。

        生成器内的所有状态（stats / 累计字符 / 首 chunk 时刻 / 用量来源）都是
        **调用级局部变量**，因此同一个 client 被多个请求并发使用也不会串包。
        """
        stats = self._new_stats(messages, stream=True)
        log_llm_start(stats, base_url=self.base_url, messages=len(messages), logger=logger)
        started = time.perf_counter()
        stream_stats = _StreamStats()
        try:
            yield from self._stream_chunks(messages, stats, started, stream_stats)
        except GeneratorExit:          # 调用方提前停止（客户端断连）
            # 哪怕被放弃也要如实登记「已经产出了多少」（见 _StreamStats 的说明）。
            # 收尾**统一交给下面的 finally**：这里不要再调一次 _finish_stream，
            # 否则断连时会多打一条 METRIC 行与多条 [LLM_DONE]（重复行会污染
            # ELK/Loki 里的按行计数）。
            raise
        except OpenAICompatError as exc:   # 已经是带上下文的错误：只补记指标
            # 超窗且**一个 chunk 都还没产出**时才可安全重试（已经吐字了就不能重来，
            # 否则用户会看到重复/错乱的两段答案）。逐档裁证据重试。
            if stream_stats.content_chunks == 0 and is_context_overflow(str(exc)):
                for ratio, candidate, info in trim_ladder(messages):
                    self._note_context_trim(ratio, info)
                    try:
                        yield from self._stream_chunks(candidate, stats, started, stream_stats)
                        return
                    except OpenAICompatError as retry_exc:
                        exc = retry_exc
                        if not is_context_overflow(str(retry_exc)):
                            break
            self._mark_error(stats, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            self._mark_error(stats, exc)
            raise OpenAICompatError(self._error_message(exc, stream=True),
                                    reason=_classify(exc)) from exc
        finally:
            self._finish_stream(stats, started, stream_stats)

    def _finish_stream(self, stats: LLMStats, started: float,
                       stream_stats: _StreamStats) -> None:
        """流式收尾：chunk 计数降级 + 调用统一的 :meth:`_finish`。

        **只能生效一次**：``stream()`` 的收尾点只有 ``finally`` 一处
        （``except GeneratorExit`` 不再重复调用，见那里的注释）；
        这里的 ``finished`` 是兜底开关 —— 万一将来又出现第二个调用点，
        也不会重复输出日志/METRIC 行。
        """
        if stream_stats.finished:
            return
        stream_stats.finished = True
        if (not stream_stats.usage_present and not stream_stats.counted_by_chunks
                and stream_stats.content_chunks > 0 and self.count_chunks_as_tokens):
            # 服务端没回 usage（或被提前放弃）：内容 chunk 数 ≈ 生成 token 数
            # （协议约定 1 个内容 chunk = 1 个 token）。这是**客户端计数**，
            # 日志/指标里会标 metrics_derived。
            stats.output_tokens = stream_stats.content_chunks
            stream_stats.counted_by_chunks = True
        self._finish(stats, started, usage_present=stream_stats.usage_present,
                     counted_by_chunks=stream_stats.counted_by_chunks)

    def _stream_chunks(self, messages: list[dict], stats: LLMStats, started: float,
                       stream_stats: _StreamStats) -> Iterator[str]:
        payload = self._payload(messages, stream=True)
        attempt = 0
        while True:
            try:
                yield from self._read_stream(payload, stats, started, stream_stats)
                return
            except Exception as exc:  # noqa: BLE001 - 网络/超时/HTTP 统一处理
                attempt += 1
                stats.attempts = attempt + 1
                # 已经吐过 token 就不能重试（会重复输出）；未产出时允许重试
                if (stats.output_chars > 0 or attempt > self.max_retries
                        or not self._retryable(exc)):
                    raise
                logger.warning("OpenAI 兼容端点流式调用失败，第 %d/%d 次重试"
                               "（尚未产出 token，%s：%s）",
                               attempt, self.max_retries, type(exc).__name__, exc)
                self._sleep(attempt)

    def _read_stream(self, payload: dict, stats: LLMStats, started: float,
                     stream_stats: _StreamStats) -> Iterator[str]:
        """读 SSE：逐块 yield 文本，并把用量来源/内容块数记进 ``stream_stats``。"""
        try:
            response = self._open(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise OpenAICompatError(f"HTTP {exc.code}: {detail}",
                                    reason=_classify(exc)) from exc

        with response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue                      # 忽略心跳之类的噪声
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    raise OpenAICompatError(f"端点返回错误：{str(chunk['error'])[:300]}",
                                            reason="bad_response")
                if self._apply_usage(stats, chunk):
                    stream_stats.usage_present = True
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if not piece:
                        continue
                    if stats.output_chars == 0:
                        # 首 token 延迟：从「请求发出」到「第一个内容 chunk 到达」
                        stats.ttft_seconds = time.perf_counter() - started
                    stats.output_chars += len(piece)
                    stream_stats.content_chunks += 1
                    yield str(piece)
        # 注：这里**不做** chunk 计数降级 —— 生成器可能被调用方提前关闭（客户端断连），
        # 那时本行不会执行；降级统一放在 stream() 的 finally 里（见 _finish_stream）。

    def health(self) -> dict:
        return {
            "provider": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_present": bool(self._api_key or os.environ.get(self.api_key_env)),
            # 本协议下结构性不可得的指标（/health 一眼可见，避免「安静地缺样本」）
            "metrics_unavailable": list(OPENAI_COMPAT_UNAVAILABLE_FIELDS),
        }
