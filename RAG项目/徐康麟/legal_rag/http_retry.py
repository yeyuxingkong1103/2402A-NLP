# -*- coding: utf-8 -*-
"""HTTP 重试与失败分类的**公共策略**（嵌入侧与生成侧共用）。

为什么要有这个模块
------------------
在 2026-09-29 的重构盘点里发现：本项目在**两个地方各写了一套** HTTP 重试与失败
分类逻辑，语义高度重复但细节已经漂移：

======================================  ====================  ========================
                                        ``embedding/_client``  ``generate/openai_compat``
======================================  ====================  ========================
可重试状态码                             ``RETRYABLE_STATUS``   ``_REASON_BY_STATUS``
不可重试分类                             非本集合的 4xx 立即失败  ``_NON_RETRYABLE_REASONS``
退避公式                                 ``backoff * 2**(n-1)``  ``backoff * attempt``
重试循环                                 手写 ``for``           手写 ``while``
======================================  ====================  ========================

两套实现导致「同一类失败在嵌入侧可重试、在生成侧不可重试」这类**隐性不一致**，
而重试策略是**实测标定过的业务规则**（见下），不该有两份。

本模块**只提供策略判断**，不替调用方发请求、不替它打日志、不替它决定降级 ——
传输仍由各自客户端负责（httpx / urllib）。这样：
* 语义集中在一处，改一次两边生效；
* 调用方保留自己的可读日志与降级可见性（本项目的一贯要求）；
* 不引入新依赖。

⚠️ 注意：这些常量是**实测标定**的结果，不是随手抄的默认值 —— 修改前请先读
``docs/REFACTOR-PLAN.md`` 与对应客户端的注释。
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "RETRYABLE_STATUS", "NON_RETRYABLE_REASONS", "RETRYABLE_REASONS", "STATUS_REASON",
    "reason_of", "is_retryable_reason", "is_retryable_status",
    "should_retry", "backoff_seconds", "exponential_backoff",
]

#: 值得重试的 HTTP 状态码：服务端瞬时问题（连接被拒 / 过载 / 网关抖动）。
#: ⚠️ **实测标定**，由 ``embedding/_client.py`` 的 ``RETRYABLE_STATUS`` 迁移而来。
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

#: HTTP 状态码 → 机器可读的失败原因。
#:
#: ⚠️ **取值是冻结契约**（迁移时逐条对齐现有代码与测试，不得随意改名）：
#: ``tests/test_openai_compat_failure_policy.py`` 断言 ``missing_api_key`` / ``timeout``、
#: ``tests/test_ollama_llm.py`` 断言 ``_classify(...) == "connect"`` / ``"model_missing"``。
#: 这些值会作为指标标签 ``llm_error_total{reason=...}`` 上报，改名会打断既有告警与看板。
#: 未列出的 4xx → ``"http_4xx"``、5xx → ``"http_5xx"``（沿用原 ``_classify`` 的默认）。
STATUS_REASON: dict[int, str] = {
    400: "bad_request",
    401: "auth",
    403: "auth",
    404: "model_missing",
    408: "timeout",
    429: "rate_limit",
}

#: 这些原因**重试没有意义**（请求本身或配置有问题，再试一次结果一样）。
#: ``missing_api_key`` 属**配置**问题：不补 key 重试多少次都一样（t121 F2 的结论）。
NON_RETRYABLE_REASONS: frozenset[str] = frozenset({
    "model_missing", "bad_request", "auth", "missing_api_key", "bad_response",
})

#: 没有 HTTP 状态码、但**明确值得重试**的原因（瞬时网络/过载类）。
#: 这两类是重试的主要目标：它们往往连响应都没拿到。
RETRYABLE_REASONS: frozenset[str] = frozenset({
    "timeout", "connect", "rate_limit", "server_error", "unavailable",
    "bad_gateway", "gateway_timeout", "http_5xx",
})


def reason_of(
    *,
    status: int = 0,
    is_timeout: bool = False,
    is_connect_error: bool = False,
    reason: str = "",
) -> str:
    """把一次失败归一成机器可读的 ``reason``（**取值与既有代码逐字一致**）。

    优先级：已分类的 ``reason`` > HTTP 状态码 > 超时 > 连接错误 > ``"unknown"``。

    :param status: HTTP 状态码（0 表示没有拿到响应）；
    :param is_timeout: 是否为超时（读写超时）；
    :param is_connect_error: 是否为连接层失败（拒绝连接 / DNS / TLS）。
    """
    if reason:
        return reason
    if status:
        mapped = STATUS_REASON.get(int(status))
        if mapped:
            return mapped
        return "http_4xx" if 400 <= int(status) < 500 else "http_5xx"
    if is_timeout:
        return "timeout"
    if is_connect_error:
        # 取值 "connect"（不是 "connect_failed"）：与 tests/test_ollama_llm.py 的
        # `_classify(ConnectionRefusedError(...)) == "connect"` 保持一致。
        return "connect"
    return "unknown"


def is_retryable_status(status: int) -> bool:
    """该 HTTP 状态码是否值得重试。"""
    return int(status) in RETRYABLE_STATUS


def is_retryable_reason(reason: str) -> bool:
    """该失败原因是否值得重试（``NON_RETRYABLE_REASONS`` 里的一律不重试）。"""
    return reason not in NON_RETRYABLE_REASONS


def should_retry(
    *,
    status: int = 0,
    reason: str = "",
    is_timeout: bool = False,
    is_connect_error: bool = False,
) -> bool:
    """综合判断一次失败是否值得重试。

    规则（**先排除"重试无意义"，再判断是否瞬时故障**）：

    1. 原因属于 :data:`NON_RETRYABLE_REASONS` ⇒ **不重试**（400 参数错、404 模型不存在、
       缺 API key 等，重试纯属浪费时间）；
    2. 状态码在 :data:`RETRYABLE_STATUS` 内 ⇒ 重试；
    3. 超时 / 连接层失败（**没有状态码**，恰恰最该重试）⇒ 重试；
    4. 原因在 :data:`RETRYABLE_REASONS` 内 ⇒ 重试；
    5. 其余（既无状态码、又无法归类为瞬时故障）⇒ 保守**不重试**。
    """
    resolved = reason_of(status=status, is_timeout=is_timeout,
                         is_connect_error=is_connect_error, reason=reason)
    if not is_retryable_reason(resolved):
        return False
    if is_timeout or is_connect_error:
        return True
    if status:
        return is_retryable_status(status)
    return resolved in RETRYABLE_REASONS


def exponential_backoff(attempt: int, base: float = 1.0, *, cap: float = 0.0) -> float:
    """指数退避：第 ``attempt`` 次重试（从 1 起）等待 ``base * 2**(attempt-1)`` 秒。

    :param cap: >0 时对结果封顶（防止重试次数大时等待过久）。
    """
    if attempt < 1:
        return 0.0
    wait = max(float(base), 0.0) * (2 ** (attempt - 1))
    if cap and cap > 0:
        wait = min(wait, float(cap))
    return wait


def backoff_seconds(policy: Any, attempt: int, *, cap: float = 0.0) -> float:
    """按某个"重试配置对象"算退避。

    ``policy`` 只需有 ``retry_backoff`` 与可选的 ``retry_linear`` 属性：

    * ``retry_linear`` 为真 ⇒ **线性**退避 ``backoff * attempt``
      （``openai_compat`` 的原语义，迁移时保持一致以免改变既有重试节奏）；
    * 否则 ⇒ **指数**退避 ``backoff * 2**(attempt-1)``（``_client`` 的原语义）。

    ⚠️ 两种公式**刻意都保留**：它们是各自实测标定的结果，统一成一种会改变
    既有重试节奏（等于悄悄改了行为）。要统一请单独一轮做，并给出对比读数。
    """
    base = float(getattr(policy, "retry_backoff", 0.0) or 0.0)
    if base <= 0:
        return 0.0
    if getattr(policy, "retry_linear", False):
        wait = base * max(int(attempt), 0)
    else:
        wait = exponential_backoff(attempt, base)
    if cap and cap > 0:
        wait = min(wait, float(cap))
    return wait
