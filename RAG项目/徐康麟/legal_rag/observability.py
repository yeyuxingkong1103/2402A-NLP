# -*- coding: utf-8 -*-
"""可观测性地基：request 上下文、@traced 调用追踪、log_io 数据出入口、异常包装。

用户诉求原话
------------
「添加详细的日志：数据出入口、函数调用前后，并且多使用 try 来包裹函数。
  要实现：如果服务实现出错了，可以根据日志快速定位问题所在。」

本模块把「函数调用前后 + 数据出入口 + 异常上下文」做成三个可复用的原语，
让业务代码只需要一行装饰器 / 一行 with 就能获得可定位的日志：

======================  ==========================================================
原语                     作用
======================  ==========================================================
``bind_request(...)``    绑定 request_id（contextvars），此后**所有**日志自动带上；
``@traced``              函数入口/出口各一条：入参摘要、返回摘要、耗时 ms、异常栈；
``log_io(...)``          数据出入口：来源、去向、条数、字节数、耗时、失败原因；
``wrap_errors(...)``     异常包装成 :class:`TracedError`，保留 ``__cause__`` 原始栈。
======================  ==========================================================

配套约定
--------
* 敏感字段自动打码：键名含 ``key/token/secret/password/passwd/authorization/
  api_key/cookie`` 时值替换为 ``***``（见 :data:`SENSITIVE_KEY_PATTERN`）；
* 摘要一律截断（默认 200 字符、8 个元素），日志里绝不出现整篇文档；
* **绝不吞异常**：所有包装器都是 ``raise``（或改抛带上下文的异常），
  只新增日志，不改变控制流；
* 日志量控制：入口/出口走 DEBUG，异常走 ERROR/EXCEPTION（完整栈），
  高频路径可用 :func:`sampled` 或 ``@traced(sample=0.01)`` 采样。

耗时口径：``elapsed_ms`` 用 ``time.perf_counter()`` 计算，仅用于日志与指标。
"""
from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import os
import random
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator

__all__ = [
    "RequestContext", "bind_request", "current_request", "current_request_id",
    "new_request_id", "reset_request", "request_context", "attach_logging_context",
    "REQUEST_ID_FALLBACK",
    "summarize", "summarize_args", "mask_secrets", "SENSITIVE_KEY_PATTERN",
    "TracedError", "traced", "log_io", "wrap_errors",
    "log_event", "log_before", "log_after", "log_exception", "safe_call",
    "sampled", "sample_rate_for",
    # P0.5 新增：耗时评估（L1 直方图 / L2 慢调用清单 / L3 全量前后）
    "timed", "timed_scope", "log_trace", "trace_level", "slow_call_ms",
    "TRACE_LEVEL_ENV", "SLOW_CALL_MS_ENV", "SLOW_CALL_SAMPLE_ENV",
    "DEFAULT_SLOW_CALL_MS", "FUNC_SECONDS_METRIC", "SLOW_LOG_TAG",
]

_LOGGER = logging.getLogger("legal_rag.observability")

REQUEST_ID_FALLBACK = "-"

# --------------------------------------------------------------------------
# P0.5：耗时评估的环境变量与默认值
# --------------------------------------------------------------------------

#: L3「全量函数前后日志」的级别来源。
#: **为什么需要它**：``@traced`` 默认 ``level=DEBUG``，而 ``LOG_LEVEL`` 默认 ``INFO``
#: ⇒ 函数入口/出口与「耗时 X.Xms」这一行会被**完全过滤掉**（这是本项目
#: 「有耗时能力却看不见耗时」的根因）。``TRACE_LEVEL`` 让这层日志能在
#: **不改变全局日志级别**（避免放量第三方噪声库）的前提下落盘。
#: 未设置 ⇒ 回落到 ``LOG_LEVEL``；都没设置 ⇒ INFO（即"看得见"）。
TRACE_LEVEL_ENV = "TRACE_LEVEL"

#: L2「慢调用清单」的阈值（毫秒）。仅当单次调用超过它才打 ``[SLOW]`` 一行。
SLOW_CALL_MS_ENV = "SLOW_CALL_MS"
DEFAULT_SLOW_CALL_MS = 500.0

#: L2 慢调用日志的采样率（0~1）。默认 1.0（不丢慢调用）。
#: 注意：它**只影响 L2 日志行**，不影响 L1 直方图（L1 永远全量，否则统计失真）。
SLOW_CALL_SAMPLE_ENV = "SLOW_CALL_SAMPLE"

#: L1 直方图的名字（``func_seconds{func=...}``）。登记在 ``metrics.METRIC_CATALOG``。
FUNC_SECONDS_METRIC = "func_seconds"

#: L2 日志行的固定标签，便于 ``grep '\[SLOW\]'`` 一次性捞出所有慢调用。
SLOW_LOG_TAG = "[SLOW]"


def _env_float(name: str, default: float) -> float:
    """读浮点环境变量；未设置/非法 ⇒ 默认值（**绝不因配置错误让业务失败**）。"""
    raw = os.environ.get(name, "")
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).strip())
    except ValueError:
        return default
    if value != value:  # NaN
        return default
    return value


def trace_level() -> int:
    """L3 级别：``TRACE_LEVEL`` 优先，否则回落 ``LOG_LEVEL``，都没有则 ``INFO``。

    **每次调用时求值**（不缓存）——这样测试可以改环境变量后立即生效，
    运维也可以在重启前用环境变量调整，无需改代码。
    """
    raw = os.environ.get(TRACE_LEVEL_ENV, "")
    if raw is not None and str(raw).strip() != "":
        return _resolve_level(str(raw).strip(), logging.INFO)
    return _resolve_level(os.environ.get("LOG_LEVEL", ""), logging.INFO)


def _resolve_level(text: str, default: int) -> int:
    """把 ``"20"`` / ``"DEBUG"`` 这类写法解析成 logging 级别常量。"""
    if not text:
        return default
    if text.isdigit():
        return int(text)
    return getattr(logging, text.upper(), default)


def slow_call_ms() -> float:
    """L2 阈值（毫秒）；``<=0`` 表示关闭慢调用日志。"""
    return _env_float(SLOW_CALL_MS_ENV, DEFAULT_SLOW_CALL_MS)


def log_trace(logger: Any, msg: str, *args: Any) -> None:
    """按 :func:`trace_level` 打一条 L3 日志（调用方无需自己判断级别）。"""
    log = logger if isinstance(logger, logging.Logger) else (
        logging.getLogger(logger) if isinstance(logger, str) else _LOGGER
    )
    level = trace_level()
    if log.isEnabledFor(level):
        log.log(level, msg, *args)

#: 键名命中即打码（大小写不敏感）。
#: 刻意**只匹配真凭据**：``api_key``/``token``/``secret``/``password``/``authorization``…
#: 保留 ``max_tokens`` 这种「名字里带 token 但不是密钥」的配置值可见，否则日志失去可读性。
SENSITIVE_KEY_PATTERN = (
    "api_key", "apikey", "authorization", "bearer", "cookie", "credential",
    "passwd", "password", "pwd", "private_key", "secret", "access_token",
    "refresh_token", "auth_token", "session_key", "_key", "key_",
)

#: 摘要截断阈值
MAX_STR_LEN = 200
MAX_ITEMS = 8
MAX_DEPTH = 3


# --------------------------------------------------------------------------
# 1. request 上下文（contextvars）
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RequestContext:
    """一次请求（追问/入库/任意任务）的上下文，随 contextvars 传播到协程与线程。"""

    request_id: str = ""
    user_id: str = ""
    session_id: str = ""
    role_id: str = ""
    path: str = ""
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id or None,
            "session_id": self.session_id or None,
            "role_id": self.role_id or None,
            "path": self.path or None,
        }

    def elapsed_ms(self) -> float:
        return (time.time() - self.started_at) * 1000.0


_EMPTY_CONTEXT = RequestContext()
_CONTEXT: contextvars.ContextVar[RequestContext] = contextvars.ContextVar(
    "legal_rag_request_context", default=_EMPTY_CONTEXT
)

# 独立暴露一个 request_id 变量：即使只 bind_request("id") 也能被日志过滤器读到
REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "legal_rag_request_id", default=""
)


def new_request_id(prefix: str = "req") -> str:
    """生成可读的 request_id：``req-20260915T073012-ab12cd34``。"""
    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def bind_request(
    request_id: str | None = None,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    role_id: str | None = None,
    path: str | None = None,
    context: RequestContext | None = None,
) -> contextvars.Token:
    """把 request_id 绑定到当前上下文，返回 token（可交给 :func:`reset_request`）。

    用法::

        token = bind_request()                      # 自动生成 id
        token = bind_request(req_id, user_id="u1")  # 复用上游 id
        ...
        reset_request(token)

    返回的 token 是 contextvars.Token（同时能拿到 id：``token.var.get()``）。
    """
    if context is not None:
        ctx = context
    else:
        current = current_request()
        ctx = replace(
            current,
            request_id=request_id or current.request_id or new_request_id(),
            user_id=user_id if user_id is not None else current.user_id,
            session_id=session_id if session_id is not None else current.session_id,
            role_id=role_id if role_id is not None else current.role_id,
            path=path if path is not None else current.path,
            started_at=time.time(),
        )
    return _CONTEXT.set(ctx)


def reset_request(token: contextvars.Token) -> None:
    """还原到 bind_request 之前的上下文（务必放在 finally 里）。

    同时把 :data:`REQUEST_ID` 还原成「还原后上下文里的 id」——即未绑定时为空串，
    否则 :func:`current_request_id` 会一直返回上一个请求的 id。
    """
    try:
        _CONTEXT.reset(token)
    except (ValueError, LookupError):  # token 已被消费/跨上下文
        _CONTEXT.set(_EMPTY_CONTEXT)
    REQUEST_ID.set(_CONTEXT.get().request_id)


def current_request() -> RequestContext:
    """当前请求上下文；未绑定时返回空的 :class:`RequestContext`。"""
    ctx = _CONTEXT.get()
    if not ctx.request_id:
        raw = REQUEST_ID.get()
        if raw:
            return RequestContext(request_id=raw)
    return ctx


def current_request_id() -> str | None:
    """当前 request_id；未绑定时返回 None（日志过滤器会显示 ``-``）。"""
    rid = _CONTEXT.get().request_id or REQUEST_ID.get()
    return rid or None


@contextmanager
def request_context(
    request_id: str | None = None,
    **kwargs: Any,
) -> Iterator[RequestContext]:
    """with 形式绑定请求上下文，退出时自动还原。"""
    token = bind_request(request_id, **kwargs)
    try:
        yield current_request()
    finally:
        reset_request(token)


def attach_logging_context(level: int | None = None) -> None:
    """把 request_id 注入根日志的过滤器（幂等），供未走 logging_setup 的场景调用。"""
    try:
        from .logging_setup import install_context_filter, setup_logging
        if level is not None:
            setup_logging(level)
        install_context_filter(logging.getLogger())
    except Exception:  # pragma: no cover - 日志配置失败不影响业务
        _LOGGER.debug("attach_logging_context 失败", exc_info=True)


# --------------------------------------------------------------------------
# 2. 摘要与打码
# --------------------------------------------------------------------------

def _is_sensitive_key(key: Any) -> bool:
    text = str(key).strip().lower()
    if not text:
        return False
    return any(pattern in text for pattern in SENSITIVE_KEY_PATTERN)


def mask_secrets(value: Any, mask: str = "***") -> Any:
    """递归把敏感键的值替换成 ``***``（字典/列表/嵌套结构）。"""
    if isinstance(value, dict):
        return {
            key: (mask if _is_sensitive_key(key) else mask_secrets(val, mask))
            for key, val in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        masked = [mask_secrets(item, mask) for item in value]
        return type(value)(masked) if isinstance(value, (list, tuple)) else masked
    return value


def summarize(value: Any, max_len: int = MAX_STR_LEN, max_items: int = MAX_ITEMS,
              depth: int = 0) -> Any:
    """把任意对象压成可安全写进日志的摘要。

    规则：``str`` 截断；容器只取前 N 个元素；``bytes`` 只报长度；
    对象优先用其 ``summary()``，否则用 ``ClassName(截断后的 repr)``；
    敏感键的值打码。
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        flat = value.replace("\n", "\\n")
        return flat if len(flat) <= max_len else f"{flat[:max_len]}...(+{len(flat) - max_len})"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{type(value).__name__} {len(bytes(value))} bytes>"
    if depth >= MAX_DEPTH:
        return f"<{type(value).__name__} depth-limit>"
    if isinstance(value, dict):
        safe = mask_secrets(value)
        items = list(safe.items())[:max_items]
        rendered = {
            str(key): summarize(val, max_len, max_items, depth + 1) for key, val in items
        }
        if len(safe) > max_items:
            rendered[f"...(+{len(safe) - max_items})"] = None
        return rendered
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = list(value)
        rendered = [summarize(item, max_len, max_items, depth + 1) for item in seq[:max_items]]
        if len(seq) > max_items:
            rendered.append(f"...(+{len(seq) - max_items})")
        return rendered
    summary = getattr(value, "summary", None)
    if callable(summary):
        try:
            return summarize(summary(), max_len, max_items, depth + 1)
        except Exception:  # pragma: no cover - 第三方对象 summary 抛异常也要能记日志
            pass
    text = repr(value).replace("\n", "\\n")
    if len(text) > max_len:
        text = f"{text[:max_len]}...(+{len(text) - max_len})"
    return f"{type(value).__name__}({text})"


def _summarize_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: summarize(val) for key, val in mapping.items()}


def _format_args(bound: dict[str, Any], max_len: int = 160) -> str:
    """把签名绑定结果渲染成 ``a=1, b=[2, 3]``；敏感键打码，过长整体截断。"""
    safe = {
        key: ("***" if _is_sensitive_key(key) else summarize(val))
        for key, val in bound.items()
    }
    text = ", ".join(f"{key}={value!r}" for key, value in safe.items())
    return text if len(text) <= max_len else f"{text[:max_len]}...(truncated)"


def _bind_arguments(func: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    try:
        signature = inspect.signature(func)
        bound = signature.bind_partial(*args, **kwargs)
        result = dict(bound.arguments)
    except (TypeError, ValueError):
        result = dict(kwargs)
        for index, value in enumerate(args):
            result[f"arg{index}"] = value
    # self/cls 只留类名，避免整对象摘要噪声
    for skip in ("self", "cls"):
        if skip in result:
            result[skip] = f"<{type(result[skip]).__name__}>"
    return result


def _format_return(value: Any, max_len: int = 160) -> str:
    if isinstance(value, (list, tuple, set, frozenset)):
        text = f"{type(value).__name__}(len={len(value)})"
    elif isinstance(value, dict):
        text = f"dict(keys={list(value)[:MAX_ITEMS]})"
    elif isinstance(value, str):
        text = repr(summarize(value))
    else:
        text = repr(summarize(value))
    return text if len(text) <= max_len else f"{text[:max_len]}...(truncated)"


def _logger_for(target: Any, logger: Any) -> logging.Logger:
    if isinstance(logger, logging.Logger):
        return logger
    if isinstance(logger, str):
        return logging.getLogger(logger)
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", "function")
    module = getattr(target, "__module__", "") or ""
    return logging.getLogger(f"{module}.{name}" if module else name)


# --------------------------------------------------------------------------
# 3. 异常包装
# --------------------------------------------------------------------------

class TracedError(Exception):
    """带业务上下文的异常包装。

    保留原始异常与栈：``__cause__`` 指向原始异常（``raise ... from exc``），
    ``context`` 携带业务字段（文件、阶段、条数……），``traceback`` 保留原始栈文本，
    便于「按日志快速定位问题」。
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
        elapsed_ms: float | None = None,
        traceback_text: str | None = None,
        function: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.context: dict[str, Any] = dict(context or {})
        self.cause = cause
        self.elapsed_ms = elapsed_ms
        self.traceback_text = traceback_text or (
            "".join(traceback.format_exception(type(cause), cause, cause.__traceback__))
            if cause is not None else ""
        )
        self.function = function
        self.original_exception = cause

    @property
    def cause_type(self) -> str:
        return type(self.cause).__name__ if self.cause is not None else ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "function": self.function,
            "message": self.message,
            "cause": self.cause_type,
            "cause_message": str(self.cause) if self.cause is not None else None,
            "elapsed_ms": round(self.elapsed_ms, 3) if self.elapsed_ms is not None else None,
            "context": summarize(self.context),
        }

    def __str__(self) -> str:
        detail = self.operation or self.function or ""
        cause = f" <- {self.cause_type}: {self.cause}" if self.cause is not None else ""
        suffix = f" [{detail}]" if detail else ""
        return f"{self.message}{suffix}{cause}"


# --------------------------------------------------------------------------
# 4. @traced：函数入口/出口日志
# --------------------------------------------------------------------------

def sample_rate_for(name: str, default: float = 1.0) -> float:
    """按函数名查采样率：环境变量 ``TRACE_SAMPLE_<函数名大写>`` 或 ``TRACE_SAMPLE``。"""
    key = "TRACE_SAMPLE_" + str(name).replace(".", "_").replace("-", "_").upper()
    raw = os.environ.get(key) or os.environ.get("TRACE_SAMPLE")
    if not raw:
        return default
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return default


def sampled(probability: float = 1.0, rng: random.Random | None = None) -> bool:
    """高频路径采样：返回 True 时调用方才打日志。``probability>=1`` 恒为 True。"""
    if probability >= 1.0:
        return True
    if probability <= 0.0:
        return False
    return (rng or random).random() < probability


def traced(
    _func: Callable | None = None,
    *,
    logger: Any = None,
    level: int | None = None,
    operation: str | None = None,
    sample: float | None = None,
    max_len: int = MAX_STR_LEN,
    log_result: bool = True,
    sensitive: bool = True,
):
    """函数调用前后打日志（含异常完整栈）——**L3：全量前后**。

    每条日志都带当前 request_id；入参/返回摘要自动截断，命中敏感键自动打码。

    :param logger: ``logging.Logger``、模块名字符串；缺省用 ``<模块>.<函数名>``；
    :param level: 入口/出口日志级别；**缺省 ``None`` = 走 :func:`trace_level`**
        （即 ``TRACE_LEVEL`` → ``LOG_LEVEL`` → ``INFO``）。
        传显式整数则完全按传入值（保持向后兼容；旧行为等价于传 ``logging.DEBUG``）。
    :param operation: 覆盖日志里的操作名；
    :param sample: 采样率 ``0~1``；缺省读环境变量 ``TRACE_SAMPLE``（默认全采）；
    :param log_result: 是否记录返回摘要（含敏感信息的大对象可关掉）；
    :param sensitive: 是否对入参/返回做敏感打码。

    **P0.5 变更说明**：``level`` 的默认值由 ``logging.DEBUG`` 改为 ``None``（=按
    :func:`trace_level` 求值）。这样"想看函数前后日志"只需设 ``TRACE_LEVEL=DEBUG``
    或保持默认 INFO，而**不必把全局 ``LOG_LEVEL`` 调到 DEBUG**（后者会放量 httpx /
    pymilvus / grpc 等第三方噪声库）。需要旧行为的调用方显式传 ``level=logging.DEBUG``。
    """
    def decorator(func: Callable) -> Callable:
        log = _logger_for(func, logger)
        op = operation or getattr(func, "__qualname__", getattr(func, "__name__", "call"))
        rate = sample_rate_for(op) if sample is None else sample
        is_async = inspect.iscoroutinefunction(func)

        def _emit(msg: str, *args: Any) -> None:
            """入口/出口日志：显式 level 优先，否则按 trace_level 动态求值。"""
            if level is None:
                log_trace(log, msg, *args)
            else:
                log.log(level, msg, *args)

        def _prepare(args: tuple, kwargs: dict) -> dict[str, Any]:
            bound = _bind_arguments(func, args, kwargs)
            return bound if not sensitive else bound  # summarize() 内部已打码

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not sampled(rate):
                    return await func(*args, **kwargs)
                bound = _prepare(args, kwargs)
                _emit("→ %s 入参(%s)", op, _format_args(bound))
                started = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    elapsed = (time.perf_counter() - started) * 1000.0
                    log.error("[FAIL] %s 异常，耗时 %.1fms：%s: %s\n%s", op, elapsed,
                              type(exc).__name__, exc, traceback.format_exc())
                    raise
                elapsed = (time.perf_counter() - started) * 1000.0
                note = f" 返回 {_format_return(result)}" if log_result else ""
                _emit("← %s 完成，耗时 %.1fms%s", op, elapsed, note)
                return result

            wrapper = async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not sampled(rate):
                    return func(*args, **kwargs)
                bound = _prepare(args, kwargs)
                _emit("→ %s 入参(%s)", op, _format_args(bound))
                started = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    elapsed = (time.perf_counter() - started) * 1000.0
                    log.error("[FAIL] %s 异常，耗时 %.1fms：%s: %s\n%s", op, elapsed,
                              type(exc).__name__, exc, traceback.format_exc())
                    raise
                elapsed = (time.perf_counter() - started) * 1000.0
                note = f" 返回 {_format_return(result)}" if log_result else ""
                _emit("← %s 完成，耗时 %.1fms%s", op, elapsed, note)
                return result

            wrapper = sync_wrapper

        wrapper.__traced__ = True  # type: ignore[attr-defined]
        return wrapper

    if _func is not None and callable(_func):
        return decorator(_func)
    return decorator


# --------------------------------------------------------------------------
# 4b. timed：耗时评估（P0.5 的 L1 + L2）
# --------------------------------------------------------------------------

def _record_func_seconds(op: str, elapsed_ms: float, logger: Any) -> None:
    """L1：把一次调用的耗时写进 ``func_seconds`` 直方图。

    **指标绝不影响业务**：任何失败只记一条 debug（与 ``metrics.py`` 的
    「指标自身出错绝不外溢」原则一致）。
    """
    try:
        from . import metrics as M  # 延迟导入：避免观测层与指标层循环依赖

        M.observe(FUNC_SECONDS_METRIC, elapsed_ms / 1000.0, func=op)
    except Exception:  # noqa: BLE001 - 指标失败不能影响业务
        _LOGGER.debug("func_seconds 指标写入失败（已忽略）：op=%s", op, exc_info=True)


def _log_slow_call(op: str, elapsed_ms: float, threshold: float, logger: Any,
                   extra: dict[str, Any] | None = None) -> None:
    """L2：超过阈值才打一行 ``[SLOW]``，便于 ``grep`` 一次性捞出慢调用清单。"""
    if threshold <= 0 or elapsed_ms < threshold:
        return
    if not sampled(_env_float(SLOW_CALL_SAMPLE_ENV, 1.0)):
        return
    log = logger if isinstance(logger, logging.Logger) else (
        logging.getLogger(logger) if isinstance(logger, str) else _LOGGER
    )
    tail = ""
    if extra:
        safe = _summarize_mapping(mask_secrets(extra))
        tail = " " + " ".join(f"{k}={v!r}" for k, v in safe.items())
    log.warning("%s func=%s elapsed=%.1fms threshold=%.0fms%s",
                SLOW_LOG_TAG, op, elapsed_ms, threshold, tail)


def timed(
    _func: Callable | None = None,
    *,
    logger: Any = None,
    operation: str | None = None,
    slow_ms: float | None = None,
    scope: dict[str, Any] | None = None,
    sample: float | None = None,
):
    """给函数加**耗时评估**：L1 汇总直方图（常开）+ L2 慢调用清单（常开，仅打慢的）。

    与 :func:`traced` 的分工（详见 ``docs/REFACTOR-PLAN.md`` §3.7）：

    * ``@traced`` —— **L3 全量前后**：每次调用都打「入参 / 完成+耗时」两行，用于追调用链。
      日志量大，按级别与采样按需开启。
    * ``@timed``  —— **L1+L2 速度评估**：每次调用都把耗时累加进 ``func_seconds`` 直方图
      （**永不被日志级别过滤**，因此统计不失真），只有**超过阈值**才额外打一行 ``[SLOW]``。
      默认常开而几乎不增加日志量。

    这样"想看大量详细日志"与"想做速度评估"两个目的被拆开，互不牺牲。

    :param logger: 记 ``[SLOW]`` 用的 logger；缺省用 ``<模块>.<函数名>``；
    :param operation: 覆盖操作名（默认 ``qualname``）；它就是直方图的 ``func`` 标签值，
        **不要**传入高基数内容（如 session_id），否则 Prometheus 序列会爆炸；
    :param slow_ms: 慢调用阈值（毫秒）；缺省读 ``SLOW_CALL_MS``（默认 500）；``<=0`` 关闭 L2；
    :param scope: 额外写进 ``[SLOW]`` 行的上下文字段（自动打码 + 截断）；
    :param sample: L1 采样率；缺省 1.0（**全量**）。仅在极端高 QPS 下才需要调低。

    用法::

        @timed
        def retrieve(self, query, top_k=None): ...

        @timed(operation="milvus.search", slow_ms=200.0, scope={"collection": "kb"})
        def search(self, vector, top_k): ...
    """
    def decorator(func: Callable) -> Callable:
        log = _logger_for(func, logger)
        op = operation or getattr(func, "__qualname__", getattr(func, "__name__", "call"))
        rate = 1.0 if sample is None else sample
        is_async = inspect.iscoroutinefunction(func)

        def _finish(elapsed_ms: float) -> None:
            _record_func_seconds(op, elapsed_ms, log)
            threshold = slow_call_ms() if slow_ms is None else float(slow_ms)
            extra = scope() if callable(scope) else scope
            _log_slow_call(op, elapsed_ms, threshold, log, extra)

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not sampled(rate):
                    return await func(*args, **kwargs)
                started = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    _finish((time.perf_counter() - started) * 1000.0)

            wrapper = async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not sampled(rate):
                    return func(*args, **kwargs)
                started = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    _finish((time.perf_counter() - started) * 1000.0)

            wrapper = sync_wrapper

        wrapper.__timed__ = True  # type: ignore[attr-defined]
        wrapper.__timed_operation__ = op  # type: ignore[attr-defined]
        return wrapper

    if _func is not None and callable(_func):
        return decorator(_func)
    return decorator


@contextmanager
def timed_scope(operation: str, *, logger: Any = None, slow_ms: float | None = None,
                scope: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """``@timed`` 的**代码块**版本：给"不能加装饰器"的位置（如路由处理器内部、
    循环体、条件分支）补同一套 L1+L2 观测。

    用法::

        with timed_scope("chat.retrieve") as t:
            hits = self.retriever.retrieve(q, where=scope)
            t["hits"] = len(hits)      # 写回慢调用行

    退出时（含异常路径）记录 ``func_seconds``；超阈值时打 ``[SLOW]``。
    """
    log = logger if isinstance(logger, logging.Logger) else (
        logging.getLogger(logger) if isinstance(logger, str) else _LOGGER
    )
    extra: dict[str, Any] = dict(scope or {})
    started = time.perf_counter()
    try:
        yield extra
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _record_func_seconds(operation, elapsed_ms, log)
        threshold = slow_call_ms() if slow_ms is None else float(slow_ms)
        _log_slow_call(operation, elapsed_ms, threshold, log, extra or None)


# --------------------------------------------------------------------------
# 5. log_io：数据出入口
# --------------------------------------------------------------------------

@contextmanager
def log_io(
    source: str,
    to: str,
    *,
    operation: str = "io",
    count: int | None = None,
    size: int | None = None,
    level: int = logging.INFO,
    logger: Any = None,
    meta: dict[str, Any] | None = None,
    raise_instead_of_log: bool = True,
) -> Iterator[dict[str, Any]]:
    """记录一次数据出入口：来源 → 去向、条数、字节数、耗时、失败原因。

    用法::

        with log_io("uploads/a.pdf", "milvus", operation="upsert") as io:
            io["count"] = len(chunks)
            io["size"] = total_bytes
            store.upsert(chunks)

    进入时打一条 INFO（起点），退出时打一条 INFO（终点 + 耗时 + 条数 + 字节），
    异常时打一条 ERROR（失败原因 + 完整栈）并把异常原样抛出（不吞）。
    在 ``finally`` 里总能拿到 ``io`` 字典，可写回汇总字段。
    """
    log = logger if isinstance(logger, logging.Logger) else (
        logging.getLogger(logger) if isinstance(logger, str) else _LOGGER
    )
    io: dict[str, Any] = {
        "operation": operation,
        "source": source,
        "to": to,
        "count": count,
        "size": size,
        "meta": dict(meta or {}),
        "ok": None,
        "elapsed_ms": None,
        "error": None,
    }
    started = time.perf_counter()
    log.log(level, "[IO] → %s: %s -> %s 开始（count=%s size=%s）",
            operation, source, to, io["count"], _fmt_size(io["size"]))
    try:
        yield io
    except Exception as exc:
        io["ok"] = False
        io["error"] = f"{type(exc).__name__}: {exc}"
        io["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        log.error("[IO] [FAIL] %s 失败: %s -> %s，耗时 %.1fms，失败原因=%s\n%s",
                  operation, source, to, io["elapsed_ms"], io["error"],
                  traceback.format_exc())
        if raise_instead_of_log:
            raise
    else:
        io["ok"] = True
        io["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        log.log(level, "[IO] [OK] %s 完成: %s -> %s，条数=%s，字节=%s，耗时 %.1fms",
                operation, source, to, io["count"], _fmt_size(io["size"]),
                io["elapsed_ms"])


def _fmt_size(size: Any) -> str:
    try:
        number = int(size)
    except (TypeError, ValueError):
        return "n/a"
    if number < 0:
        return "n/a"
    for unit, scale in (("B", 1), ("KB", 1024), ("MB", 1024 ** 2), ("GB", 1024 ** 3)):
        if number < scale * 1024 or unit == "GB":
            return f"{number}{unit}" if scale == 1 else f"{number / scale:.1f}{unit}({number}B)"
    return f"{number}B"


# --------------------------------------------------------------------------
# 6. wrap_errors：统一异常包装
# --------------------------------------------------------------------------

@contextmanager
def wrap_errors(
    operation: str,
    **context: Any,
) -> Iterator[dict[str, Any]]:
    """把块内异常包成 :class:`TracedError`（保留 ``__cause__`` 原始栈 + 业务上下文）。

    用法::

        with wrap_errors("milvus.upsert", collection=name, count=len(rows)):
            store.upsert(rows)

    异常一定被记录（ERROR + 完整栈），一定被重新抛出；已经是 TracedError 时
    只追加一层上下文，不重复包装。
    """
    log = _LOGGER
    started = time.perf_counter()
    try:
        yield context
    except TracedError as exc:
        if context:
            exc.context.update({k: v for k, v in context.items() if k not in exc.context})
        log.error("[ERR] %s 失败：%s\n%s", operation, exc, exc.traceback_text or traceback.format_exc())
        raise
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        wrapped = TracedError(
            f"{operation} 失败",
            operation=operation,
            context=dict(context),
            cause=exc,
            elapsed_ms=elapsed,
        )
        log.error("[ERR] %s 失败，耗时 %.1fms：%s: %s\n%s",
                  operation, elapsed, type(exc).__name__, exc, traceback.format_exc())
        raise wrapped from exc


def wrap_errors_decorator(operation: str | None = None, **context: Any):
    """装饰器形式：把函数内异常统一包成 TracedError（保留原始栈）。"""
    def decorator(func: Callable) -> Callable:
        op = operation or getattr(func, "__qualname__", getattr(func, "__name__", "call"))
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any):
                with wrap_errors(op, **context):
                    return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            with wrap_errors(op, **context):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# --------------------------------------------------------------------------
# 7. 轻量日志助手（单点日志，不用装饰器时用这些）
# --------------------------------------------------------------------------

def log_event(event: str, /, level: int = logging.INFO, logger: Any = None,
              **fields: Any) -> str:
    """打一条结构化单点日志：``[事件] key=value ...``（敏感字段打码、值截断）。"""
    log = logger if isinstance(logger, logging.Logger) else (
        logging.getLogger(logger) if isinstance(logger, str) else _LOGGER
    )
    safe = _summarize_mapping(mask_secrets(fields)) if fields else {}
    tail = " ".join(f"{key}={value!r}" for key, value in safe.items())
    log.log(level, "[%s] %s", event, tail)
    return tail


def log_before(operation: str, logger: Any = None, level: int = logging.DEBUG,
               **fields: Any) -> None:
    """外部服务调用前：记「要调什么、带什么」。"""
    log_event(f"BEFORE {operation}", level=level, logger=logger, **fields)


def log_after(operation: str, elapsed_ms: float, logger: Any = None,
              level: int = logging.DEBUG, **fields: Any) -> None:
    """外部服务调用后：记「耗时多少、返回什么」。"""
    log_event(f"AFTER {operation} 耗时={elapsed_ms:.1f}ms", level=level, logger=logger, **fields)


def log_exception(operation: str, exc: BaseException | None = None,
                  logger: Any = None, **context: Any) -> TracedError:
    """记录异常（完整栈）并返回对应的 :class:`TracedError`，供调用方 raise。"""
    log = logger if isinstance(logger, logging.Logger) else _LOGGER
    message = f"{operation} 失败"
    wrapped = TracedError(message, operation=operation, context=dict(context), cause=exc)
    log.error("[ERR] %s：%s\n%s", operation, wrapped, traceback.format_exc())
    return wrapped


def safe_call(func: Callable, *args: Any, default: Any = None, operation: str | None = None,
              logger: Any = None, **kwargs: Any) -> Any:
    """**明确需要**「失败也继续」时使用：记录异常后返回 ``default``。

    默认业务代码请优先用 :func:`wrap_errors` / :func:`traced`——本项目原则是
    「异常一定被记录、绝不静默吞掉」，本函数仅用于降级链这类可容忍失败的旁路。
    """
    op = operation or getattr(func, "__name__", "call")
    started = time.perf_counter()
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        log = logger if isinstance(logger, logging.Logger) else _LOGGER
        log.warning("[DEGRADED] %s 失败（已降级，耗时 %.1fms）：%s: %s\n%s",
                    op, elapsed, type(exc).__name__, exc, traceback.format_exc())
        return default
