"""日志配置：loguru 控制台 + 三个文件（app.log / error.log / rag_trace.jsonl）。

工单要求：
- 使用 loguru；
- 每个函数入口/出口记录输入输出（通过 ``@trace`` 装饰器实现）；
- 异常必须记录堆栈，禁止静默失败；
- 日志格式为 JSON，包含时间、模块、函数、输入、输出、耗时、异常信息。

实现说明：
- ``logger`` 是配置好的 loguru logger（仅配置一次）。
- ``@trace`` 装饰器负责记录函数级输入/输出/耗时，并写入 rag_trace.jsonl。
- 若环境未安装 loguru，自动降级到标准库 logging，**接口保持完全一致**，
  保证“禁止静默失败”这一硬性要求在缺依赖时依然成立。
"""

from __future__ import annotations

import functools
import json
import sys
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from app.core.config import get_settings

F = TypeVar("F", bound=Callable[..., Any])

_configured = False
_TRACE_HANDLE = None

try:  # pragma: no cover - 依赖可用性分支
    from loguru import logger as _loguru_logger

    HAS_LOGURU = True
except Exception:  # pragma: no cover
    _loguru_logger = None
    HAS_LOGURU = False


# --------------------------------------------------------------------------
# 值截断：日志里不能塞入整篇 PDF 文本
# --------------------------------------------------------------------------
MAX_LOG_VALUE_CHARS = 800
# 追踪日志单文件上限与保留份数。
# 每次函数调用都会写一行 rag_trace.jsonl，长时间运行会累积到数十 MB，
# 因此在进程启动时做一次大小检查并轮转，避免把磁盘写满。
TRACE_MAX_BYTES = 32 * 1024 * 1024
TRACE_KEEP_FILES = 3


def _shrink(value: Any, depth: int = 0) -> Any:
    """把任意对象压缩成可 JSON 序列化的短表示。"""
    if depth > 3:
        return "..."
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= MAX_LOG_VALUE_CHARS else value[:MAX_LOG_VALUE_CHARS] + f"...(+{len(value) - MAX_LOG_VALUE_CHARS})"
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        head = [_shrink(item, depth + 1) for item in items[:5]]
        if len(items) > 5:
            head.append(f"...(+{len(items) - 5})")
        return head
    if isinstance(value, dict):
        return {str(key): _shrink(val, depth + 1) for key, val in list(value.items())[:20]}
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):  # pydantic 模型
        try:
            return _shrink(value.model_dump(), depth + 1)
        except Exception:
            return repr(value)[:MAX_LOG_VALUE_CHARS]
    return repr(value)[:MAX_LOG_VALUE_CHARS]


def _rotate_trace_file(target: Path) -> None:
    """启动时按大小轮转 rag_trace.jsonl。

    loguru 的 ``rotation`` 只作用于它自己的 sink；rag_trace.jsonl 由本模块直接
    持有文件句柄写入，因此需要显式轮转。轮转失败不能影响日志系统启动。
    """
    try:
        if not target.exists() or target.stat().st_size < TRACE_MAX_BYTES:
            return
        for index in range(TRACE_KEEP_FILES - 1, 0, -1):
            older = target.with_name(f"{target.name}.{index}")
            newer = target.with_name(f"{target.name}.{index + 1}")
            if older.exists():
                if index + 1 >= TRACE_KEEP_FILES:
                    older.unlink(missing_ok=True)
                else:
                    older.replace(newer)
        target.replace(target.with_name(f"{target.name}.1"))
    except Exception:  # pragma: no cover - 轮转失败不应影响主流程
        pass


def _configure() -> None:
    """初始化日志系统（幂等）。"""
    global _configured, _TRACE_HANDLE
    if _configured:
        return

    settings = get_settings()
    log_dir = settings.paths.logs
    log_dir.mkdir(parents=True, exist_ok=True)
    trace_path = log_dir / "rag_trace.jsonl"
    _rotate_trace_file(trace_path)

    if HAS_LOGURU:
        _loguru_logger.remove()
        _loguru_logger.add(
            sys.stderr,
            level=settings.app.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            colorize=True,
            enqueue=False,
        )
        # app.log：全量 JSON 行日志
        _loguru_logger.add(
            log_dir / "app.log",
            level="DEBUG",
            serialize=True,
            rotation="50 MB",
            retention=10,
            encoding="utf-8",
            enqueue=False,
        )
        # error.log：仅异常，带堆栈
        _loguru_logger.add(
            log_dir / "error.log",
            level="ERROR",
            serialize=True,
            backtrace=True,
            diagnose=False,
            rotation="20 MB",
            retention=10,
            encoding="utf-8",
            enqueue=False,
        )
        _TRACE_HANDLE = open(trace_path, "a", encoding="utf-8", newline="\n")
    else:  # pragma: no cover - 降级路径
        import logging

        root = logging.getLogger("rag")
        root.setLevel(settings.app.log_level)
        if not root.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"))
            root.addHandler(handler)
            file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            root.addHandler(file_handler)
        _TRACE_HANDLE = open(trace_path, "a", encoding="utf-8", newline="\n")

    _configured = True


# --------------------------------------------------------------------------
# 统一 logger 门面：无论底层是 loguru 还是 logging，调用方式一致
# --------------------------------------------------------------------------
class _Logger:
    """日志门面。"""

    def _emit(self, level: str, module: str, message: str, **extra: Any) -> None:
        _configure()
        payload = {"module": module, **{key: _shrink(val) for key, val in extra.items()}}
        if HAS_LOGURU:
            _loguru_logger.opt(depth=1).bind(**payload).log(level, message)
        else:  # pragma: no cover
            import logging

            logging.getLogger(f"rag.{module}").log(
                getattr(logging, level), "%s | %s", message, json.dumps(payload, ensure_ascii=False)
            )

    def debug(self, module: str, message: str, **extra: Any) -> None:
        self._emit("DEBUG", module, message, **extra)

    def info(self, module: str, message: str, **extra: Any) -> None:
        self._emit("INFO", module, message, **extra)

    def warning(self, module: str, message: str, **extra: Any) -> None:
        self._emit("WARNING", module, message, **extra)

    def error(self, module: str, message: str, **extra: Any) -> None:
        self._emit("ERROR", module, message, **extra)

    def exception(self, module: str, message: str, **extra: Any) -> None:
        """记录异常并附带完整堆栈。"""
        extra = {**extra, "traceback": traceback.format_exc()}
        self._emit("ERROR", module, message, **extra)


logger = _Logger()


# --------------------------------------------------------------------------
# 追踪日志：函数入口/出口/耗时 -> rag_trace.jsonl
# --------------------------------------------------------------------------
def _write_trace(record: dict[str, Any]) -> None:
    _configure()
    if _TRACE_HANDLE is None:  # pragma: no cover
        return
    try:
        _TRACE_HANDLE.write(json.dumps(record, ensure_ascii=False) + "\n")
        _TRACE_HANDLE.flush()
    except Exception:  # pragma: no cover - 日志本身失败不能影响主流程
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def trace(func: F) -> F:
    """装饰器：记录被装饰函数的输入、输出、耗时与异常。

    ``self``/``cls`` 参数在记录时会被剔除，避免把整个对象写进日志。
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        module = func.__module__
        qualname = func.__qualname__
        bound = list(args)
        if bound and hasattr(bound[0], "__dict__") and not isinstance(bound[0], (str, bytes, int, float)):
            bound = bound[1:]
        record: dict[str, Any] = {
            "ts": _now_iso(),
            "event": "enter",
            "module": module,
            "function": qualname,
            "args": _shrink(bound),
            "kwargs": _shrink(kwargs),
        }
        _write_trace(record)
        started = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            _write_trace(
                {
                    "ts": _now_iso(),
                    "event": "error",
                    "module": module,
                    "function": qualname,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            logger.exception(module, f"{qualname} 执行失败", function=qualname, elapsed_ms=round(elapsed_ms, 3))
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        _write_trace(
            {
                "ts": _now_iso(),
                "event": "exit",
                "module": module,
                "function": qualname,
                "elapsed_ms": round(elapsed_ms, 3),
                "result": _shrink(result),
            }
        )
        return result

    return wrapper  # type: ignore[return-value]


def log_stage(stage: str, message: str, **extra: Any) -> None:
    """记录阶段级进度（阶段 0~5 的里程碑）。"""
    logger.info("app.stage", f"[{stage}] {message}", stage=stage, **extra)


def setup_logging() -> None:
    """显式初始化日志（供 main.py / 测试 fixture 调用）。"""
    _configure()
    logger.info("app.core.logging_conf", "日志系统初始化完成", loguru=HAS_LOGURU)
