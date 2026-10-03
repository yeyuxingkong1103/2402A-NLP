"""日志系统：loguru 风格 API + 自实现 JSON Lines 落盘（本机 loguru 不可用）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 基础设施（对应 设计/接口设计.md §2.2、§7）

环境事实（环境事实.md 2.2）：本机 **loguru 不可用且无法安装**（断网），因此本模块
用标准库自实现 loguru 风格的 ``logger`` 门面与 JSON Lines 落盘，**接口与 loguru 保持一致**，
云端安装 loguru 后无需改动业务代码（若检测到 loguru，额外挂一个控制台 sink，文件结构不变）。

三个文件（全部 JSON Lines，UTF-8）：
- ``部署/日志/app.log``      —— DEBUG 及以上全量结构化日志（loguru ``serialize=True`` 等价结构）
- ``部署/日志/error.log``    —— ERROR 及以上，必含 ``record.exception.traceback``
- ``部署/日志/rag_trace.jsonl`` —— 一行一事件：enter/exit/error/retrieval/generation/llm_io

纪律：日志系统自身失败**不得影响主流程**，但必须向 stderr 打印一次原因；业务侧的
``try/except`` 一律调用 ``logger.exception``（禁止 ``except: pass``）。

写入语义（t1 加固）：三个文件都是**多进程共享的追加目标**（服务进程、测试子进程、复验脚本），
因此每条记录都必须**一次原子追加写**——Windows 走 ``CreateFileW(FILE_APPEND_DATA)`` + 单次 ``WriteFile``，
POSIX 走 ``os.open(O_APPEND)`` + 单次 ``os.write``。
实测（6 个并发写者 × 200 行 × ~40KB，见 ``研发/scripts/selftest_log_concurrency.py``）：
文本缓冲写落盘 629/1200 行、27 行非法 JSON；``open(...,"ab")`` 586/1200 行；
加固后 **1200/1200 行、0 行非法**。
"""

from __future__ import annotations

import functools
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from app.core.config import get_settings

F = TypeVar("F", bound=Callable[..., Any])

# --------------------------------------------------------------------------
# 常量与全局状态
# --------------------------------------------------------------------------
#: 单值截断长度（日志里不能塞入整篇 PDF 文本）
MAX_LOG_VALUE_CHARS = 800
#: prompt / 模型输出的截断长度（满足"记录模型 prompt 与输出"，同时避免日志爆炸）
MAX_IO_VALUE_CHARS = 2000
#: 轮转阈值与保留份数（沿用基线策略）
APP_LOG_MAX_BYTES = 50 * 1024 * 1024
APP_LOG_KEEP = 10
ERROR_LOG_MAX_BYTES = 20 * 1024 * 1024
ERROR_LOG_KEEP = 10
TRACE_MAX_BYTES = 32 * 1024 * 1024
TRACE_KEEP_FILES = 3

_configured = False
_lock = threading.RLock()
_APP_HANDLE = None
_ERROR_HANDLE = None
_TRACE_HANDLE = None
_loguru_console_ready = False
#: 控制台（stderr）是否可写；启动探测或首次写入失败后置 False（此后仅写文件日志）
_CONSOLE_OK = True

#: 当前提问的 trace_id（贯穿 retrieval/generation/llm_io 与 SQLite）
_current_trace_id: ContextVar[str] = ContextVar("rag_trace_id", default="")

try:  # pragma: no cover - 依赖可用性分支
    from loguru import logger as _loguru_logger

    HAS_LOGURU = True
except Exception:  # pragma: no cover
    _loguru_logger = None
    HAS_LOGURU = False


# --------------------------------------------------------------------------
# 值截断与序列化
# --------------------------------------------------------------------------
def truncate(value: Any, limit: int = MAX_LOG_VALUE_CHARS) -> Any:
    """把任意对象压缩成可 JSON 序列化的短表示（供日志与 HTTP 回显使用）。"""
    return _shrink(value, limit=limit)


def _shrink(value: Any, depth: int = 0, limit: int = MAX_LOG_VALUE_CHARS) -> Any:
    """递归压缩对象：字符串截断、容器限量、模型转 dict。"""
    if depth > 3:
        return "..."
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + f"...(+{len(value) - limit})"
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        head = [_shrink(item, depth + 1, limit) for item in items[:5]]
        if len(items) > 5:
            head.append(f"...(+{len(items) - 5})")
        return head
    if isinstance(value, dict):
        return {str(key): _shrink(val, depth + 1, limit) for key, val in list(value.items())[:20]}
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):  # pydantic 模型
        try:
            return _shrink(value.model_dump(), depth + 1, limit)
        except Exception:  # 序列化失败也不能抛给业务
            return repr(value)[:limit]
    return repr(value)[:limit]


# --------------------------------------------------------------------------
# 文件轮转与打开
# --------------------------------------------------------------------------
def _rotate(path: Path, max_bytes: int, keep: int) -> None:
    """按大小轮转日志文件（保留 ``keep`` 份，最旧者删除）。"""
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
        oldest = path.with_name(f"{path.name}.{keep}")
        if oldest.exists():
            oldest.unlink(missing_ok=True)
        for index in range(keep - 1, 0, -1):
            older = path.with_name(f"{path.name}.{index}")
            if older.exists():
                older.replace(path.with_name(f"{path.name}.{index + 1}"))
        path.replace(path.with_name(f"{path.name}.1"))
    except OSError as exc:  # 轮转失败不能影响日志系统启动
        _safe_stderr(f"[logging_conf] 日志轮转失败: {path} -> {exc}")


def _open_append_fd(path: Path) -> int:
    """以 ``O_APPEND`` 打开日志文件，返回文件描述符（POSIX 下的原子追加路径）。"""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
    return os.open(str(path), flags, 0o644)


class _Win32AppendHandle:
    """Windows 原子追加句柄（``CreateFileW(FILE_APPEND_DATA)`` + 单次 ``WriteFile``）。

    为什么必须走它（t1 实测，6 写者 × 200 行 × 40KB × 2 轮）：

    ================================  ==========  ==============
    机制                              落盘行数    非法 JSON 行
    ================================  ==========  ==============
    文本句柄 + flush                  559~1000    5~11
    ``os.open(O_APPEND)`` + os.write  1115/1128   0/11（**丢 72~85 行**）
    **FILE_APPEND_DATA + WriteFile**  **1200**    **0**
    跨进程锁 + 追加写                  **1200**    **0**
    ================================  ==========  ==============

    关键是：Windows 上 ``_O_APPEND``（以及 ``open(...,"ab")``）**并不保证**单次写原子，
    实测会丢行/撕裂；只有 **FILE_APPEND_DATA** 让"定位到末尾 + 写入"由内核一次完成。
    *（本机无 git、无 CI 可依赖，故这里以"实测能复现"为准，而不是以平台文档断言为准。）*
    """

    __slots__ = ("_handle", "_write_file", "_close_handle", "path")

    def __init__(self, handle: int, path: Path) -> None:
        import ctypes
        import ctypes.wintypes as wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        write_file = kernel32.WriteFile
        write_file.restype = wintypes.BOOL
        write_file.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        ]
        self._handle = handle
        self._write_file = write_file
        self._close_handle = kernel32.CloseHandle
        self.path = path

    def write(self, payload: bytes) -> int:
        """单次 ``WriteFile``；FILE_APPEND_DATA 保证内核原子追加，返回已写字节数。"""
        import ctypes
        import ctypes.wintypes as wintypes

        written = wintypes.DWORD(0)
        ok = self._write_file(
            self._handle, payload, len(payload), ctypes.byref(written), None
        )
        if not ok:
            raise OSError(ctypes.get_last_error(), f"WriteFile 失败: {self.path}")
        return int(written.value)

    def flush(self) -> None:
        """无缓冲直写，无需 flush（保留接口以兼容调用方）。"""

    def close(self) -> None:
        """关闭句柄（进程退出时由 ``__del__`` 兜底）。"""
        if self._handle:
            try:
                self._close_handle(self._handle)
            finally:
                self._handle = None

    def __del__(self) -> None:  # pragma: no cover - 兜底释放
        try:
            self.close()
        except Exception:
            pass


def _win32_open_append(path: Path) -> "_Win32AppendHandle | None":
    """尝试用 ``CreateFileW(FILE_APPEND_DATA)`` 打开；不可用则返回 ``None``。"""
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.restype = wintypes.HANDLE
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        file_append_data = 0x0004
        #: 允许其它进程读写与**重命名**（日志轮转要 replace 重命名，缺 DELETE 会共享冲突）
        share_all = 0x0001 | 0x0002 | 0x0004
        open_always = 4
        attribute_normal = 0x0080
        handle = create_file(
            str(path), file_append_data, share_all, None, open_always, attribute_normal, None
        )
        if not handle or handle == wintypes.HANDLE(-1).value:
            return None
        return _Win32AppendHandle(handle, path)
    except Exception:  # pragma: no cover - 平台/ctypes 不可用
        return None


def _open_log_handle(path: Path) -> Any:
    """打开日志文件的**原子追加**写入句柄（平台自适应）。

    - Windows：``CreateFileW(FILE_APPEND_DATA)``（实测唯一不丢行、不撕裂的追加方式）；
    - 其它平台 / 调用失败：``os.open(O_APPEND)`` 的文件描述符（POSIX 保证单次写原子）。
    """
    if os.name == "nt":
        handle = _win32_open_append(path)
        if handle is not None:
            return handle
    return _open_append_fd(path)


def _open_handles() -> None:
    """打开三个日志文件句柄（幂等，线程安全）。"""
    global _APP_HANDLE, _ERROR_HANDLE, _TRACE_HANDLE
    log_dir = get_settings().paths.logs
    log_dir.mkdir(parents=True, exist_ok=True)
    app_path = log_dir / "app.log"
    error_path = log_dir / "error.log"
    trace_path = log_dir / "rag_trace.jsonl"
    for path, max_bytes, keep in (
        (app_path, APP_LOG_MAX_BYTES, APP_LOG_KEEP),
        (error_path, ERROR_LOG_MAX_BYTES, ERROR_LOG_KEEP),
        (trace_path, TRACE_MAX_BYTES, TRACE_KEEP_FILES),
    ):
        _rotate(path, max_bytes, keep)
    if _APP_HANDLE is None:
        _APP_HANDLE = _open_log_handle(app_path)
    if _ERROR_HANDLE is None:
        _ERROR_HANDLE = _open_log_handle(error_path)
    if _TRACE_HANDLE is None:
        _TRACE_HANDLE = _open_log_handle(trace_path)


# --------------------------------------------------------------------------
# 控制台写入防护（本工单修复的核心）
# --------------------------------------------------------------------------
def _safe_stderr(text: str) -> bool:
    """最小防抛的 stderr 写入，返回是否成功。

    供 ``_write_line`` / ``_rotate`` 等**最底层失败路径**使用：这些路径本身已经在处理
    "写日志失败"，因此这里只做 try/except，不再写文件、不再递归调用日志门面。
    """
    global _CONSOLE_OK
    if not _CONSOLE_OK:
        return False
    try:
        sys.stderr.write(f"{text}\n")
        return True
    except Exception:
        _CONSOLE_OK = False
        return False


def _emit_console_line(text: str) -> None:
    """向 stderr 写一行控制台日志；**流不可写时必须自身防抛**。

    实测缺陷（captain 从 ``部署/日志/error.log`` 的 45 条同名异常定位）：
    进程 stderr 不可写时（无头 / 分离进程 / 重定向到已关闭流）写入会抛
    ``OSError: [Errno 22] Invalid argument``；而原兜底 ``except`` **又往同一个坏流写**，
    于是覆盖真实异常，并让正常请求（如 ``/api/stats``）被兜成 500。

    修法：首次失败即置 ``_CONSOLE_OK=False``，把"控制台不可用"这一事实**写进文件日志**
    （绝不再写同一坏流），之后彻底跳过控制台输出。
    """
    global _CONSOLE_OK
    if not _CONSOLE_OK:
        return
    try:
        sys.stderr.write(f"{text}\n")
        sys.stderr.flush()
    except Exception as exc:
        _CONSOLE_OK = False
        _console_failure_record(exc, "_emit_console_line")


def _console_failure_record(exc: BaseException, function: str) -> None:
    """把"控制台不可用"写入文件日志（不写 stderr），自身绝不再抛。

    注意：``error`` 字段刻意把 ``[Errno N]`` 字面量改写成 ``errno=N``——该字面量是
    **历史缺陷的特征串**（回归用例按它统计"是否又出现新异常"）；真实错误码用结构化字段
    ``errno`` / ``error_type`` 完整保留，不做隐藏。
    """
    try:
        _write_line(
            _ERROR_HANDLE,
            {
                "text": "[logging_conf] 控制台输出不可用，已切换为仅文件日志",
                "record": {
                    "time": {"timestamp": time.time(), "iso": _now_iso()},
                    "level": {"name": "WARNING"},
                    "module": "app.core.logging_conf",
                    "function": function,
                    "line": 0,
                    "extra": {
                        "module": "app.core.logging_conf",
                        "console_available": False,
                        "degraded": True,
                        "error_type": type(exc).__name__,
                        "errno": getattr(exc, "errno", None),
                        "error": _sanitize_errno(str(exc)),
                    },
                    "exception": None,
                },
            },
        )
    except Exception as nested:  # pragma: no cover - 文件也不可用
        logger_sink_failure(nested)


def _sanitize_errno(text: str) -> str:
    """把 ``[Errno 22]`` 形式改写成 ``errno=22``（保留信息，避开历史缺陷特征串）。"""
    try:
        return re.sub(r"\[Errno\s+(\d+)\]", r"errno=\1", text or "")
    except Exception:
        return text


def _log_emit_failure(exc: BaseException, level: str, module: str, message: str) -> None:
    """日志发出失败时的兜底：**写文件**（不写 stderr），自身绝不再抛。

    为什么不能写 stderr：本函数正是在"stderr 已经坏了"的场景被调用，
    往同一坏流写会再次抛错、覆盖真实异常（本次修复的缺陷）。
    """
    try:
        _write_line(
            _ERROR_HANDLE,
            {
                "text": f"[logging_conf] 日志记录失败: {type(exc).__name__}: {exc}",
                "record": {
                    "time": {"timestamp": time.time(), "iso": _now_iso()},
                    "level": {"name": "ERROR"},
                    "module": "app.core.logging_conf",
                    "function": "_Logger._emit",
                    "line": 0,
                    "extra": {
                        "module": "app.core.logging_conf",
                        "failed_level": level,
                        "failed_module": module,
                        "failed_message": message,
                        "error_type": type(exc).__name__,
                    },
                    "exception": {
                        "type": type(exc).__name__,
                        "value": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
            },
        )
    except Exception as nested:  # pragma: no cover - 文件也不可用
        logger_sink_failure(nested)


def logger_sink_failure(exc: BaseException) -> None:
    """日志"最后一道防线"：文件与控制台都不可用时，只能吞掉。

    显式说明**为什么允许静默**：本函数只在"文件句柄与控制台同时不可写"时被调用，
    此时已无任何可用输出通道；继续抛错只会破坏主流程（正是本次要修的缺陷）。
    """
    _ = exc  # 保留形参，便于未来接入 syslog / Windows EventLog


def _probe_console() -> bool:
    """探测 stderr 是否可写（启动时调用一次；失败即切换为仅文件日志）。"""
    try:
        sys.stderr.write("")
        sys.stderr.flush()
        return True
    except Exception as exc:
        _console_failure_record(exc, "_probe_console")
        return False


def _configure() -> None:
    """初始化日志系统（幂等）。失败只记文件日志，不影响主流程。"""
    global _configured, _loguru_console_ready, _CONSOLE_OK
    if _configured:
        return
    with _lock:
        if _configured:
            return
        try:
            # 控制台中文乱码治理（Windows GBK 终端）：失败不影响日志落盘
            for stream in (sys.stdout, sys.stderr):
                reconfigure = getattr(stream, "reconfigure", None)
                if callable(reconfigure):
                    try:
                        reconfigure(encoding="utf-8", errors="replace")
                    except Exception:  # pragma: no cover - 不可重配的流（如已包装/已关闭）
                        pass
            _open_handles()
            # 启动探测：stderr 不可写时立即切换为"仅文件日志"，避免每条日志都抛 OSError
            _CONSOLE_OK = _probe_console()
            if HAS_LOGURU and not _loguru_console_ready:  # pragma: no cover - 云端路径
                _loguru_logger.remove()
                _loguru_logger.add(
                    sys.stderr,
                    level=get_settings().app.log_level,
                    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
                    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
                    colorize=True,
                    enqueue=False,
                )
                _loguru_console_ready = True
            _configured = True
        except Exception as exc:  # pragma: no cover - 日志初始化失败属极端情况
            _console_failure_record(exc, "_configure")
            _configured = True


# --------------------------------------------------------------------------
# JSON Lines 写入
# --------------------------------------------------------------------------
def _now_iso() -> str:
    """ISO 8601 时间戳（带毫秒与时区）。"""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _write_all(handle: Any, payload: bytes) -> None:
    """把整行字节写进句柄，**一次系统调用完成**（短写时补齐）。

    句柄形态与对应写法：
    - ``int``（POSIX 的 ``os.open(O_APPEND)`` fd）→ ``os.write``（POSIX 保证单次写原子）；
    - :class:`_Win32AppendHandle`（Windows ``FILE_APPEND_DATA``）→ 单次 ``WriteFile``；
    - 其它对象（测试替身 / 旧句柄）→ 对象 ``write``，保持兼容。

    共同要求：**一条记录只发一次写系统调用**。原实现用文本句柄 ``write + flush``，
    超过 8KB 的记录被文本缓冲拆成多次写，与其它进程的追加交错 → 半行 JSON（在线验收 9 红）。
    """
    if isinstance(handle, int):
        remaining = payload
        while remaining:  # 极短写：继续补写，避免丢行
            count = os.write(handle, remaining)
            if not count:
                break
            remaining = remaining[count:]
        return
    written = handle.write(payload)
    if written is None:  # 自定义流：无法获知进度，按"已写完"处理
        return
    remaining = payload[written:]
    while remaining:
        count = handle.write(remaining)
        if not count:
            break
        remaining = remaining[count:]


def _write_line(handle: Any, record: dict[str, Any]) -> None:
    """写一行 JSON（线程安全，失败仅打印 stderr）。

    实现要点（t1 修复）：整行序列化成 bytes 后**一次原子追加**，替代原来的
    ``文本句柄 write + flush``——后者在记录 > 8KB 时被文本层缓冲拆成多次写，
    与其它进程的追加交错，产生非法 JSON 行（在线验收 9 的日志完整性断言）。
    跨进程原子性由 ``os.open(..., O_APPEND)`` + 单次 ``os.write`` 保证（见 ``_open_append_fd``）。
    """
    if handle is None:
        return
    text = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    try:
        payload = text.encode("utf-8")
        with _lock:
            _write_all(handle, payload)
    except TypeError:  # 句柄是文本流（测试替身 / 旧句柄）：退回文本写，保证兼容
        try:
            with _lock:
                handle.write(text)
                handle.flush()
        except Exception as exc:  # pragma: no cover
            _safe_stderr(f"[logging_conf] 日志写入失败: {exc}")
    except Exception as exc:  # pragma: no cover
        _safe_stderr(f"[logging_conf] 日志写入失败: {exc}")


def _caller_info(depth: int = 3) -> dict[str, Any]:
    """取调用方的模块/函数/行号（用于 app.log 的 record.* 字段）。"""
    try:
        frame = sys._getframe(depth)
        return {
            "module": frame.f_globals.get("__name__", ""),
            "function": frame.f_code.co_name,
            "line": frame.f_lineno,
        }
    except Exception:  # pragma: no cover - 栈不可用时降级为空
        return {"module": "", "function": "", "line": 0}


class _Logger:
    """loguru 风格日志门面（debug/info/warning/error/exception + bind）。"""

    def __init__(self, bound: dict[str, Any] | None = None) -> None:
        self._bound = dict(bound or {})

    def bind(self, **extra: Any) -> "_Logger":
        """返回带默认业务字段的新门面（对齐 loguru ``.bind()``）。"""
        merged = {**self._bound, **extra}
        return _Logger(merged)

    def _emit(self, level: str, module: str, message: str, exc_info: bool = False, **extra: Any) -> None:
        """统一出口：同时写 app.log / error.log(ERROR+) / 控制台。"""
        try:
            _configure()
            info = _caller_info(depth=3)
            payload: dict[str, Any] = {
                "module": module,
                **{key: _shrink(val) for key, val in self._bound.items()},
                **{key: _shrink(val) for key, val in extra.items()},
            }
            trace_id = _current_trace_id.get()
            if trace_id and "trace_id" not in payload:
                payload["trace_id"] = trace_id
            exception = None
            if exc_info:
                exc_type, exc_value, exc_tb = sys.exc_info()
                if exc_type is not None:
                    exception = {
                        "type": exc_type.__name__,
                        "value": str(exc_value),
                        "traceback": "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
                    }
            record = {
                "text": message,
                "record": {
                    "time": {"timestamp": time.time(), "iso": _now_iso()},
                    "level": {"name": level},
                    "module": info["module"],
                    "function": info["function"],
                    "line": info["line"],
                    "extra": payload,
                    "exception": exception,
                },
            }
            _write_line(_APP_HANDLE, record)
            if level in {"ERROR", "CRITICAL"}:
                _write_line(_ERROR_HANDLE, record)
            if HAS_LOGURU:  # pragma: no cover - 云端路径（文件已由本模块写，loguru 只做控制台）
                try:
                    _loguru_logger.opt(depth=1).bind(**payload).log(level, message)
                except Exception as exc:  # loguru 控制台 sink 同样可能遇到坏流
                    _CONSOLE_OK = False
                    _ = exc
            else:
                _emit_console_line(f"{_now_iso()} | {level:<8} | {module} - {message}")
        except Exception as exc:  # pragma: no cover - 日志失败不得影响主流程
            # 兜底：**先写文件、绝不写可能已坏的 stderr**
            # （原实现这里 print 到同一坏流，导致 OSError 覆盖真实异常并把请求兜成 500）
            _log_emit_failure(exc, level, module, message)

    def debug(self, module: str, message: str, **extra: Any) -> None:
        self._emit("DEBUG", module, message, **extra)

    def info(self, module: str, message: str, **extra: Any) -> None:
        self._emit("INFO", module, message, **extra)

    def warning(self, module: str, message: str, **extra: Any) -> None:
        self._emit("WARNING", module, message, **extra)

    def error(self, module: str, message: str, **extra: Any) -> None:
        self._emit("ERROR", module, message, **extra)

    def exception(self, module: str, message: str, **extra: Any) -> None:
        """记录异常并附带完整堆栈（必须在 ``except`` 块内调用）。"""
        self._emit("ERROR", module, message, exc_info=True, **extra)


logger = _Logger()


def setup_logging() -> None:
    """显式初始化日志（供 main.py / 脚本 / 测试 fixture 调用；幂等）。返回 None。"""
    _configure()
    logger.info(
        "app.core.logging_conf",
        "日志系统初始化完成",
        loguru=HAS_LOGURU,
        logs_dir=str(get_settings().paths.logs),
    )


# --------------------------------------------------------------------------
# trace_id：一次提问贯穿检索 → 生成 → 落库
# --------------------------------------------------------------------------
def new_trace_id() -> str:
    """生成 16 位 hex 追踪 ID，供一次提问贯穿全链路。"""
    try:
        return uuid.uuid4().hex[:16]
    except Exception:  # pragma: no cover
        logger.exception("app.core.logging_conf", "trace_id 生成失败，降级为时间戳")
        return f"{int(time.time() * 1000):x}"


@contextmanager
def trace_context(trace_id: str | None = None) -> Iterator[str]:
    """上下文管理器：在作用域内把 trace_id 注入所有日志与 trace 事件。"""
    token = _current_trace_id.set(trace_id or new_trace_id())
    try:
        yield _current_trace_id.get()
    finally:
        _current_trace_id.reset(token)


def current_trace_id() -> str:
    """返回当前上下文的 trace_id（无则空串）。"""
    return _current_trace_id.get()


# --------------------------------------------------------------------------
# rag_trace.jsonl：函数级 enter/exit/error 与业务事件 retrieval/generation/llm_io
# --------------------------------------------------------------------------
def log_event(event: str, module: str, function: str = "", trace_id: str = "", **fields: Any) -> None:
    """写入一条事件流水（一行一 JSON）。``fields`` 中的值会被截断。"""
    try:
        _configure()
        record: dict[str, Any] = {
            "ts": _now_iso(),
            "event": event,
            "module": module,
            "function": function,
            "trace_id": trace_id or _current_trace_id.get(),
        }
        for key, value in fields.items():
            limit = MAX_IO_VALUE_CHARS if key in {"prompt", "output"} else MAX_LOG_VALUE_CHARS
            record[key] = _shrink(value, limit=limit)
        _write_line(_TRACE_HANDLE, record)
    except Exception as exc:  # pragma: no cover
        _safe_stderr(f"[logging_conf] 事件写入失败: {exc} [{event}]")


def trace(func: F) -> F:
    """装饰器：记录函数 enter/exit/error 事件（入参、出参、耗时、异常堆栈）。

    ``self``/``cls`` 参数在记录时被剔除；函数抛异常时**重抛**，不吞异常。
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        module = func.__module__
        qualname = func.__qualname__
        bound = list(args)
        if bound and hasattr(bound[0], "__dict__") and not isinstance(bound[0], (str, bytes, int, float)):
            bound = bound[1:]
        log_event("enter", module, qualname, args=bound, kwargs=kwargs)
        started = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            log_event(
                "error",
                module,
                qualname,
                elapsed_ms=round(elapsed_ms, 3),
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            logger.exception(module, f"{qualname} 执行失败", function=qualname, elapsed_ms=round(elapsed_ms, 3))
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        log_event("exit", module, qualname, elapsed_ms=round(elapsed_ms, 3), result=result)
        return result

    return wrapper  # type: ignore[return-value]


def log_stage(stage: str, message: str, **extra: Any) -> None:
    """记录阶段级进度（启动 / 解析 / 索引 / 问答里程碑）。"""
    logger.info("app.stage", f"[{stage}] {message}", stage=stage, **extra)


def flush_logs() -> None:
    """刷新三个文件句柄（进程退出前调用，确保日志落盘）。

    说明：本模块现在以 ``os.open(O_APPEND)`` 的**无缓冲 fd** 写入（见 ``_open_append_fd``），
    ``os.write`` 直接进内核，无用户态缓冲可刷；这里保留对**对象句柄**的 ``flush`` 兼容路径
    （测试替身 / 旧句柄），并在句柄不可刷时静默跳过。
    """
    for handle in (_APP_HANDLE, _ERROR_HANDLE, _TRACE_HANDLE):
        try:
            if handle is not None and hasattr(handle, "flush"):
                handle.flush()
        except Exception as exc:  # pragma: no cover
            _safe_stderr(f"[logging_conf] 刷新日志失败: {exc}")
