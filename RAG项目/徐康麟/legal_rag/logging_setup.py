# -*- coding: utf-8 -*-
"""统一日志配置：控制台 + 可选滚动文件，格式固定带 request_id。

日志格式（console 与文件一致）::

    2026-09-15 07:30:12,345 | INFO  | legal_rag.engine | req-20260915T073012-ab12cd34 | 消息

字段次序：**时间 | 级别 | 模块 | request_id | 消息**。
``request_id`` 由 :mod:`legal_rag.observability` 的 contextvars 维护，
通过 :class:`ContextFilter` 注入每一条日志记录——只要在请求入口
``bind_request()``，下游所有模块的日志都会自动带上同一个 id，
于是「按 request_id 串一次请求」只需要 ``grep req-xxxx``。

日志量控制（别把磁盘写满）
---------------------------------
* ``LOG_LEVEL`` 默认 INFO：**INFO 只记汇总**，逐条明细一律 DEBUG；
* 滚动文件 :class:`~logging.handlers.RotatingFileHandler`：**20MB × 5** 备份；
* 高频路径用 ``@traced(sample=0.01)`` 或 ``observability.sampled()`` 采样；
* 第三方库（httpx/urllib3/pymilvus…）默认提到 WARNING，可用
  ``LOG_NOISY_LEVEL`` 调整。
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .observability import current_request

__all__ = [
    "LOG_FORMAT", "DATE_FORMAT", "MAX_BYTES", "BACKUP_COUNT", "ContextFilter",
    "MillisecondFormatter", "build_formatter", "SafeRotatingFileHandler",
    "setup_utf8_stdout", "setup_logging", "get_logger", "install_context_filter",
    "resolve_log_level", "log_file_path", "current_log_file", "DEFAULT_LOG_FORMAT",
    "ROLLOVER_FAILURE_REASONS", "CONSOLE_SUBSTITUTIONS", "console_safe",
]

#: 时间 | 级别 | 模块 | request_id | 消息
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-5s | %(name)s | "
    "%(legal_rag_request_id)s | %(message)s"
)
DEFAULT_LOG_FORMAT = LOG_FORMAT
#: ``%(asctime)s`` 的日期格式：``%f`` 展开成毫秒，最终形如 ``2026-09-15 07:30:12,345``
DATE_FORMAT = "%Y-%m-%d %H:%M:%S,%f"

#: 滚动文件策略：20MB × 5 个备份（最多占用约 120MB）
MAX_BYTES = 20 * 1024 * 1024
BACKUP_COUNT = 5

#: 第三方库噪声默认压制级别
NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "pymilvus", "pymilvus.grpc_gen",
    "grpc", "asyncio", "multipart", "python_multipart", "chromadb",
    "sentence_transformers", "transformers", "modelscope", "filelock",
    "uvicorn.access", "watchfiles",
)

#: 尚未绑定任何 handler 时也要能输出的兜底哨兵
MISSING = "-"

_LOG_HANDLER_MARK = "_legal_rag_handler"
_FILE_HANDLER_MARK = "_legal_rag_file_handler"
_configured = False
_current_file_path: str | None = None


class ContextFilter(logging.Filter):
    """把 contextvars 里的 request_id / session_id / user_id 注入每条 LogRecord。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - 标准接口名
        try:
            ctx = current_request()
            record.legal_rag_request_id = ctx.request_id or MISSING  # type: ignore[attr-defined]
            record.legal_rag_session_id = ctx.session_id or MISSING  # type: ignore[attr-defined]
            record.legal_rag_user_id = ctx.user_id or MISSING  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - 过滤器绝不能抛异常
            record.legal_rag_request_id = MISSING  # type: ignore[attr-defined]
            record.legal_rag_session_id = MISSING  # type: ignore[attr-defined]
            record.legal_rag_user_id = MISSING  # type: ignore[attr-defined]
        return True


_CONTEXT_FILTER = ContextFilter()


def install_context_filter(logger: logging.Logger) -> logging.Logger:
    """给某个 logger（及其 handler）挂上上下文过滤器，幂等。"""
    if not any(isinstance(f, ContextFilter) for f in logger.filters):
        logger.addFilter(_CONTEXT_FILTER)
    for handler in logger.handlers:
        if not any(isinstance(f, ContextFilter) for f in handler.filters):
            handler.addFilter(_CONTEXT_FILTER)
    return logger


def setup_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，把三个标准流都切到 UTF-8，避免中文乱码。

    stdin 也要处理：交互式输入中文时，若 stdin 按 GBK 解码同样会乱码。
    """
    for name in ("stdout", "stderr", "stdin"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


#: 常见"好看但不能进 GBK"的字符 -> 控制台安全写法。
#: 约定来自 `tests/test_console_encoding.py`：**打印/日志里的文本一律 GBK 安全**，
#: emoji 只留给写进文件的报告（.md / .json）。``⇒`` 这类箭头换成 ASCII ``->``。
CONSOLE_SUBSTITUTIONS = {
    "⇒": "->", "→": "->", "←": "<-", "⚠️": "[注意]", "⚠": "[注意]",
    "✅": "[OK]", "❌": "[FAIL]", "⏭": "[SKIP]", "⭐": "*",
}

#: 这些是**零宽/变体选择符**：它们只修饰前一个字符，单独打出来没意义。
#: 直接丢掉，否则 `⚠️` 会变成 `[注意]?`（看着像 bug）。
CONSOLE_DROP = {"\ufe0f", "\ufe0e", "\u200d", "\u200b"}


def console_safe(text: object) -> str:
    """把一段文本变成**GBK 控制台一定打得出来**的样子。

    为什么需要它（真事）：`scripts/eval_answers.py` 会把**模型答案的前若干字**直接打印出来；
    答案里出现一个 emoji（模型很爱写）时，GBK 控制台会 **UnicodeEncodeError**
    => **整行丢失、退出码变 1** => 排查时被误判成"脚本自己失败了"。
    静态闸门只能管"写在 print 里的字面量"，管不到"来自数据/模型的字符"，所以补这道运行期保险。

    规则：零宽/变体选择符丢掉 -> 常见符号按 :data:`CONSOLE_SUBSTITUTIONS` 换可读写法 ->
    剩下的逐字符试编码，编不出来的退化成 ``?``。**永不抛异常**。
    """
    out: list[str] = []
    for char in str(text):
        if char in CONSOLE_DROP:
            continue
        mapped = CONSOLE_SUBSTITUTIONS.get(char)
        if mapped is not None:
            out.append(mapped)
            continue
        try:
            char.encode("gbk")
        except (UnicodeEncodeError, LookupError):
            out.append("?")
            continue
        out.append(char)
    return "".join(out)


def resolve_log_level(level: int | str | None = None, default: int = logging.INFO) -> int:
    """把 ``LOG_LEVEL``（数字或 INFO/DEBUG/... 名称）解析成 logging 级别常量。"""
    if level is None:
        level = os.environ.get("LOG_LEVEL") or ""
    if isinstance(level, int):
        return level
    text = str(level).strip()
    if not text:
        return default
    if text.isdigit():
        return int(text)
    return getattr(logging, text.upper(), default)


def log_file_path(log_dir: str | os.PathLike[str] | None = None,
                  filename: str | None = None) -> Path:
    """解析日志文件路径（LOG_DIR 环境变量可覆盖，默认项目根 logs/app.log）。"""
    if log_dir is None:
        log_dir = os.environ.get("LOG_DIR") or "logs"
    base = Path(log_dir)
    if not base.is_absolute():
        base = Path(__file__).resolve().parents[1] / base
    return base / (filename or os.environ.get("LOG_FILE") or "app.log")


_ENV_FALSE = {"0", "false", "no", "off", "n", "f"}


def _file_enabled(log_to_file: bool | None) -> bool:
    if log_to_file is not None:
        return bool(log_to_file)
    raw = os.environ.get("LOG_TO_FILE", "")
    if not raw:
        return True
    return raw.strip().lower() not in _ENV_FALSE


def _clear_our_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, _LOG_HANDLER_MARK, False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover
                pass


#: 轮转失败的归类取值（供日志/探针/后续口径登记使用）
ROLLOVER_FAILURE_REASONS = ("winerror32", "permission", "other")


def _classify_rollover_error(exc: BaseException) -> str:
    """把轮转异常归类：Windows 文件占用（WinError 32）单列，便于排查。"""
    text = f"{type(exc).__name__}: {exc}"
    if "WinError 32" in text or "being used by another process" in text:
        return "winerror32"
    if isinstance(exc, PermissionError):
        return "permission"
    return "other"


class SafeRotatingFileHandler(RotatingFileHandler):
    """滚动文件 handler：**多实例共用同一文件时不丢日志、不刷屏**（t46/t69）。

    标准 ``RotatingFileHandler`` 在 Windows 上多实例共用同一文件时有两个硬伤：

    1. **Windows 不允许重命名"仍被另一个进程打开"的文件** => ``doRollover()`` 抛
       ``PermissionError: [WinError 32]``，而异常发生在"写这一条"之前 => **这条记录丢失**；
    2. 异常经 ``logging.Handler.handle`` 落到 ``handleError`` => **每条**都往 stderr 打一段
       traceback => 刷屏（实测两实例各写 400 行：丢 744/800 行、stderr 里 86 次 WinError 32、
       93 段 traceback）。

    本类的处置（要点：**不再长期持有文件句柄** + **跨进程互斥**）：

    * **每次写入现开现关**（``open(..., "a")``），进程在两次写之间**不持有**该文件 =>
      Windows 上的重命名不再被占用挡住；
    * **跨进程互斥**：用旁路锁文件 ``<log>.lock``（Windows ``msvcrt.locking`` /
      POSIX ``fcntl.flock``）把「查大小 → 轮转 → 追加」整段串行化 => 两个实例不会互相
      顶掉对方刚轮转出来的备份（那是"看起来没报错但内容丢失"的隐蔽路径）；
    * 轮转**失败可见一次**（写进日志文件 + stderr 各一次，附归类与原因），**不再刷屏**；
    * 轮转失败**不放弃写入**：直接追加到当前文件（暂时超过 ``maxBytes``，对方释放后
      下一次写入会自然重试轮转）=> **任何实例都不丢已写日志**；恢复后再报一次 INFO。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: 本进程内轮转失败累计次数（探针/测试直接读）
        self.rollover_failures = 0
        self.rollover_failure_reason = ""
        self._failure_reported = False
        self._recovered = False
        self._internal_errors = 0
        self._lock_path = f"{self.baseFilename}.lock"
        self.stream = None                     # 刻意不留常驻句柄（见类文档第 1 点）

    # ---------- 跨进程锁 ----------
    @contextlib.contextmanager
    def _file_lock(self, timeout: float = 10.0):
        """旁路锁文件上的**跨进程互斥**（拿不到锁也不阻塞业务：超时即放行写入）。"""
        handle = None
        try:
            handle = open(self._lock_path, "a+b")
            deadline = time.monotonic() + timeout
            while True:
                try:
                    self._lock(handle)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("等待日志轮转锁超时")
                    time.sleep(0.005)
            yield
        except Exception:                      # noqa: BLE001 - 锁不可用时退化为"直接写"
            yield
        finally:
            if handle is not None:
                try:
                    self._unlock(handle)
                finally:
                    handle.close()

    @staticmethod
    def _lock(handle) -> None:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:                                  # pragma: no cover - POSIX 分支
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle) -> None:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:                              # pragma: no cover - POSIX 分支
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:                      # noqa: BLE001 - 解锁失败无伤大雅
            pass

    # ---------- 写入 ----------
    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102 - 标准接口
        try:
            message = self.format(record)
        except Exception:                      # noqa: BLE001 - 格式化失败不抛
            message = str(record.getMessage())
        try:
            with self._file_lock():
                self._rotate_if_needed(len(message.encode("utf-8", "replace")) + 1)
                self._append(message)
        except Exception:                      # noqa: BLE001 - 任何异常都不许丢消息
            self._internal_errors += 1
            self._append(message)              # 兜底：不加锁直写

    def _append(self, message: str) -> None:
        with open(self.baseFilename, "a", encoding=self.encoding or "utf-8",
                  errors="replace", newline="") as handle:
            handle.write(message + "\n")

    def _rotate_if_needed(self, incoming: int) -> None:
        """按 ``maxBytes`` 判断是否需要轮转（在**持锁**状态下调用）。"""
        if self.maxBytes <= 0 or self.backupCount <= 0:
            return
        try:
            size = os.path.getsize(self.baseFilename)
        except OSError:
            return
        if size + incoming < self.maxBytes:
            return
        try:
            self._rotate_locked()
        except (PermissionError, OSError, ValueError) as exc:
            self._note_rollover_failure(exc)
            return                             # 关键：**照样写**，绝不因为轮转失败丢这条
        if self._failure_reported and not self._recovered:
            self._recovered = True
            self._notify("INFO", "日志轮转已恢复：多实例共用日志文件时可以正常轮转了")

    def _rotate_locked(self) -> None:
        """持锁轮转（标准腾挪算法）：此刻没有别的轮转者，不会顶掉别人的备份。"""
        if os.path.exists(self.baseFilename):
            for index in range(self.backupCount - 1, 0, -1):
                source = self.rotation_filename(f"{self.baseFilename}.{index}")
                target = self.rotation_filename(f"{self.baseFilename}.{index + 1}")
                if os.path.exists(source):
                    if os.path.exists(target):
                        os.remove(target)
                    os.rename(source, target)
            target = self.rotation_filename(f"{self.baseFilename}.1")
            if os.path.exists(target):
                os.remove(target)
            os.rename(self.baseFilename, target)

    def doRollover(self) -> None:  # noqa: N802 - 标准接口名
        """保留标准入口（有人直接调它时不抛）：失败只记一次可见告警。"""
        try:
            with self._file_lock():
                self._rotate_locked()
        except (PermissionError, OSError, ValueError) as exc:
            self._note_rollover_failure(exc)

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802 - 标准接口名
        """**不往 stderr 打 traceback**（那正是多实例下刷屏的来源），只计数。"""
        self._internal_errors += 1

    def _note_rollover_failure(self, exc: BaseException) -> None:
        self.rollover_failures += 1
        reason = _classify_rollover_error(exc)
        self.rollover_failure_reason = reason
        if self._failure_reported:
            return                             # **只报一次**，同一原因重复失败不再刷屏
        self._failure_reported = True
        self._notify(
            "WARNING",
            f"日志轮转失败（归类={reason}）：{type(exc).__name__}: {exc} —— "
            f"**已跳过本次轮转、继续写当前文件**（不丢日志）；多实例共用 "
            f"{self.baseFilename} 属预期，占用释放后会自动恢复")

    def _notify(self, level: str, message: str) -> None:
        """把"轮转失败/恢复"写进日志文件（不经本 handler，避免递归）+ stderr 一次。"""
        line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')},{int((time.time() % 1) * 1000):03d} | "
                f"{level:<5s} | legal_rag.logging_setup | {MISSING} | {message}")
        try:
            self._append(line)
        except Exception:                      # noqa: BLE001 - 文件不可写时退化为 stderr
            pass
        try:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
        except Exception:                      # noqa: BLE001
            pass

    def close(self) -> None:
        try:
            super().close()
        finally:
            self.stream = None                 # 本类不持有常驻句柄


class MillisecondFormatter(logging.Formatter):
    """与标准 Formatter 行为一致，只额外支持 ``datefmt`` 里的 ``%f`` 毫秒占位符。

    内置实现把 ``datefmt`` 交给 ``time.strftime``，无法表达毫秒；这里接管
    :meth:`formatTime`：``%f`` 展开成三位毫秒（与 logging 默认的 ``%s,%03d`` 一致），
    于是日志行是 ``2026-09-15 07:30:12,345``。``datefmt`` 不含 ``%f`` 时行为不变。
    """

    _SENTINEL = "\x00MS\x00"

    def __init__(self, fmt=None, datefmt=None, style="%", **kwargs):
        self._wants_ms = bool(datefmt) and "%f" in datefmt
        if self._wants_ms:
            # 用等长哨兵占位：格式串里 ``%f`` 前面已写逗号，替换值只给三位毫秒
            datefmt = datefmt.replace("%f", "{MS}")
        super().__init__(fmt, datefmt, style, **kwargs)

    def formatTime(self, record, datefmt=None):  # noqa: N802 - 标准接口名
        """自己实现时间格式化，避免标准实现再追加一次毫秒造成 ``,,031``。"""
        fmt = datefmt or self.datefmt
        if fmt:
            text = time.strftime(fmt, time.localtime(record.created))
        else:
            text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        if self._wants_ms and "{MS}" in text:
            millis = int(getattr(record, "msecs", (record.created % 1) * 1000) or 0)
            text = text.replace("{MS}", f"{millis:03d}")
        return text


def build_formatter(log_format: str = LOG_FORMAT,
                    date_format: str = DATE_FORMAT) -> logging.Formatter:
    """构造统一格式的 Formatter（支持 datefmt 里的 ``%f`` 毫秒占位符）。"""
    return MillisecondFormatter(log_format, datefmt=date_format)


def setup_logging(
    level: int | str | None = None,
    log_dir: str | os.PathLike[str] | None = None,
    log_to_file: bool | None = None,
    log_format: str = LOG_FORMAT,
    date_format: str = DATE_FORMAT,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
    force: bool = False,
) -> logging.Logger:
    """配置根日志。幂等：重复调用只重挂自己装的 handler，不叠加。

    :param level: 级别；缺省读 ``LOG_LEVEL``（默认 INFO）；
    :param log_dir: 日志目录；缺省读 ``LOG_DIR``（默认 ``logs/``，相对项目根）；
    :param log_to_file: 是否写滚动文件；缺省读 ``LOG_TO_FILE``（默认开）；
    :param max_bytes/backup_count: 滚动策略，默认 20MB × 5；
    :param force: True 时先清掉其它 handler（测试里想拿到干净输出时用）。
    :return: root logger
    """
    global _configured, _current_file_path
    setup_utf8_stdout()

    root = logging.getLogger()
    resolved = resolve_log_level(level)

    if force and _configured:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover
                pass
    _clear_our_handlers(root)

    formatter = build_formatter(log_format, date_format)

    if not _configured or force:
        root.setLevel(resolved)

    # 控制台：stderr（保留原有行为，避免污染 stdout 的结构化输出）
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console.addFilter(_CONTEXT_FILTER)
    setattr(console, _LOG_HANDLER_MARK, True)
    console.setLevel(resolved)
    root.addHandler(console)

    # 可选滚动文件
    _current_file_path = None
    if _file_enabled(log_to_file):
        path = log_file_path(log_dir)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = SafeRotatingFileHandler(
                path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(_CONTEXT_FILTER)
            setattr(file_handler, _LOG_HANDLER_MARK, True)
            setattr(file_handler, _FILE_HANDLER_MARK, True)
            file_handler.setLevel(resolved)
            root.addHandler(file_handler)
            _current_file_path = str(path)
        except (OSError, ValueError) as exc:  # 磁盘满/无权限：降级到仅控制台
            root.warning("日志文件初始化失败，降级为仅控制台输出：%s", exc)

    noise = resolve_log_level(os.environ.get("LOG_NOISY_LEVEL") or "WARNING",
                              default=logging.WARNING)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noise)

    root.addFilter(_CONTEXT_FILTER)
    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """取模块 logger；未初始化时先做一次默认初始化，保证首条日志就有格式与 request_id。"""
    if not _configured:
        setup_logging()
    return logging.getLogger(name)


def current_log_file() -> str | None:
    """当前滚动文件路径（未启用文件日志时为 None）。"""
    return _current_file_path
