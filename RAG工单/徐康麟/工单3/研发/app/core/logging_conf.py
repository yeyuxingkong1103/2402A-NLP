# -*- coding: utf-8 -*-
"""工单3 自实现结构化 JSON 日志（设计/接口设计.md §3.2、§4、§5.1 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

为什么自实现：本机 ``loguru`` 目录存在（元数据 0.7.3）但 ``import loguru``
抛 ``ModuleNotFoundError: No module named 'win32_setctime'``，**实际不可用**，
且本机断网无法安装 —— 因此按硬性要求自实现 loguru 风格的结构化 logger。

落盘（追加模式，每行一个 JSON 对象，``ensure_ascii=False``）：
    * ``部署/日志/app.log``        全部 >= RAG_LOG__LEVEL 事件
    * ``部署/日志/error.log``      ERROR/CRITICAL（含 error.traceback）
    * ``部署/日志/rag_trace.jsonl`` 链路事件（retrieval./generation./llm./citation./answerability./qa./eval.question）

并发写入策略（T18 修复，多进程安全 —— 详见 部署/日志/日志字段说明.md「并发写入策略」）：
    * 句柄以 **``open(path, "ab", buffering=0)``（无缓冲二进制 + O_APPEND）** 打开，
      **每行只调用一次 ``handle.write(bytes)``**；
    * 写入前后加 **跨进程写锁**（Windows ``msvcrt.locking`` 第 0 字节 / POSIX ``fcntl.flock``）——
      实测「只做单次 write、不加锁」在 Windows 上仍会交错（内核把大 write 拆块），加锁后压测非法行 = 0；
    * 进程内由 ``self._lock`` 串行化（同一句柄多线程安全）；跨进程靠文件锁 + O_APPEND 原子追加双重保证；
    * 旧实现（文本模式 ``open(..., "a")`` + write + flush）在长记录上会被拆成多次 ``write`` → 行被截断/拼接。

设计要点：
    * 公共字段固定：ts/level/event/func/module/run_id/trace_id/span_id/stage/pid/thread（§4.1）；
    * 函数入口/出口成对写 ``func.enter`` / ``func.exit``（异常写 ``func.error`` + traceback）；
    * 摘要规则见 §4.4，禁止空 inputs/outputs、禁止只写「开始/结束」；
    * 线程安全（每行 write + flush 持锁）；初始化失败抛 LogSetupError，不静默。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .errors import LogSetupError

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

LEVELS: dict[str, int] = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

# 需要同时写入 rag_trace.jsonl 的事件族前缀（§5.1）
TRACE_PREFIXES: tuple[str, ...] = (
    "retrieval.", "generation.", "llm.", "citation.", "answerability.", "qa.", "eval.question",
)

_SPAN_COUNTER = 0


def _next_span_id() -> str:
    """生成短 span_id（同进程内单调递增 + 时间后缀）。"""
    global _SPAN_COUNTER
    _SPAN_COUNTER += 1
    return f"s{_SPAN_COUNTER:05d}{int(time.perf_counter() * 1000) % 1000:03d}"


def _now_iso() -> str:
    """本地时区 ISO-8601（毫秒精度）。"""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _open_append_atomic(path: Path) -> Any:
    """以 **O_APPEND + 无缓冲二进制** 打开日志文件（多进程并发追加的前提）。

    为什么不用 ``open(path, "a", encoding="utf-8")``：
        * 文本模式 = TextIOWrapper + BufferedWriter；超过写缓冲（``io.DEFAULT_BUFFER_SIZE`` = 8 KB）的
          长记录会被拆成多次底层 ``write``，两次写之间可能插入其它进程的写入 → 行被**截断/拼接**；
        * ``buffering=0`` 的二进制句柄是 ``io.FileIO``，每次 ``write`` 直接落到一次 ``write(2)``/``WriteFile``；
          配合 ``O_APPEND``（"ab" 隐含），「整行 + 换行」在同一次系统调用内追加。
    """
    return open(path, "ab", buffering=0)  # noqa: SIM115 —— 句柄由 StructuredLogger 持有并在 close() 释放


def _acquire_write_lock(handle: Any) -> bool:
    """对已打开句柄加**跨进程写锁**（Windows：``msvcrt`` 字节锁；POSIX：``fcntl.flock``）。

    为什么「单次 write」还不够（T18 实测）：Windows 内核会把较大的 ``WriteFile`` 拆成多个块写盘，
    两个进程的块仍可互相插入 —— 4 进程 × 400 行 × 16 KB 压测下，**只用单次 write 仍有 131/1211 行非法**；
    加锁后同一参数下为 **0/1600**。因此锁是必需的，不是锦上添花。

    * 所有进程都锁**日志文件的第 0 字节**，因此拿到锁的进程独占地完成「整行 + 换行」的追加；
    * 句柄没有 ``fileno``（例如测试注入的假句柄）时无法加锁，按设计跳过并返回 False；
    * 锁竞争失败不抛异常到调用方：由 ``_write`` 以**显式降级**消息提示并继续 best-effort 写入。
    """
    fileno = getattr(handle, "fileno", None)
    if not callable(fileno):
        return False
    try:
        fd = int(fileno())
    except (OSError, ValueError, TypeError):
        return False
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)      # 阻塞式：最多重试 ~10 s，超时抛 OSError
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)
    return True


def _release_write_lock(handle: Any) -> None:
    """释放 ``_acquire_write_lock`` 加的锁（失败时同样显式提示，不静默）。"""
    fileno = getattr(handle, "fileno", None)
    if not callable(fileno):
        return
    fd = int(fileno())
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# 摘要规则（§4.4）
# ---------------------------------------------------------------------------
def _sha1_8(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:8]


def summarize_input(value: Any, *, limit: int = 120) -> Any:
    """按 §4.4 规则生成输入摘要（str 截断 / 集合取样 / dataclass 取关键字段）。"""
    return _summarize(value, limit=limit, depth=0)


def summarize_output(value: Any, *, limit: int = 8) -> Any:
    """按 §4.4 规则生成输出摘要（列表最多 8 项，超出写 "...": n-8）。"""
    return _summarize(value, limit=limit, depth=0)


def _summarize(value: Any, *, limit: int, depth: int) -> Any:
    if depth > 3:
        return "<深度截断>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return {"chars": len(value), "head": value[:limit], "sha1": _sha1_8(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        items = list(value.items())[:12]
        return {str(k): _summarize(v, limit=limit, depth=depth + 1) for k, v in items}
    if is_dataclass(value) and not isinstance(value, type):
        data = asdict(value)
        keys = ("chunk_id", "file_name", "page", "type", "char_count", "table_id", "table_count")
        picked = {k: data[k] for k in keys if k in data}
        if not picked:
            picked = {k: _summarize(v, limit=limit, depth=depth + 1) for k, v in list(data.items())[:8]}
        return picked
    if isinstance(value, (list, tuple, set)):
        seq: Sequence[Any] = list(value)
        head = [_summarize(v, limit=limit, depth=depth + 1) for v in seq[:limit]]
        if len(seq) > limit:
            head.append({"...": len(seq) - limit})
        return head
    text = repr(value)
    if len(text) <= limit:
        return text
    return {"repr_head": text[:limit], "sha1": _sha1_8(text)}


def mask_secret(secret: str) -> str:
    """密钥脱敏：保留前 4 后 4。"""
    if not secret:
        return ""
    if len(secret) < 12:
        return "*" * len(secret)
    return f"{secret[:4]}{'*' * 8}{secret[-4:]}"


# ---------------------------------------------------------------------------
# 日志跨度（with 语法）
# ---------------------------------------------------------------------------
class LogSpan:
    """由 ``StructuredLogger.enter()`` 返回：负责写 ``func.exit`` / ``func.error``。"""

    def __init__(self, logger: "StructuredLogger", func: str, inputs: Mapping[str, Any], stage: str) -> None:
        self._logger = logger
        self.func = func
        self.inputs = dict(inputs)
        self.stage = stage
        self.span_id = _next_span_id()
        self._t0 = time.perf_counter()
        self._outputs: dict[str, Any] | None = None

    def set_output(self, output: Mapping[str, Any]) -> None:
        """记录出口摘要（必须非空，否则 ``func.exit`` 记为 WARN）。

        健壮性：调用方可能传 list/scalar（实测踩过 ``dict([{...}])`` 抛 ValueError，
        让**日志参数形状把业务打挂**），因此这里对非 Mapping 输入做显式包装，绝不因日志参数抛异常。
        """
        if isinstance(output, Mapping):
            self._outputs = summarize_output(output)
        else:
            self._outputs = summarize_output({"value": output})

    def note(self, event: str, /, **fields: Any) -> None:
        """在跨度内写一条自定义事件（沿用同一 span_id）。"""
        self._logger.log_event(event, func=self.func, span_id=self.span_id, stage=self.stage, **fields)

    def __enter__(self) -> "LogSpan":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        elapsed_ms = round((time.perf_counter() - self._t0) * 1000, 2)
        if exc is not None:
            self._logger.error(self.func, exc, inputs=self.inputs, elapsed_ms=elapsed_ms,
                               span_id=self.span_id, stage=self.stage)
            return False  # 传播异常，绝不吞掉
        outputs = self._outputs if self._outputs else {}
        level = "INFO" if outputs else "WARNING"
        self._logger.log_event(
            "func.exit", level=level, func=self.func, span_id=self.span_id, stage=self.stage,
            inputs=self.inputs, outputs=outputs, elapsed_ms=elapsed_ms,
            note=None if outputs else "未设置 outputs（调用方必须 set_output）",
        )
        return False


# ---------------------------------------------------------------------------
# 结构化 logger
# ---------------------------------------------------------------------------
class StructuredLogger:
    """线程安全的 JSON Lines logger；``bind`` 后携带固定字段。"""

    def __init__(
        self,
        *,
        run_id: str,
        log_dir: Path,
        level: str = "INFO",
        module: str = "app",
        trace_id: str | None = None,
        stage: str = "core",
        extra: Mapping[str, Any] | None = None,
        echo_error: bool = True,
    ) -> None:
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.level = (level or "INFO").upper()
        self.module = module
        self.trace_id = trace_id
        self.stage = stage
        self.extra: dict[str, Any] = dict(extra or {})
        self.echo_error = echo_error
        self._lock = threading.Lock()
        self._handles: dict[str, Any] = {}
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            # T18：无缓冲二进制 + O_APPEND（见模块 docstring「并发写入策略」）；不用文本模式，
            # 避免 BufferedWriter 把长行拆成多次 write 后被其它进程从中间插入。
            self._handles = {
                "app": _open_append_atomic(self.log_dir / "app.log"),
                "error": _open_append_atomic(self.log_dir / "error.log"),
                "trace": _open_append_atomic(self.log_dir / "rag_trace.jsonl"),
            }
        except OSError as exc:
            raise LogSetupError(f"日志文件打开失败：{self.log_dir}（{exc}）",
                                detail={"log_dir": str(self.log_dir)}) from exc

    # -- 绑定 ------------------------------------------------------------
    def bind(self, **fields: Any) -> "StructuredLogger":
        """返回携带额外固定字段的新 logger 句柄（共享同一组文件句柄）。"""
        clone = StructuredLogger.__new__(StructuredLogger)
        clone.__dict__.update(self.__dict__)
        merged = dict(self.extra)
        merged.update(fields)
        clone.extra = merged
        if "module" in fields:
            clone.module = str(fields["module"])
        if "stage" in fields:
            clone.stage = str(fields["stage"])
        if "trace_id" in fields:
            clone.trace_id = fields["trace_id"]
        clone._lock = self._lock  # 共享锁，保证多线程写同一文件句柄安全
        clone._handles = self._handles
        return clone

    def with_trace(self, trace_id: str | None) -> "StructuredLogger":
        """派生一个绑定 trace_id 的 logger（一次问答链路）。"""
        return self.bind(trace_id=trace_id)

    # -- 写事件 ----------------------------------------------------------
    def log_event(self, event: str, /, level: str = "INFO", **fields: Any) -> None:
        """写一条结构化事件（公共字段 + 业务字段）。"""
        lvl = (level or "INFO").upper()
        record: dict[str, Any] = {
            "ts": _now_iso(),
            "level": lvl,
            "event": str(event),
            "func": str(fields.pop("func", self.extra.get("func", ""))) or None,
            "module": str(fields.pop("module", self.module)),
            "run_id": str(fields.pop("run_id", self.run_id)),
            "trace_id": fields.pop("trace_id", self.trace_id),
            "span_id": fields.pop("span_id", None),
            "stage": str(fields.pop("stage", self.stage)),
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            "work_order": WORK_ORDER,
        }
        for key, value in self.extra.items():
            record.setdefault(key, value)
        record.update({k: v for k, v in fields.items()})
        line = json.dumps(record, ensure_ascii=False, default=str)

        is_trace = str(event).startswith(TRACE_PREFIXES)
        with self._lock:
            try:
                if LEVELS.get(lvl, 20) >= LEVELS.get(self.level, 20):
                    self._write("app", line)
                if LEVELS.get(lvl, 20) >= LEVELS["ERROR"]:
                    self._write("error", line)
                if is_trace:
                    self._write("trace", line)
            except OSError as exc:
                # 显式降级：日志写失败必须让人看见，绝不静默
                print(f"[logging_conf 降级] 写日志失败：{type(exc).__name__}: {exc}", flush=True)
        if self.echo_error and LEVELS.get(lvl, 20) >= LEVELS["ERROR"]:
            print(f"[{lvl}] {line}", flush=True)

    def _write(self, kind: str, line: str) -> None:
        """把**一整行**追加到目标文件（多进程安全：O_APPEND + 单次 write + 跨进程锁）。

        * 先编码成 bytes（``errors="backslashreplace"``：极端非法代理字符也不会让写入失败）；
        * 持有跨进程写锁后 ``handle.write(payload)`` 只调用一次 —— 这是「整行不被其它进程从中间插入」的关键；
        * 写入字节数不足（部分写/磁盘满）视为失败：显式降级打印 stderr 并摘掉该句柄，绝不静默；
        * 加锁失败（竞争超时/句柄无 fileno）不静默：打印显式降级消息后 best-effort 写入并保留句柄。
        """
        handle = self._handles.get(kind)
        if handle is None:
            return
        payload = (line + "\n").encode("utf-8", errors="backslashreplace")
        locked = False
        try:
            try:
                locked = _acquire_write_lock(handle)
            except OSError as exc:
                # 显式降级：拿不到锁也要让人看见，并说明后续是 best-effort 写入（不静默）
                print(f"[logging_conf 降级] 写 {kind} 前加锁失败（{type(exc).__name__}: {exc}），"
                      f"本次改为无锁写入（仍保留句柄）", file=sys.stderr, flush=True)
            written = handle.write(payload)
            if written is not None and int(written) != len(payload):
                raise OSError(f"部分写入：{written}/{len(payload)} 字节")
            handle.flush()
        except (OSError, ValueError, TypeError) as exc:
            # 显式降级：句柄已关闭/不可写时必须让人看见，并把坏句柄摘掉（不静默）
            print(f"[logging_conf 降级] 写 {kind} 失败（已停用该句柄）：{type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            self._handles.pop(kind, None)
        finally:
            if locked:
                try:
                    _release_write_lock(handle)
                except OSError as exc:
                    print(f"[logging_conf 降级] 释放 {kind} 写锁失败：{type(exc).__name__}: {exc}",
                          file=sys.stderr, flush=True)

    # -- 函数入口/出口 ---------------------------------------------------
    def enter(self, func: str, /, inputs: Mapping[str, Any] | None = None, **fields: Any) -> LogSpan:
        """写 ``func.enter`` 并返回 LogSpan（配合 ``with`` 使用）。"""
        summary = summarize_input(dict(inputs or {}))
        span = LogSpan(self, func, summary, str(fields.pop("stage", self.stage)))
        self.log_event("func.enter", func=func, span_id=span.span_id, stage=span.stage,
                       inputs=summary, **fields)
        return span

    def error(
        self,
        func: str,
        exc: BaseException,
        /,
        inputs: Mapping[str, Any] | None = None,
        elapsed_ms: float | None = None,
        **fields: Any,
    ) -> None:
        """写 ``func.error``（含 error.type/message/traceback 与耗时）。"""
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.log_event(
            "func.error", level="ERROR", func=func,
            inputs=summarize_input(dict(inputs or {})),
            elapsed_ms=round(elapsed_ms, 2) if elapsed_ms is not None else None,
            error={"type": type(exc).__name__, "message": str(exc), "traceback": tb},
            **fields,
        )

    # -- 生命周期 --------------------------------------------------------
    def close(self) -> None:
        """关闭文件句柄（幂等）。

        注意：``bind()`` 产生的句柄与本对象**共享同一个 _handles 字典**，
        因此这里必须 ``clear()`` 原地清空，而不能 ``self._handles = {}`` 重新绑定 ——
        否则克隆句柄仍持有已关闭的文件对象，写入时抛 ``ValueError: I/O operation on closed file``。
        """
        with self._lock:
            for kind, handle in list(self._handles.items()):
                try:
                    handle.close()
                except (OSError, ValueError) as exc:  # 显式降级：关不掉也要让人看见
                    print(f"[logging_conf 降级] 关闭 {kind} 失败：{type(exc).__name__}: {exc}",
                          file=sys.stderr, flush=True)
            self._handles.clear()

    def file_paths(self) -> dict[str, str]:
        """返回三个日志文件的绝对路径。"""
        return {
            "app": str(self.log_dir / "app.log"),
            "error": str(self.log_dir / "error.log"),
            "trace": str(self.log_dir / "rag_trace.jsonl"),
        }


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------
_STATE: dict[str, Any] = {"logger": None, "initializing": False, "fallback": None}
_LOCK = threading.RLock()


def _fallback_logger() -> StructuredLogger:
    """重入保护用的降级 logger：只写 stderr，不落盘（避免递归初始化）。"""
    clone = StructuredLogger.__new__(StructuredLogger)
    clone.run_id = "r-bootstrap"
    clone.log_dir = Path(".")
    clone.level = "INFO"
    clone.module = "logging_conf"
    clone.trace_id = None
    clone.stage = "config"
    clone.extra = {}
    clone.echo_error = True
    clone._lock = threading.RLock()
    clone._handles = {}
    return clone


def setup_logging(cfg: Any = None, *, run_id: str | None = None, force: bool = False) -> StructuredLogger:
    """初始化全局 logger（幂等）；目录创建失败抛 ``LogSetupError``。"""
    global _STATE
    with _LOCK:
        existing = _STATE.get("logger")
        if isinstance(existing, StructuredLogger) and not force:
            return existing
        if _STATE.get("initializing"):
            # 重入（config 加载期间写日志/报错）：返回 stderr-only 句柄，绝不递归
            fallback = _STATE.get("fallback")
            if fallback is None:
                fallback = _fallback_logger()
                _STATE["fallback"] = fallback
            return fallback
        _STATE["initializing"] = True
    try:
        if cfg is None:
            from .config import get_config

            cfg = get_config()
        logger = StructuredLogger(
            run_id=run_id or getattr(cfg, "run_id", "r-bootstrap"),
            log_dir=Path(getattr(getattr(cfg, "paths", None), "log_dir", Path("部署/日志"))),
            level=str(getattr(cfg, "log_level", "INFO")),
            module="app",
            stage="core",
        )
        with _LOCK:
            _STATE["logger"] = logger
        raw_dir = Path(getattr(getattr(cfg, "paths", None), "raw_dir", Path("研发/data/raw")))
        files: list[str] = []
        try:
            files = sorted(p.name for p in raw_dir.glob("*.pdf")) if raw_dir.is_dir() else []
        except OSError as exc:  # 显式降级：清单读不到要留痕
            logger.log_event("config.discover_skipped", level="WARNING", raw_dir=str(raw_dir),
                             error_type=type(exc).__name__, message=str(exc))
        logger.log_event(
            "run.start", run_id=logger.run_id, work_order=WORK_ORDER, python=_py_version(),
            executable=os.sys.executable, cwd=os.getcwd(), raw_dir=str(raw_dir),
            raw_files=files, log_files=logger.file_paths(), cfg=cfg.to_dict() if hasattr(cfg, "to_dict") else None,
        )
        return logger
    finally:
        with _LOCK:
            _STATE["initializing"] = False


def _py_version() -> str:
    import sys

    return sys.version.split()[0]


def get_logger(name: str, **fields: Any) -> StructuredLogger:
    """取（必要时初始化）logger，并绑定 ``module`` 等固定字段。"""
    logger = setup_logging()
    bound = dict(fields)
    bound.setdefault("module", name)
    return logger.bind(**bound)


def log_call(func_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """装饰器：等价于 ``with logger.enter(func_name, inputs) as span: ...``。

    被装饰函数的首个位置参数若为 Mapping，会作为 inputs；返回值摘要写入 outputs。
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__.rsplit(".", 1)[-1])
            inputs: dict[str, Any] = {}
            if args and isinstance(args[0], Mapping):
                inputs.update(dict(args[0]))
            inputs.update({k: v for k, v in kwargs.items() if k != "logger"})
            with logger.enter(func_name or func.__name__, inputs, stage="core") as span:
                result = func(*args, **kwargs)
                if isinstance(result, Mapping):
                    span.set_output(dict(result))
                elif isinstance(result, (list, tuple)):
                    span.set_output({"count": len(result)})
                else:
                    span.set_output({"result": result})
                return result

        wrapper.__name__ = getattr(func, "__name__", "wrapped")
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


def shutdown_logging() -> None:
    """关闭全局 logger（进程退出/测试清理用）。"""
    global _STATE
    with _LOCK:
        logger = _STATE.get("logger")
        if isinstance(logger, StructuredLogger):
            logger.close()
        _STATE["logger"] = None
        _STATE["fallback"] = None
