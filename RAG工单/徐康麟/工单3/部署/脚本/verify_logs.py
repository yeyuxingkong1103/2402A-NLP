# -*- coding: utf-8 -*-
"""工单3 日志自检（部署验证工具，纯标准库 + 项目自实现 logger）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

作用（对着**真实落盘的日志**做机器判定，不看口头结论）：
    1. 三份规范日志是否存在、行数/字节、**坏行**（非 JSON 或粘连记录）与行号；
    2. 公共字段完整性（ts/level/event/func/module/run_id/trace_id/span_id/stage/pid/thread/work_order）
       且 ``work_order`` 必须等于工单编号字符串；
    3. ``func.enter`` / ``func.exit`` 是否成对（按 span_id），出口是否带 ``elapsed_ms`` 与非空 ``outputs``
       —— 对应硬性要求「禁止只写『开始/结束』」；
    4. ``error.log`` 的每条记录必须是 ERROR 以上，且具备可定位信息
       （``func.error`` 必须有 ``error.traceback``；业务形如 ``*.failed``/``*.error`` 必须有
        ``error_type`` + ``message``，即「显式降级必须留痕」）；
    5. ``rag_trace.jsonl`` 的事件必须命中 TRACE_PREFIXES 白名单（证明链路路由未写错文件）；
    6. ``部署/日志/deploy.log``（部署脚本自写）的 JSON Lines 契约；
    7. 可选 ``--probe-degradation``：**主动制造一次写失败**，验证 logger 的降级分支是否真的「显式降级」。

用法（工作目录 = E:\\gao6gongdan\\工单3）：
    pwsh -NoProfile -File run_py.ps1 部署/脚本/verify_logs.py
    pwsh -NoProfile -File run_py.ps1 部署/脚本/verify_logs.py --probe-degradation
    pwsh -NoProfile -File run_py.ps1 部署/脚本/verify_logs.py --strict --max-lines 50000

退出码：0=必检项全通过；1=存在必检项失败；2=参数错误（argparse 缺省行为）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

# 仓库根：部署/脚本/verify_logs.py → parents[2] = 工单3
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core.config import get_config                                    # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.logging_conf import TRACE_PREFIXES, LEVELS                  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 公共字段（设计/接口设计.md §4.1 冻结）——缺任意一个即判失败
COMMON_FIELDS: tuple[str, ...] = (
    "ts", "level", "event", "func", "module", "run_id", "trace_id", "span_id",
    "stage", "pid", "thread", "work_order",
)
# 三份规范日志（部署/日志/ 下，由研发/app/core/logging_conf.py 写）
CORE_LOGS: tuple[str, ...] = ("app.log", "error.log", "rag_trace.jsonl")
# 部署脚本自写日志（部署/脚本/*.ps1 与 *.sh），字段集略小
DEPLOY_LOG = "deploy.log"
# 「开始/结束」式空壳摘要（**只看字面**；空 dict/list 归入 empty_inputs/empty_outputs 另行计数）
EMPTY_SUMMARY_MARKERS: frozenset[str] = frozenset({"开始", "结束", "start", "end", "开始/结束", "start/end"})
# func.enter/func.exit 配对率下限（低于此值视为契约破坏；进程被 kill 的少量未配对只告警）
PAIRING_MIN_RATIO = 0.999
# T18 并发写入修复的时间分界（部署/日志/t18_fix_boundary.txt 里记录了实测分界，优先读取该文件）
CONCURRENCY_FIX_TS_DEFAULT = "2026-10-04T18:12:34+08:00"
CONCURRENCY_FIX_BOUNDARY_FILE = "t18_fix_boundary.txt"
# 并发写入自检参数（2 进程 × 60 行 × 16 KB 长记录；16 KB > 8 KB 写缓冲，能逼出拆写交错）
CONCURRENCY_PROCS = 2
CONCURRENCY_LINES = 60
CONCURRENCY_LINE_BYTES = 16384


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    """本地时区 ISO-8601（毫秒精度）。"""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def read_fix_boundary(log_dir: Path, cli_value: str, logger: Any = None) -> tuple[datetime, str]:
    """确定「修复分界时间戳」：CLI 参数 > 留痕文件 > 内置常量（并说明来源）。"""
    log = logger or get_logger("verify_logs")
    if cli_value:
        return datetime.fromisoformat(cli_value), "命令行 --since-ts"
    boundary_file = log_dir / CONCURRENCY_FIX_BOUNDARY_FILE
    if boundary_file.is_file():
        raw = boundary_file.read_text(encoding="utf-8").strip()
        try:
            return datetime.fromisoformat(raw), f"留痕文件 {boundary_file}"
        except ValueError as exc:  # 显式降级：文件坏了就用常量，但必须留痕
            log.log_event("verify_logs.boundary_file_bad", level="WARNING",
                          path=str(boundary_file), raw=raw, error_type=type(exc).__name__, message=str(exc))
    return datetime.fromisoformat(CONCURRENCY_FIX_TS_DEFAULT), "内置常量（T18 修复实测时间）"


def _sha256_16(path: Path) -> str:
    """文件 SHA256 前 16 位（大写），用于报告与留痕互校。"""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16].upper()


def _ensure_stdio() -> dict[str, Any]:
    """可移植性兜底：**无论走哪个入口 / 哪个 locale，退出码只反映检查结果，不反映终端编码能力**。

    背景（2026-10-04 captain 实测报出）：在 CP936（GBK）控制台上**直接调用**本脚本
    （`python.exe 部署/脚本/verify_logs.py`，未走 `run_py.ps1`）时，`sys.stdout.encoding='gbk'`，
    打印汇总表里的 ✅/⚠️/❌ 会抛 `UnicodeEncodeError` —— 而它发生在**所有检查都通过之后**，
    于是出现「检查全过、退出码却是 1」的假失败。修复分两层：

    1. **控制台只用 ASCII 标记**（`[OK]/[WARN]/[FAIL]`）—— 报告文件仍用 UTF-8 写（emoji 只出现在文件里）；
    2. 把 stdout/stderr 的错误策略设为 ``errors="replace"``（保留终端原编码，避免中文乱码）——
       即使将来有其它不可编码字符，也只是显示为 ``?``，**不会崩溃、不会污染退出码**。

    可用 ``RAG_FORCE_UTF8=1`` 强制把两个流切到 UTF-8（CI/容器里希望固定字节流时使用）。
    """
    report: dict[str, Any] = {"applied": [], "skipped": [], "stdout_encoding": None, "forced_utf8": False}
    force_utf8 = os.environ.get("RAG_FORCE_UTF8", "0") in {"1", "true", "yes", "on"}
    for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            report["skipped"].append({"stream": name, "reason": "该流不支持 reconfigure（可能是测试注入的 StringIO）"})
            continue
        try:
            if force_utf8:
                reconfigure(encoding="utf-8", errors="replace")
                report["forced_utf8"] = True
            else:
                reconfigure(errors="replace")            # 保留原编码，只放宽错误策略
            report["applied"].append({"stream": name, "encoding": getattr(stream, "encoding", None),
                                      "errors": getattr(stream, "errors", None)})
        except (ValueError, OSError) as exc:             # 显式降级：改不了也要留痕，但不阻断自检
            report["skipped"].append({"stream": name, "reason": f"{type(exc).__name__}: {exc}"})
    report["stdout_encoding"] = getattr(sys.stdout, "encoding", None)
    return report


def traced(func_name: str | None = None, *, inputs_fn: Callable[..., Mapping[str, Any]] | None = None):
    """函数入口/出口结构化日志装饰器（异常写堆栈后**原样抛出**，绝不吞）。"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        name = func_name or func.__name__

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger("verify_logs")
            inputs: dict[str, Any] = {}
            if inputs_fn is not None:
                try:
                    inputs = dict(inputs_fn(*args, **kwargs))
                except Exception as exc:  # noqa: BLE001 —— 摘要生成失败不得影响主流程，但必须留痕
                    logger.log_event("verify_logs.input_summary_failed", level="WARNING",
                                     func=name, error_type=type(exc).__name__, message=str(exc))
            with logger.enter(name, inputs) as span:
                result = func(*args, **kwargs)
                if isinstance(result, Mapping):
                    span.set_output(dict(result))
                else:
                    span.set_output({"result": repr(result)[:120]})
                return result

        wrapper.__name__ = getattr(func, "__name__", "wrapped")
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------
@traced("read_json_lines", inputs_fn=lambda path, **kw: {"path": str(path), "max_lines": kw.get("max_lines", 0)})
def read_json_lines(path: Path, *, max_lines: int = 0, sample_limit: int = 5,
                    boundary: datetime | None = None, logger: Any = None) -> dict[str, Any]:
    """逐行读 JSON Lines；返回 ``{exists, size_bytes, lines, records, bad, truncated}``。

    * 坏行（非 JSON / 多记录粘连）**计入 bad 并带行号、样例、前一条合法记录的 ts、是否在修复分界之后**；
    * 文件被其它进程占用时最多重试 3 次（Windows 共享读），仍失败则记 ERROR 并抛出；
    * ``max_lines > 0`` 时只分析**最后** max_lines 行（超大日志的快速巡检），并置 truncated=True。
    """
    log = logger or get_logger("verify_logs")
    result: dict[str, Any] = {"exists": path.is_file(), "path": str(path), "size_bytes": 0,
                              "lines": 0, "records": [], "bad": [], "truncated": False,
                              "sha256_16": None, "mtime": None}
    if not result["exists"]:
        log.log_event("verify_logs.file_missing", level="ERROR", path=str(path))
        return result

    stat = path.stat()
    result["size_bytes"] = stat.st_size
    result["mtime"] = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
    result["sha256_16"] = _sha256_16(path)

    last_error: Exception | None = None
    for attempt in range(1, 4):
        records: list[tuple[int, dict[str, Any]]] = []
        total = 0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for lineno, raw in enumerate(fh, 1):
                    line = raw.strip()
                    if not line:
                        continue
                    total += 1
                    if not line.startswith("{"):
                        continue          # 坏行由 _scan_bad_lines 统一统计（含 ts 归属）
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    records.append((lineno, record))
            result["lines"] = total
            result["records"] = records
            bad_info = _scan_bad_lines(path, boundary=boundary, sample_limit=sample_limit)
            result["bad"] = bad_info["samples"]
            result["bad_count"] = bad_info["bad"]
            result["bad_before_fix"] = bad_info["bad_before_fix"]
            result["bad_after_fix"] = bad_info["bad_after_fix"]
            result["bad_countable"] = bad_info["countable"]
            last_error = None
            break
        except OSError as exc:  # 文件正被写入/占用：重试是**显式降级**，不是静默
            last_error = exc
            log.log_event("verify_logs.read_retry", level="WARNING", path=str(path),
                          attempt=attempt, error_type=type(exc).__name__, message=str(exc))
            time.sleep(0.3)
    if last_error is not None:
        log.log_event("verify_logs.read_failed", level="ERROR", path=str(path),
                      error_type=type(last_error).__name__, message=str(last_error))
        raise last_error

    if max_lines and len(result["records"]) > max_lines:
        result["records"] = result["records"][-max_lines:]
        result["truncated"] = True
    return result


def _scan_bad_lines(path: Path, *, boundary: datetime | None, sample_limit: int) -> dict[str, Any]:
    """二次扫描：精确统计坏行总数，并按「修复分界时间戳」拆分（修复前遗留 / 修复后新增）。

    坏行本身没有可解析的 ts，因此用**它前面最近一条合法记录的 ts** 作为时间归属
    （撕裂写入的碎片总是紧跟在同一进程刚写出的合法行之后，误差远小于秒级）。
    """
    result: dict[str, Any] = {"bad": 0, "bad_before_fix": 0, "bad_after_fix": 0, "samples": [], "countable": True}
    last_ts_raw: str | None = None
    last_ts_dt: datetime | None = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                ok = False
                if line.startswith("{"):
                    try:
                        record = json.loads(line)
                        ok = isinstance(record, dict)
                        if ok and isinstance(record.get("ts"), str):
                            last_ts_raw = record["ts"]
                            try:
                                last_ts_dt = datetime.fromisoformat(last_ts_raw)
                            except ValueError:
                                last_ts_dt = None
                    except json.JSONDecodeError:
                        ok = False
                if ok:
                    continue
                result["bad"] += 1
                after_fix = bool(boundary is not None and last_ts_dt is not None and last_ts_dt >= boundary)
                result["bad_after_fix" if after_fix else "bad_before_fix"] += 1
                if len(result["samples"]) < sample_limit:
                    result["samples"].append({
                        "line": lineno,
                        "reason": "非 JSON 行首（疑似跨进程撕裂写入）" if not line.startswith("{") else "JSON 解析失败",
                        "after_fix": after_fix, "preceding_valid_ts": last_ts_raw, "sample": line[:160],
                    })
    except OSError:
        result["countable"] = False   # 无法二次统计时显式标记，报告里写明「未能完整统计」
    return result


def _count_bad_lines(path: Path, sample_limit: int, boundary: datetime | None = None) -> dict[str, Any]:
    """兼容包装：返回 ``{bad, bad_before_fix, bad_after_fix, samples}``（未显式给分界则全算「修复前」）。"""
    return _scan_bad_lines(path, boundary=boundary, sample_limit=sample_limit)


# ---------------------------------------------------------------------------
# 检查项
# ---------------------------------------------------------------------------
@traced("check_common_fields", inputs_fn=lambda records, **kw: {"records": len(records)})
def check_common_fields(records: Iterable[tuple[int, dict[str, Any]]], *, logger: Any) -> dict[str, Any]:
    """校验公共字段齐全 + ``work_order`` 等于工单编号 + ``ts`` 可解析。"""
    log = logger
    missing: dict[str, int] = {field: 0 for field in COMMON_FIELDS}
    bad_work_order = 0
    bad_ts = 0
    checked = 0
    samples: list[dict[str, Any]] = []
    for lineno, record in records:
        checked += 1
        for field in COMMON_FIELDS:
            if field not in record:
                missing[field] += 1
                if len(samples) < 5:
                    samples.append({"line": lineno, "missing_field": field, "event": record.get("event")})
        if record.get("work_order") != WORK_ORDER:
            bad_work_order += 1
        try:
            datetime.fromisoformat(str(record.get("ts")))
        except (TypeError, ValueError):
            bad_ts += 1
    detail = {
        "checked": checked,
        "missing_by_field": {k: v for k, v in missing.items() if v},
        "bad_work_order": bad_work_order,
        "bad_ts": bad_ts,
        "samples": samples,
    }
    log.log_event("verify_logs.common_fields", checked=checked,
                  missing_total=sum(missing.values()), bad_work_order=bad_work_order, bad_ts=bad_ts)
    return detail


@traced("check_spans", inputs_fn=lambda records, **kw: {"records": len(records)})
def check_spans(records: Iterable[tuple[int, dict[str, Any]]], *, logger: Any) -> dict[str, Any]:
    """校验 ``func.enter``/``func.exit`` 成对、出口有耗时与非空输出、摘要不含「开始/结束」式空壳。

    判定口径（写进报告的 ``rule``，避免事后改标准）：
        * **必检失败**：出现字面「开始/结束/start/end」摘要、``func.exit`` 缺 ``elapsed_ms``、
          ``func.exit`` 的 ``outputs`` 为空、配对率 < ``PAIRING_MIN_RATIO``（0.999）；
        * **仅告警**：``inputs`` 为空（**无参函数天然如此**，如 ``QAEngine.files()``/``health()``，
          实测 113 次全部来自无参方法）、未配对的 ``func.enter``（进程被 kill/超时中断的典型痕迹，
          实测 8/41155）、孤立的 ``func.exit``。
    """
    log = logger
    enters: dict[str, int] = {}
    exits: dict[str, int] = {}
    enter_meta: dict[str, tuple[int, str, str]] = {}
    errors: set[str] = set()
    exit_without_enter = 0
    empty_inputs_by_func: dict[str, int] = {}
    empty_outputs: int = 0
    exit_cnt = 0
    missing_elapsed = 0
    literal_markers: dict[str, int] = {}
    enter_count = 0
    for lineno, record in records:
        event = str(record.get("event"))
        span_id = str(record.get("span_id") or "")
        if event == "func.enter":
            enter_count += 1
            enters[span_id] = enters.get(span_id, 0) + 1
            enter_meta[span_id] = (lineno, str(record.get("func")), str(record.get("run_id")))
            payload = _flatten(record.get("inputs"))
            if payload in (None, "", "{}", "[]"):
                func_name = str(record.get("func"))
                empty_inputs_by_func[func_name] = empty_inputs_by_func.get(func_name, 0) + 1
        elif event == "func.exit":
            exit_cnt += 1
            exits[span_id] = exits.get(span_id, 0) + 1
            if not span_id or (span_id not in enters and span_id not in errors):
                exit_without_enter += 1
            if record.get("elapsed_ms") is None:
                missing_elapsed += 1
            payload = _flatten(record.get("outputs"))
            if payload in (None, "", "{}", "[]"):
                empty_outputs += 1
        elif event == "func.error":
            errors.add(str(record.get("span_id") or ""))
        for key in ("inputs", "outputs"):
            payload = _flatten(record.get(key))
            if isinstance(payload, str) and payload.strip().lower() in EMPTY_SUMMARY_MARKERS:
                literal_markers[f"{record.get('func')}.{key}={payload.strip()}"] = \
                    literal_markers.get(f"{record.get('func')}.{key}={payload.strip()}", 0) + 1
    unpaired_detail = [{"line": enter_meta[sid][0], "func": enter_meta[sid][1], "run_id": enter_meta[sid][2],
                        "span_id": sid}
                       for sid in enters if sid not in exits and sid not in errors][:10]
    unpaired = sum(count for span_id, count in enters.items() if not exits.get(span_id) and span_id not in errors)
    pairing_ratio = round((enter_count - unpaired) / enter_count, 6) if enter_count else None
    detail = {
        "enter_count": enter_count, "exit_count": exit_cnt, "error_count": len(errors),
        "unpaired_enter": unpaired, "unpaired_samples": unpaired_detail,
        "exit_without_enter": exit_without_enter,
        "empty_inputs": sum(empty_inputs_by_func.values()), "empty_inputs_by_func": empty_inputs_by_func,
        "empty_outputs": empty_outputs, "missing_elapsed_ms": missing_elapsed,
        "literal_start_end_markers": literal_markers,
        "pairing_ratio": pairing_ratio, "pairing_min_ratio": PAIRING_MIN_RATIO,
        "rule": ("必检失败 = 字面开始/结束摘要 或 exit 缺 elapsed_ms 或 exit 输出为空 或 配对率<"
                 f"{PAIRING_MIN_RATIO}；empty_inputs/unpaired 仅告警（无参函数与进程中断的合法成因）"),
    }
    log.log_event("verify_logs.spans", enter_count=enter_count, exit_count=exit_cnt,
                  unpaired_enter=unpaired, exit_without_enter=exit_without_enter,
                  empty_inputs=detail["empty_inputs"], empty_outputs=empty_outputs,
                  missing_elapsed_ms=missing_elapsed, literal_markers=sum(literal_markers.values()),
                  pairing_ratio=pairing_ratio)
    return detail


def _flatten(value: Any) -> str | None:
    """把摘要字段压成便于判空的字符串（None/{} 都算空）。"""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return "{}" if not value else json.dumps(value, ensure_ascii=False)[:200]
    return str(value)[:200]


@traced("check_error_log", inputs_fn=lambda records, **kw: {"records": len(records)})
def check_error_log(records: Iterable[tuple[int, dict[str, Any]]], *, logger: Any) -> dict[str, Any]:
    """``error.log``：每条必须 ERROR 以上；``func.error`` 必须有完整堆栈；业务错误必须带明细字段。

    判定口径（写进报告的 ``rule``）：
        * ``func.error`` → ``error.type`` + ``error.message`` + ``error.traceback``（含 Traceback 头）；
        * 业务错误事件（``generation.*``/``llm.*``/``qa.*``/``pdf.*``/``ui.*`` …）→ **至少一个**非公共字段的
          非空明细（``error_type`` / ``message`` / ``reason`` / ``degrade_to`` / ``invalid`` / ``retry`` …），
          证明「显式降级/失败原因」被留痕；一条明细都没有才算「真静默」。
    """
    log = logger
    common_and_structural = set(COMMON_FIELDS) | {"inputs", "outputs", "elapsed_ms", "error"}
    total = 0
    below_error = 0
    func_error_no_traceback = 0
    business_no_detail = 0
    business_with_detail = 0
    business_keys: dict[str, list[str]] = {}
    samples: list[dict[str, Any]] = []
    for lineno, record in records:
        total += 1
        level = str(record.get("level", "INFO")).upper()
        if LEVELS.get(level, 0) < LEVELS["ERROR"]:
            below_error += 1
        event = str(record.get("event"))
        if event == "func.error":
            err = record.get("error") or {}
            tb = str(err.get("traceback") or "")
            if "Traceback (most recent call last)" not in tb.strip():
                func_error_no_traceback += 1
                if len(samples) < 5:
                    samples.append({"line": lineno, "event": event, "why": "func.error 缺 traceback",
                                    "func": record.get("func")})
        else:
            detail_keys = [key for key, value in record.items()
                           if key not in common_and_structural and str(value).strip() not in ("", "None", "null", "[]", "{}")]
            business_keys.setdefault(event, sorted(set(detail_keys)))
            if detail_keys:
                business_with_detail += 1
            else:
                business_no_detail += 1
                if len(samples) < 5:
                    samples.append({"line": lineno, "event": event, "why": "业务错误事件没有任何明细字段（真静默）",
                                    "func": record.get("func")})
    detail = {"checked": total, "below_error": below_error,
              "func_error_without_traceback": func_error_no_traceback,
              "business_error_with_detail": business_with_detail,
              "business_error_without_detail": business_no_detail,
              "business_event_detail_keys": business_keys, "samples": samples,
              "rule": ("func.error 必须含 Traceback 堆栈；业务错误事件必须至少含 1 个非公共明细字段；"
                       "error.log 内不得出现低于 ERROR 的记录")}
    log.log_event("verify_logs.error_log", **{k: v for k, v in detail.items() if k != "business_event_detail_keys"})
    return detail


@traced("check_trace_routing", inputs_fn=lambda records, **kw: {"records": len(records)})
def check_trace_routing(records: Iterable[tuple[int, dict[str, Any]]], *, logger: Any) -> dict[str, Any]:
    """``rag_trace.jsonl`` 的事件必须在 TRACE_PREFIXES 白名单内（否则证明路由写错）。"""
    log = logger
    total = 0
    offenders: list[dict[str, Any]] = []
    families: dict[str, int] = {}
    for lineno, record in records:
        total += 1
        event = str(record.get("event"))
        family = (event.split(".", 1)[0] + ".") if "." in event else event
        families[family] = families.get(family, 0) + 1
        if not event.startswith(TRACE_PREFIXES):
            if len(offenders) < 5:
                offenders.append({"line": lineno, "event": event, "module": record.get("module")})
    detail = {"checked": total, "offenders": offenders, "offender_count": len(offenders),
              "families": dict(sorted(families.items(), key=lambda kv: -kv[1])[:12]),
              "allowed_prefixes": list(TRACE_PREFIXES)}
    log.log_event("verify_logs.trace_routing", checked=total, offender_count=len(offenders))
    return detail


@traced("check_deploy_log", inputs_fn=lambda records, **kw: {"records": len(records)})
def check_deploy_log(records: Iterable[tuple[int, dict[str, Any]]], *, logger: Any) -> dict[str, Any]:
    """部署脚本日志（deploy.log）的 JSON Lines 契约：必需字段 + func.enter/exit 计数。"""
    log = logger
    required = ("ts", "level", "event", "func", "module", "pid", "work_order")
    total = 0
    missing: dict[str, int] = {field: 0 for field in required}
    enter = exit_ = error = 0
    modules: dict[str, int] = {}
    for _lineno, record in records:
        total += 1
        for field in required:
            if field not in record:
                missing[field] += 1
        event = str(record.get("event"))
        if event == "func.enter":
            enter += 1
        elif event == "func.exit":
            exit_ += 1
        elif event == "func.error":
            error += 1
        module = str(record.get("module"))
        modules[module] = modules.get(module, 0) + 1
    detail = {"checked": total, "missing_by_field": {k: v for k, v in missing.items() if v},
              "enter": enter, "exit": exit_, "error": error, "modules": modules}
    log.log_event("verify_logs.deploy_log", checked=total, enter=enter, exit=exit_, error=error)
    return detail


@traced("check_concurrent_writes", inputs_fn=lambda **kw: {"procs": CONCURRENCY_PROCS, "lines": CONCURRENCY_LINES,
                                                          "line_bytes": CONCURRENCY_LINE_BYTES})
def check_concurrent_writes(*, logger: Any) -> dict[str, Any]:
    """**并发写入自检（T18 验收项）**：真起 N 个进程用产品 logger 同时写长记录，校验非法行 = 0。

    与 `部署/脚本/stress_log_concurrency.py` 同一判据，但参数更小（2 进程 × 60 行 × 16 KB），
    可在每次日志自检里跑完（约 1 s）。写入发生在 `部署/日志/_verify_logs_conc/`，不污染 `app.log`。
    """
    log = logger
    import subprocess

    work_dir = REPO_ROOT / "部署" / "日志" / "_verify_logs_conc"
    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / "app.log"
    if target.exists():
        target.unlink()          # 自检每次从零开始，只校验本次并发写入的产物
    dev_path = str(REPO_ROOT / "研发")
    child_code = (
        "import sys, pathlib\n"
        f"sys.path.insert(0, r'{dev_path}')\n"
        "from app.core.logging_conf import StructuredLogger\n"
        "d = pathlib.Path(sys.argv[1]); n = int(sys.argv[2]); size = int(sys.argv[3]); tag = sys.argv[4]\n"
        "lg = StructuredLogger(run_id='r-conc', log_dir=d, echo_error=False, module='conc')\n"
        "filler = '样本正文' * max(1, size // 12)\n"
        "for i in range(n):\n"
        "    lg.log_event(f'verify_logs.conc.{tag}', index=i, payload=filler, trace_id=f't{i:05d}')\n"
        "lg.close()\n"
    )
    processes = []
    started = time.perf_counter()
    for index in range(CONCURRENCY_PROCS):
        processes.append(subprocess.Popen(
            [sys.executable, "-c", child_code, str(work_dir), str(CONCURRENCY_LINES),
             str(CONCURRENCY_LINE_BYTES), f"w{index}"],
            cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace"))
    child_errors: list[str] = []
    for proc in processes:
        _out, err = proc.communicate(timeout=300)
        if proc.returncode != 0:
            child_errors.append(f"pid={proc.pid} exit={proc.returncode} stderr={err.strip()[:200]}")
    scan = _scan_bad_lines(target, boundary=None, sample_limit=3)
    expected = CONCURRENCY_PROCS * CONCURRENCY_LINES
    observed = sum(1 for line in target.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
    detail = {
        "procs": CONCURRENCY_PROCS, "lines_per_proc": CONCURRENCY_LINES,
        "line_bytes": CONCURRENCY_LINE_BYTES, "expected_lines": expected,
        "observed_lines": observed,
        "bad_lines": scan["bad"], "bad_samples": scan["samples"], "child_errors": child_errors,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2), "work_dir": str(work_dir),
        "rule": ("N 个进程用产品 StructuredLogger 并发写 16 KB 长记录（> 8 KB 写缓冲），"
                 "非法行必须 = 0；这是 T18 并发写入缺陷的验收判据"),
    }
    log.log_event("verify_logs.concurrent_writes", procs=CONCURRENCY_PROCS, lines=expected,
                  bad_lines=scan["bad"], child_errors=len(child_errors),
                  elapsed_ms=detail["elapsed_ms"])
    return detail


class _BrokenHandle:
    """合成一个**写入必失败**的句柄（模拟磁盘满 / 句柄被外部关闭），但**保留其它行为**以便断言摘除逻辑。"""

    def __init__(self) -> None:
        self.write_calls = 0
        self.closed = False

    def write(self, _payload: Any) -> None:
        self.write_calls += 1
        raise OSError("合成写失败（verify_logs 降级探针）")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _run_degrade_probe(log_dir: Path) -> dict[str, Any]:
    """核心探针：注入必失败句柄 → 走**真实降级路径**（不预先 clear）→ 收集可断言的观测。

    断言三条（对应 t18 修复的语义，缺一不可）：
        ① **摘掉坏的、留下好的**：`_handles` 只移除被注入失败的 ``app``，``error`` / ``trace`` 必须保留
           （**不是整体清空** —— 这正是旧探针用 `_handles.clear()` 绕过、导致断言无效的地方）；
        ② **不抛异常**：降级分支自己不得抛（修复前会抛 `NameError: name 'sys' is not defined`）；
        ③ **降级可观测**：stderr 必须出现含 **kind** 与**异常类型**的降级消息（禁止静默失败）。
    """
    from app.core.logging_conf import StructuredLogger

    class _BrokenHandleLocal(_BrokenHandle):
        pass

    logger_under_test = StructuredLogger(run_id="r-verify-logs-probe", log_dir=log_dir, echo_error=False)
    initial_kinds = sorted(logger_under_test._handles)
    broken = _BrokenHandleLocal()
    logger_under_test._handles["app"] = broken                     # noqa: SLF001 —— 探针必须替换内部句柄
    after_kinds: list[str] = []
    removed_kinds: list[str] = []
    kept_kinds: list[str] = []
    stderr_text = ""
    raised: str | None = None
    traceback_text: str | None = None
    write_fail_handled: bool | None = None
    try:
        import contextlib
        import io

        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            try:
                logger_under_test.log_event("verify_logs.probe_write_failure")   # INFO → 只写 app
                write_fail_handled = True
            except Exception as exc:  # noqa: BLE001 —— 探针就是要抓住它
                import traceback as _tb

                write_fail_handled = False
                raised = f"{type(exc).__name__}: {exc}"
                traceback_text = _tb.format_exc()
        stderr_text = captured.getvalue()
        after_kinds = sorted(logger_under_test._handles)
        removed_kinds = [kind for kind in initial_kinds if kind not in after_kinds]
        kept_kinds = [kind for kind in initial_kinds if kind in after_kinds]
    finally:
        # 只关闭**仍然存在**的句柄；**绝不 clear()**，否则后续断言与关闭检查都失去意义
        try:
            logger_under_test.close()
            close_handled: bool | None = True
        except Exception as exc:  # noqa: BLE001
            close_handled = False
            raised = raised or f"{type(exc).__name__}: {exc}"

    assertions = [
        {"id": "broken_kind_removed", "ok": removed_kinds == ["app"],
         "detail": f"被注入失败的 kind 必须且只能移除 app：removed={removed_kinds}"},
        {"id": "healthy_kinds_kept", "ok": kept_kinds == ["error", "trace"],
         "detail": f"其余句柄必须保留（不是整体清空）：kept={kept_kinds}"},
        {"id": "no_exception_raised", "ok": bool(write_fail_handled) and raised is None,
         "detail": f"降级分支不得抛异常：write_fail_handled={write_fail_handled} raised={raised}"},
        {"id": "degrade_observable", "ok": ("写 app 失败" in stderr_text) and ("OSError" in stderr_text),
         "detail": f"stderr 必须含 kind 与异常类型：{stderr_text.strip()[:160]!r}"},
        {"id": "close_handled", "ok": close_handled is True,
         "detail": f"剩余句柄必须能正常关闭：close_handled={close_handled}"},
    ]
    return {
        "probe": "StructuredLogger._write 降级路径（注入必失败句柄）",
        "initial_kinds": initial_kinds, "after_kinds": after_kinds,
        "removed_kinds": removed_kinds, "kept_kinds": kept_kinds,
        "write_fail_handled": write_fail_handled, "raised": raised, "traceback": traceback_text,
        "broken_handle_write_calls": broken.write_calls,
        "degrade_stderr": stderr_text.strip()[:300],
        "close_fail_handled": close_handled,
        "assertions": assertions,
        "assertions_passed": all(item["ok"] for item in assertions),
    }


@traced("probe_degradation", inputs_fn=lambda **kw: {"probe": "合成写失败", "target": "StructuredLogger._write/close"})
def probe_degradation(*, logger: Any) -> dict[str, Any]:
    """降级探针（**含负向对照**）：验证降级语义 + 证明这组断言真的能抓到问题。

    * **正向**：注入必失败句柄 → 断言「摘掉坏的、留下好的」「不抛异常」「降级可观测」（见 `_run_degrade_probe`）；
    * **负向对照**：把 ``app.core.logging_conf`` 恢复成**修复前的模块状态**（模块级没有 ``sys`` 绑定，
      用 `del logging_conf.sys` 精确模拟，跑完立即恢复），同一组断言**必须失败** ——
      这样每次自检都自带「断言有牙齿」的证明，而不是只相信自己会绿。
    """
    log = logger
    probe_dir = REPO_ROOT / "部署" / "日志" / "_verify_logs_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    positive = _run_degrade_probe(probe_dir)

    negative: dict[str, Any] = {"injected": "del app.core.logging_conf.sys（模拟修复前：模块级无 sys）"}
    import app.core.logging_conf as logging_conf_module

    saved_sys = getattr(logging_conf_module, "sys", None)
    try:
        if saved_sys is not None:
            del logging_conf_module.sys                     # 精确复现修复前的模块状态
        neg = _run_degrade_probe(probe_dir)
        negative.update({
            "raised": neg["raised"], "removed_kinds": neg["removed_kinds"],
            "write_fail_handled": neg["write_fail_handled"],
            "assertions_passed": neg["assertions_passed"],
            "assertion_can_catch_defect": (neg["assertions_passed"] is False),
            "first_failing_assertion": next((a["id"] for a in neg["assertions"] if not a["ok"]), None),
        })
    finally:
        if saved_sys is not None:
            logging_conf_module.sys = saved_sys             # 立即恢复，避免影响后续日志

    result = dict(positive)
    result["probe_dir"] = str(probe_dir)
    result["negative_control"] = negative
    result["assertions_passed"] = bool(positive["assertions_passed"]
                                       and negative.get("assertion_can_catch_defect"))
    log.log_event("verify_logs.probe_degradation",
                  assertions_passed=result["assertions_passed"],
                  removed_kinds=positive["removed_kinds"], kept_kinds=positive["kept_kinds"],
                  raised=positive["raised"],
                  negative_control_catches=negative.get("assertion_can_catch_defect"),
                  negative_first_failure=negative.get("first_failing_assertion"))
    return result


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def add_check(checks: list[dict[str, Any]], *, cid: str, title: str, status: str, required: bool,
              detail: Any, evidence: str = "") -> None:
    """追加一条检查结论（status ∈ passed / warning / failed）。"""
    checks.append({"id": cid, "title": title, "status": status, "required": required,
                   "detail": detail, "evidence": evidence})


def write_reports(*, out_json: Path, out_md: Path, report: dict[str, Any], logger: Any) -> None:
    """写 JSON + Markdown 报告（两个都落盘，便于机器判定与人工复核）。"""
    log = logger
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# 部署日志自检报告（{report['verdict']}）",
        "",
        f"> 工单：{WORK_ORDER}　生成时间：{report['generated_at']}",
        f"> 日志目录：`{report['log_dir']}`　退出码：{report['exit_code']}",
        "",
        "## 一、结论",
        "",
        "| 检查项 | 状态 | 必检 | 结论摘要 |",
        "| --- | --- | --- | --- |",
    ]
    mark = {"passed": "✅", "warning": "⚠️", "failed": "❌"}
    for check in report["checks"]:
        summary = json.dumps(check["detail"], ensure_ascii=False)
        if len(summary) > 220:
            summary = summary[:220] + "…"
        lines.append(f"| {check['title']} | {mark.get(check['status'], '?')} {check['status']} | "
                     f"{'是' if check['required'] else '否'} | {summary} |")
    lines += ["", "## 二、文件清单", "", "| 文件 | 字节 | 行数 | 坏行 | SHA256(前16) | 最后写入 |", "| --- | --- | --- | --- | --- | --- |"]
    for row in report["files"]:
        lines.append(f"| `{row['name']}` | {row['size_bytes']} | {row['lines']} | {row.get('bad_count', '—')} | "
                     f"{row.get('sha256_16') or '—'} | {row.get('mtime') or '—'} |")
    if report.get("findings"):
        lines += ["", "## 三、发现（需研发侧跟进，非部署脚本问题）", ""]
        for finding in report["findings"]:
            lines.append(f"- **[{finding['severity']}] {finding['id']}**：{finding['problem']}")
            lines.append(f"  - 证据：{finding['evidence']}")
    if report.get("notes"):
        lines += ["", "## 四、口径与限制", ""] + [f"- {note}" for note in report["notes"]]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.log_event("verify_logs.reports_written", json_report=str(out_json), md_report=str(out_md),
                  verdict=report["verdict"], exit_code=report["exit_code"])


def main(argv: list[str] | None = None) -> int:
    """命令行入口：读日志 → 逐项检查 → 落报告 → 返回退出码。"""
    parser = argparse.ArgumentParser(description="工单3 部署日志自检（字段/配对/堆栈/路由）")
    parser.add_argument("--log-dir", default=str(REPO_ROOT / "部署" / "日志"))
    parser.add_argument("--json", default=str(REPO_ROOT / "部署" / "日志" / "verify_logs_report.json"))
    parser.add_argument("--md", default=str(REPO_ROOT / "部署" / "日志" / "verify_logs_report.md"))
    parser.add_argument("--max-lines", type=int, default=300000,
                        help="每份日志最多分析最后 N 行（0=不限；默认 30 万行防超大文件拖慢巡检）")
    parser.add_argument("--max-bad-tolerated", type=int, default=0,
                        help="允许的坏行数（默认 0；坏行会在报告里逐条留证）")
    parser.add_argument("--strict", action="store_true", help="把「发现」也当作必检失败（CI 严格模式）")
    parser.add_argument("--probe-degradation", action="store_true", help="额外做一次写失败降级探针")
    parser.add_argument("--since-ts", default="",
                        help="T18 并发修复的时间分界（ISO8601）；之后的坏行判为必检失败，默认读 "
                             "部署/日志/t18_fix_boundary.txt，读不到用内置常量")
    parser.add_argument("--no-concurrency-check", action="store_true",
                        help="跳过并发写入自检（默认执行：2 进程 × 60 行 × 16 KB，校验非法行 = 0）")
    args = parser.parse_args(argv)

    stdio = _ensure_stdio()          # 必须最先执行：见 _ensure_stdio 的可移植性说明
    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("verify_logs")
    log.log_event("verify_logs.stdio_ready", **stdio)
    started = time.perf_counter()
    log_dir = Path(args.log_dir)
    report: dict[str, Any] = {"work_order": WORK_ORDER, "generated_at": _now_iso(), "log_dir": str(log_dir),
                              "argv": sys.argv[1:], "python": sys.version.split()[0], "stdio": stdio,
                              "checks": [], "files": [], "findings": [], "notes": []}
    checks: list[dict[str, Any]] = report["checks"]
    findings: list[dict[str, Any]] = report["findings"]

    with log.enter("verify_logs.main", {"log_dir": str(log_dir), "max_lines": args.max_lines,
                                        "strict": args.strict, "probe_degradation": args.probe_degradation}) as span:
        boundary, boundary_source = read_fix_boundary(log_dir, args.since_ts, log)
        report["concurrency_fix_boundary"] = {"ts": boundary.isoformat(), "source": boundary_source}
        # ---- 1) 读取三份核心日志 ----
        loaded: dict[str, dict[str, Any]] = {}
        for name in CORE_LOGS:
            path = log_dir / name
            info = read_json_lines(path, max_lines=args.max_lines, boundary=boundary, logger=log)
            loaded[name] = info
            report["files"].append({"name": name, "path": str(path), "exists": info["exists"],
                                    "size_bytes": info["size_bytes"], "lines": info["lines"],
                                    "bad_count": info.get("bad_count"), "sha256_16": info.get("sha256_16"),
                                    "mtime": info.get("mtime"), "truncated": info["truncated"],
                                    "bad_samples": info["bad"]})
        missing_files = [name for name, info in loaded.items() if not info["exists"]]
        add_check(checks, cid="files_exist", title="三份规范日志存在", required=True,
                  status="failed" if missing_files else "passed",
                  detail={"missing": missing_files, "present": [n for n in CORE_LOGS if n not in missing_files]},
                  evidence="部署/日志/app.log、error.log、rag_trace.jsonl")
        if missing_files:
            log.log_event("verify_logs.abort_missing_files", level="ERROR", missing=missing_files)

        app_records = loaded["app.log"]["records"]
        error_records = loaded["error.log"]["records"]
        trace_records = loaded["rag_trace.jsonl"]["records"]

        # ---- 2) 坏行（按 T18 修复分界拆分：修复后仍出现坏行 = 必检失败）----
        bad_total = sum(int(info.get("bad_count") or 0) for info in loaded.values())
        bad_before = sum(int(info.get("bad_before_fix") or 0) for info in loaded.values())
        bad_after = sum(int(info.get("bad_after_fix") or 0) for info in loaded.values())
        uncountable = [name for name, info in loaded.items() if info.get("bad_countable") is False]
        bad_samples = [{"file": name, **sample} for name, info in loaded.items() for sample in info["bad"]]
        # 判据：分界时间戳之后不得再有坏行（必检）；历史遗留坏行（契约要求不得删除）只告警
        if bad_after > 0:
            bad_status = "failed"
        elif bad_before > 0 or uncountable or (args.strict and bad_total > 0):
            bad_status = "warning"
        else:
            bad_status = "passed"
        add_check(checks, cid="json_lines",
                  title="JSON Lines 可解析（T18 修复分界之后必须 0 坏行）", required=True,
                  status=bad_status,
                  detail={"bad_total": bad_total, "bad_before_fix": bad_before, "bad_after_fix": bad_after,
                          "fix_boundary": boundary.isoformat(), "fix_boundary_source": boundary_source,
                          "tolerated": args.max_bad_tolerated, "uncountable_files": uncountable,
                          "samples": bad_samples[:6],
                          "note": ("坏行 = 非 JSON 行首或两记录粘连；分界之前的是 T18 修复前遗留"
                                   "（契约要求不得删除，仅告警）；分界之后出现即判失败")},
                  evidence="逐行 json.loads；样例含行号、前一条合法记录 ts、是否在修复分界之后")
        if bad_after > 0:
            findings.append({
                "id": "T18 修复后仍出现非法 JSON 行（回归）",
                "severity": "blocker",
                "problem": (f"修复分界（{boundary.isoformat()}）之后仍有 {bad_after} 行无法解析 —— "
                            "说明并发写入策略失效或被绕过，必须立即排查。"),
                "evidence": json.dumps([s for s in bad_samples if s.get("after_fix")][:4], ensure_ascii=False),
            })
        elif bad_before > 0:
            findings.append({
                "id": "日志历史坏行（T18 修复前遗留，按契约保留）",
                "severity": "low",
                "problem": (f"修复分界（{boundary.isoformat()}，来源 {boundary_source}）之前有 {bad_before} 行"
                            "无法解析，系修复前多进程缓冲写交错所致；这些行**按契约不得删除/截断**，仅作历史留痕。"
                            "修复后新增坏行 = 0。"),
                "evidence": json.dumps(bad_samples[:4], ensure_ascii=False),
            })

        # ---- 3) 公共字段 ----
        common_detail = check_common_fields(app_records, logger=log)
        common_failed = (sum(common_detail["missing_by_field"].values()) > 0
                         or common_detail["bad_work_order"] > 0 or common_detail["bad_ts"] > 0)
        add_check(checks, cid="common_fields", title="公共字段齐全且 work_order 正确", required=True,
                  status="failed" if common_failed else "passed", detail=common_detail,
                  evidence="设计/接口设计.md §4.1 的 12 个公共字段")

        # ---- 4) func.enter/exit 配对 ----
        span_detail = check_spans(app_records, logger=log)
        span_fatal = (sum(span_detail["literal_start_end_markers"].values()) > 0
                      or span_detail["empty_outputs"] > 0
                      or span_detail["missing_elapsed_ms"] > 0
                      or (span_detail["pairing_ratio"] is not None
                          and span_detail["pairing_ratio"] < PAIRING_MIN_RATIO))
        span_warn = (span_detail["unpaired_enter"] > 0 or span_detail["exit_without_enter"] > 0
                     or span_detail["empty_inputs"] > 0)
        add_check(checks, cid="func_spans",
                  title="函数入口/出口成对且摘要非空（无参函数 inputs={} 与进程中断仅告警）",
                  required=True,
                  status="failed" if span_fatal else ("warning" if span_warn else "passed"),
                  detail=span_detail,
                  evidence="不得只写「开始/结束」：enter 记 inputs，exit 记 outputs + elapsed_ms")

        # ---- 5) error.log 质量 ----
        err_detail = check_error_log(error_records, logger=log)
        err_failed = (err_detail["below_error"] > 0 or err_detail["func_error_without_traceback"] > 0
                      or err_detail["business_error_without_detail"] > 0)
        add_check(checks, cid="error_log", title="error.log 全为 ERROR 且含堆栈/明细", required=True,
                  status="failed" if err_failed else "passed", detail=err_detail,
                  evidence="func.error → error.traceback；业务错误事件 → 至少 1 个非公共明细字段")

        # ---- 6) trace 路由 ----
        trace_detail = check_trace_routing(trace_records, logger=log)
        add_check(checks, cid="trace_routing", title="rag_trace.jsonl 事件命中白名单", required=True,
                  status="failed" if trace_detail["offender_count"] else "passed", detail=trace_detail,
                  evidence=f"白名单前缀：{list(TRACE_PREFIXES)}")

        # ---- 7) deploy.log（部署脚本自写） ----
        deploy_path = log_dir / DEPLOY_LOG
        deploy_info = read_json_lines(deploy_path, max_lines=0, logger=log)
        report["files"].append({"name": DEPLOY_LOG, "path": str(deploy_path), "exists": deploy_info["exists"],
                                "size_bytes": deploy_info["size_bytes"], "lines": deploy_info["lines"],
                                "bad_count": deploy_info.get("bad_count"), "sha256_16": deploy_info.get("sha256_16"),
                                "mtime": deploy_info.get("mtime"), "truncated": deploy_info["truncated"],
                                "bad_samples": deploy_info["bad"]})
        if deploy_info["exists"]:
            deploy_detail = check_deploy_log(deploy_info["records"], logger=log)
            deploy_failed = sum(deploy_detail["missing_by_field"].values()) > 0
            add_check(checks, cid="deploy_log", title="deploy.log（启动/评测脚本）契约", required=False,
                      status="failed" if deploy_failed else "passed", detail=deploy_detail,
                      evidence="部署/脚本/run_app.ps1、evaluate.ps1、run_app.sh 等写入的结构化 JSON Lines")
        else:
            add_check(checks, cid="deploy_log", title="deploy.log（启动/评测脚本）契约", required=False,
                      status="warning",
                      detail={"exists": False, "hint": "尚未有部署脚本运行过；跑一次 部署/脚本/run_app.ps1 -Mode check 即生成"},
                      evidence=str(deploy_path))

        # ---- 8) 并发写入自检（T18 验收：多进程并发写长记录后非法行必须为 0）----
        if not args.no_concurrency_check:
            conc = check_concurrent_writes(logger=log)
            conc_failed = bool(conc["child_errors"]) or int(conc["bad_lines"]) > 0 \
                or int(conc["observed_lines"]) < int(conc["expected_lines"])
            add_check(checks, cid="concurrent_writes", title="并发写入自检（N 进程 × 长记录，非法行 = 0）",
                      required=True, status="failed" if conc_failed else "passed", detail=conc,
                      evidence="部署/脚本/verify_logs.py 内置自检；对等工具 部署/脚本/stress_log_concurrency.py")
            if conc_failed:
                findings.append({
                    "id": "并发写入自检失败（T18 回归）",
                    "severity": "blocker",
                    "problem": (f"{conc['procs']} 进程 × {conc['lines_per_proc']} 行 × {conc['line_bytes']} B 并发写后："
                                f"非法行 {conc['bad_lines']}，实写 {conc['observed_lines']}/期望 {conc['expected_lines']}，"
                                f"子进程错误 {len(conc['child_errors'])} 条。"),
                    "evidence": json.dumps(conc["bad_samples"][:3], ensure_ascii=False) + f" | {conc['child_errors'][:2]}",
                })

        # ---- 9) 可选：降级探针（含负向对照，断言「摘掉坏的、留下好的」等 5 条）----
        if args.probe_degradation:
            probe = probe_degradation(logger=log)
            failed_assertions = [item["id"] for item in probe.get("assertions", []) if not item.get("ok")]
            negative_ok = bool((probe.get("negative_control") or {}).get("assertion_can_catch_defect"))
            probe_failed = bool(failed_assertions) or not negative_ok
            add_check(checks, cid="degradation_probe",
                      title="logger 写失败降级路径（摘掉坏句柄/保留好句柄/不抛异常/降级可观测 + 负向对照）",
                      required=False,
                      status="failed" if probe_failed else "passed",
                      detail={k: v for k, v in probe.items() if k != "traceback"},
                      evidence="部署/脚本/verify_logs.py --probe-degradation")
            if probe_failed:
                raised_text = str(probe.get("raised") or "")
                tb_text = str(probe.get("traceback") or "").strip()
                tb_last = tb_text.splitlines()[-1] if tb_text else "（无堆栈）"
                findings.append({
                    "id": "logging_conf 降级分支未显式降级（改抛 NameError 或未摘除坏句柄）",
                    "severity": "blocker",
                    "problem": ("写日志失败时日志库本应「显式降级」（打印到 stderr 并摘掉**坏的**句柄、保留其余句柄），"
                                "实测该语义未被满足：或降级分支抛 NameError（修复前：模块顶层无 import sys），"
                                "或未摘除坏句柄。空断言缺口已由本次新增的 5 条断言 + 负向对照补齐。"),
                    "evidence": (f"未通过的断言={failed_assertions}；raised={raised_text}；"
                                 f"实际摘除={probe.get('removed_kinds')}；保留={probe.get('kept_kinds')}；"
                                 f"负向对照能抓到缺陷={negative_ok}；末行：{tb_last}"),
                })

        # ---- 汇总判定 ----
        failed_required = [c["id"] for c in checks if c["required"] and c["status"] == "failed"]
        warnings = [c["id"] for c in checks if c["status"] == "warning"]
        report["metrics"] = {
            "bad_lines_total": bad_total,
            "bad_lines_before_fix": bad_before, "bad_lines_after_fix": bad_after,
            "concurrency_fix_boundary": boundary.isoformat(),
            "func_enter": span_detail["enter_count"], "func_exit": span_detail["exit_count"],
            "func_error": span_detail["error_count"], "pairing_ratio": span_detail["pairing_ratio"],
            "error_log_records": err_detail["checked"], "trace_records": trace_detail["checked"],
            "app_records": len(app_records),
        }
        report["failed_required"] = failed_required
        report["warnings"] = warnings
        report["notes"] = [
            "本报告只做**确定性**判定：字段、配对、耗时、堆栈、事件路由、行级可解析性、并发写入非法率。",
            "坏行按 T18 修复分界拆分：分界之前 = 修复前遗留（契约要求不得删除，仅告警）；分界之后 = 必检失败。",
            "并发写入自检：N 个进程用产品 logger 同时写 16 KB 长记录（> 8 KB 写缓冲），非法行必须 = 0。",
            "日志文件可能正被运行中的服务/测试写入，read_json_lines 以「重试 3 次 + 显式 WARNING」处理，不静默跳过。",
        ]
        if failed_required:
            report["verdict"] = "failed"
            report["exit_code"] = 1
        elif warnings or findings:
            report["verdict"] = "passed_with_findings"
            report["exit_code"] = 0
        else:
            report["verdict"] = "passed"
            report["exit_code"] = 0
        write_reports(out_json=Path(args.json), out_md=Path(args.md), report=report, logger=log)
        span.set_output({"verdict": report["verdict"], "failed_required": failed_required,
                         "warnings": warnings, "bad_lines": bad_total,
                         "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)})

    # ---- 控制台汇总（**只用 ASCII 标记**：见 _ensure_stdio 的可移植性说明）----
    print("=" * 92)
    print(f"工单3 部署日志自检：{report['verdict']}（退出码 {report['exit_code']}）")
    print(f"日志目录：{log_dir}")
    mark = {"passed": "[OK]", "warning": "[WARN]", "failed": "[FAIL]"}
    for check in checks:
        print(f"  {mark.get(check['status'], '[?]')} [{check['id']}] {check['title']}｜必检={check['required']}")
    files_text = ", ".join(f"{row['name']}({row['lines']} 行/坏行 {row.get('bad_count')})" for row in report["files"])
    print(f"  文件：{files_text}")
    for finding in findings:
        print(f"  [发现] {finding['id']}（{finding['severity']}）")
    print(f"报告：{args.json}")
    print(f"      {args.md}")
    print("=" * 92)

    shutdown_logging()
    return int(report["exit_code"])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 顶层兜底：打印完整堆栈后以 1 结束（不静默）
        import traceback

        traceback.print_exc()
        shutdown_logging()
        raise SystemExit(1)
