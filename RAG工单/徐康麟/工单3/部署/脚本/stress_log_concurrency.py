# -*- coding: utf-8 -*-
"""工单3 日志并发写入压测与非法行校验（部署验证工具）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

用途：为「多进程并发追加同一日志文件会不会写出非法 JSON 行」提供**可复现的对照实验**。

三种写入策略（``--strategy``）：
    * ``legacy``  —— 复刻缺陷时代的写法：``open(path, "a", encoding="utf-8")`` +
      ``handle.write(line + "\\n")`` + ``handle.flush()``（文本模式 → BufferedWriter；
      超长记录会被拆成多次 ``write`` 系统调用 → 与其它进程交错）。
    * ``atomic``  —— 修复后的写法：``open(path, "ab", buffering=0)``（O_APPEND）+ **每行一次
      ``write``**，整行在同一次系统调用内追加。
    * ``product`` —— 直接使用产品代码 ``app.core.logging_conf.StructuredLogger``（即真实修复代码路径）。

典型用法（工作目录 = E:\\gao6gongdan\\工单3）：

    # ① 修复前对照：4 进程 × 400 行 × 16KB 长记录，文本模式写入
    pwsh -NoProfile -File run_py.ps1 部署/脚本/stress_log_concurrency.py \
        --strategy legacy --procs 4 --lines 400 --line-bytes 16384 --out-dir 部署/日志/_stress_legacy

    # ② 修复后对照：同样参数，原子追加
    pwsh -NoProfile -File run_py.ps1 部署/脚本/stress_log_concurrency.py \
        --strategy atomic --procs 4 --lines 400 --line-bytes 16384 --out-dir 部署/日志/_stress_atomic

    # ③ 产品代码路径 + 真实日志目录（多进程同时写 部署/日志/app.log）
    pwsh -NoProfile -File run_py.ps1 部署/脚本/stress_log_concurrency.py \
        --strategy product --procs 4 --lines 300 --line-bytes 16384 --out-dir 部署/日志 --tag t18fix

    # ④ 只校验既有文件（不写入）
    pwsh -NoProfile -File run_py.ps1 部署/脚本/stress_log_concurrency.py \
        --validate-only 部署/日志/app.log --since-ts 2026-10-04T19:00:00+08:00

退出码：0 = 压测执行成功且**非法行 = 0**（``legacy`` 策略下非法行 > 0 属预期，此时退出码 1）；
        1 = 存在非法行或校验失败；2 = 参数错误。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"


# ---------------------------------------------------------------------------
# 工具自身的结构化日志（独立文件，避免污染被校验的 app.log）
# ---------------------------------------------------------------------------
def log_event(path: Path, event: str, *, level: str = "INFO", func: str = "", inputs: Any = None,
              outputs: Any = None, elapsed_ms: float | None = None, error: Any = None) -> None:
    """写一条 JSON Lines 结构化事件（ts/level/event/func/inputs/outputs/elapsed_ms/error）。"""
    record = {
        "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "level": level, "event": event, "func": func, "work_order": WORK_ORDER,
        "module": "部署/脚本/stress_log_concurrency.py", "pid": os.getpid(),
        "inputs": inputs, "outputs": outputs, "elapsed_ms": elapsed_ms, "error": error,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    if level in {"ERROR", "CRITICAL", "WARNING"}:
        print(f"[{level}] {event} :: {func} :: {error or outputs}")


# ---------------------------------------------------------------------------
# 写入端（子进程执行）
# ---------------------------------------------------------------------------
def build_line(index: int, line_bytes: int, tag: str) -> str:
    """构造一条**符合产品日志字段规范**的长记录（长度可控，用于逼出写缓冲拆分）。"""
    filler = "样本正文" * max(1, line_bytes // 12)
    record = {
        "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "level": "INFO", "event": f"stress.{tag}", "func": "stress_writer", "module": "stress",
        "run_id": f"r-stress-{tag}", "trace_id": f"t{index:06d}", "span_id": f"s{index:06d}",
        "stage": "core", "pid": os.getpid(), "thread": "MainThread", "work_order": WORK_ORDER,
        "inputs": {"index": index, "tag": tag, "payload": filler},
        "outputs": {"ok": True}, "elapsed_ms": 0.01,
    }
    return json.dumps(record, ensure_ascii=False, default=str)


def write_legacy(path: Path, lines: int, line_bytes: int, tag: str, sleep_ms: float) -> int:
    """**缺陷复刻**写法：文本模式 append + write + flush（与修复前的 logging_conf 完全一致）。"""
    handle = path.open("a", encoding="utf-8")
    try:
        for index in range(lines):
            handle.write(build_line(index, line_bytes, tag) + "\n")
            handle.flush()                       # 与旧实现一致：每行 flush，但缓冲层仍可能拆多次 write
            if sleep_ms:
                time.sleep(sleep_ms)
    finally:
        handle.close()
    return lines


def write_single(path: Path, lines: int, line_bytes: int, tag: str, sleep_ms: float) -> int:
    """**只改单次 write、不加跨进程锁**：O_APPEND + 无缓冲二进制 + 每行一次 write。

    实测（Windows）：**这只解决了一部分**——16 KB 长记录仍偶发交错（内核会把大 write 拆块，
    两个进程的块可以互相插入）。保留该策略是为了让「为什么必须加锁」有实测依据。
    """
    handle = path.open("ab", buffering=0)
    try:
        for index in range(lines):
            handle.write((build_line(index, line_bytes, tag) + "\n").encode("utf-8", errors="backslashreplace"))
            if sleep_ms:
                time.sleep(sleep_ms)
    finally:
        handle.close()
    return lines


def write_locked(path: Path, lines: int, line_bytes: int, tag: str, sleep_ms: float) -> int:
    """**修复后写法**：O_APPEND + 单次 write + **跨进程写锁**（Windows msvcrt 字节锁 / POSIX flock）。

    锁与产品代码 ``app.core.logging_conf._lock_handle`` 同一套实现（此处内联以避免 import 依赖），
    保证「整行 + 换行」的写入在任何并发下都不被其它进程插入。
    """
    handle = path.open("ab", buffering=0)
    fd = handle.fileno()
    try:
        for index in range(lines):
            payload = (build_line(index, line_bytes, tag) + "\n").encode("utf-8", errors="backslashreplace")
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                try:
                    handle.write(payload)
                    handle.flush()
                finally:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    handle.write(payload)
                    handle.flush()
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            if sleep_ms:
                time.sleep(sleep_ms)
    finally:
        handle.close()
    return lines


def write_atomic(path: Path, lines: int, line_bytes: int, tag: str, sleep_ms: float) -> int:
    """兼容旧名：等价于 ``write_locked``。"""
    return write_locked(path, lines, line_bytes, tag, sleep_ms)


def write_product(log_dir: Path, lines: int, line_bytes: int, tag: str, sleep_ms: float) -> int:
    """走**产品真实代码路径**（StructuredLogger），验证修复后的 logger 本身。"""
    from app.core.logging_conf import StructuredLogger   # noqa: PLC0415 —— 仅此角色需要

    logger = StructuredLogger(run_id=f"r-stress-{tag}", log_dir=log_dir, module="stress", stage="core",
                              echo_error=False, level="INFO")
    filler = "样本正文" * max(1, line_bytes // 12)
    try:
        for index in range(lines):
            logger.log_event(f"stress.{tag}", index=index, payload=filler, trace_id=f"t{index:06d}")
            if sleep_ms:
                time.sleep(sleep_ms)
    finally:
        logger.close()
    return lines


def writer_main(args: argparse.Namespace) -> int:
    """子进程入口：按策略写 N 行后退出（退出码非 0 表示写入失败，父进程会判失败）。"""
    target = Path(args.target)
    tag = args.tag
    started = time.perf_counter()
    if args.strategy == "legacy":
        written = write_legacy(target, args.lines, args.line_bytes, tag, args.sleep_ms)
    elif args.strategy == "single":
        written = write_single(target, args.lines, args.line_bytes, tag, args.sleep_ms)
    elif args.strategy in ("atomic", "locked"):
        written = write_locked(target, args.lines, args.line_bytes, tag, args.sleep_ms)
    else:
        written = write_product(target.parent, args.lines, args.line_bytes, tag, args.sleep_ms)
    print(json.dumps({"role": "writer", "pid": os.getpid(), "written": written, "target": str(target),
                      "strategy": args.strategy, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)},
                     ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# 校验端
# ---------------------------------------------------------------------------
def validate(path: Path, *, since_ts: str | None, sample_limit: int, logger_path: Path,
             func: str = "validate") -> dict[str, Any]:
    """逐行校验：统计总行数、非法 JSON 行（含行号、成因、**前一条合法记录的 ts**）。"""
    started = time.perf_counter()
    log_event(logger_path, "func.enter", func=func,
              inputs={"path": str(path), "since_ts": since_ts, "exists": path.is_file()})
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "size_bytes": 0, "lines": 0,
                              "bad": 0, "bad_before_fix": 0, "bad_after_fix": 0, "samples": [],
                              "first_ts": None, "last_ts": None}
    if not result["exists"]:
        log_event(logger_path, "func.error", level="ERROR", func=func, inputs={"path": str(path)},
                  error={"type": "FileNotFoundError", "message": "目标文件不存在"})
        raise FileNotFoundError(f"目标文件不存在：{path}")

    result["size_bytes"] = path.stat().st_size
    boundary = None
    if since_ts:
        boundary = datetime.fromisoformat(since_ts)
    last_ts_raw: str | None = None
    last_ts_dt: datetime | None = None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            result["lines"] += 1
            ok = False
            ts_value = None
            if line.startswith("{"):
                try:
                    record = json.loads(line)
                    ok = isinstance(record, dict)
                    if ok:
                        ts_value = record.get("ts")
                except json.JSONDecodeError:
                    ok = False
            if ok:
                if isinstance(ts_value, str):
                    if result["first_ts"] is None:
                        result["first_ts"] = ts_value
                    result["last_ts"] = ts_value
                    last_ts_raw = ts_value
                    try:
                        last_ts_dt = datetime.fromisoformat(ts_value)
                    except ValueError:
                        last_ts_dt = None
                continue
            # —— 非法行 ——
            result["bad"] += 1
            after_fix = bool(boundary is not None and last_ts_dt is not None and last_ts_dt >= boundary)
            if after_fix:
                result["bad_after_fix"] += 1
            else:
                result["bad_before_fix"] += 1
            if len(result["samples"]) < sample_limit:
                result["samples"].append({
                    "line": lineno, "reason": "非 JSON 行首" if not line.startswith("{") else "JSON 解析失败",
                    "after_fix": after_fix, "preceding_valid_ts": last_ts_raw, "sample": line[:200],
                })
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    result["bad_rate"] = round(result["bad"] / max(1, result["lines"]), 6)
    log_event(logger_path, "func.exit", func=func, inputs={"path": str(path)},
              outputs={k: result[k] for k in ("lines", "bad", "bad_before_fix", "bad_after_fix", "bad_rate")},
              elapsed_ms=result["elapsed_ms"])
    return result


# ---------------------------------------------------------------------------
# 驱动端
# ---------------------------------------------------------------------------
def driver_main(args: argparse.Namespace) -> int:
    """父进程：起 N 个进程并发写同一文件 → 校验非法行 → 打印对照结论。"""
    repo_log = Path(args.out_dir) / "_stress_driver.log"
    target = Path(args.target) if args.target else Path(args.out_dir) / "app.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    log_event(repo_log, "func.enter", func="driver_main",
              inputs={"strategy": args.strategy, "procs": args.procs, "lines": args.lines,
                      "line_bytes": args.line_bytes, "target": str(target)})

    before = validate(target, since_ts=None, sample_limit=0, logger_path=repo_log, func="validate_before") \
        if target.exists() else {"lines": 0, "bad": 0, "size_bytes": 0}

    children: list[subprocess.Popen] = []
    for index in range(args.procs):
        command = [sys.executable, str(Path(__file__).resolve()), "--role", "writer",
                   "--strategy", args.strategy, "--lines", str(args.lines),
                   "--line-bytes", str(args.line_bytes), "--tag", f"{args.tag}-w{index}",
                   "--sleep-ms", str(args.sleep_ms)]
        if args.strategy == "product":
            command += ["--target", str(target)]
        else:
            command += ["--target", str(target)]
        children.append(subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, encoding="utf-8"))
    failures = 0
    per_child: list[dict[str, Any]] = []
    for child in children:
        out, err = child.communicate()
        if child.returncode != 0:
            failures += 1
            log_event(repo_log, "stress.child_failed", level="ERROR", func="driver_main",
                      inputs={"pid": child.pid}, error={"exit_code": child.returncode, "stderr": err[-500:]})
        for raw in (out or "").splitlines():
            raw = raw.strip()
            if raw.startswith("{"):
                per_child.append(json.loads(raw))
    expected_new = args.procs * args.lines
    after = validate(target, since_ts=args.since_ts, sample_limit=args.sample_limit, logger_path=repo_log,
                     func="validate_after")
    new_lines = after["lines"] - before["lines"]
    elapsed = round((time.perf_counter() - started) * 1000, 2)

    summary = {
        "strategy": args.strategy, "procs": args.procs, "lines_per_proc": args.lines,
        "line_bytes": args.line_bytes, "expected_new_lines": expected_new, "observed_new_lines": new_lines,
        "children_ok": args.procs - failures, "children_failed": failures,
        "before": {"lines": before["lines"], "bad": before["bad"], "size_bytes": before["size_bytes"]},
        "after": {"lines": after["lines"], "bad": after["bad"], "bad_before_fix": after["bad_before_fix"],
                  "bad_after_fix": after["bad_after_fix"], "bad_rate": after["bad_rate"],
                  "size_bytes": after["size_bytes"]},
        "bad_samples": after["samples"], "elapsed_ms": elapsed,
    }
    print("=" * 96)
    print(f"并发压测（策略 {args.strategy}）：{args.procs} 进程 × {args.lines} 行 × {args.line_bytes} B/行")
    print(f"  写入前：{before['lines']} 行 / 非法 {before['bad']}")
    print(f"  新增行：期望 {expected_new}，实测 {new_lines}（子进程成功 {args.procs - failures}/{args.procs}）")
    print(f"  写入后：{after['lines']} 行 / 非法 {after['bad']}"
          f"（修复前遗留 {after['bad_before_fix']}，修复后新增 {after['bad_after_fix']}）非法率 {after['bad_rate']}")
    for sample in after["samples"]:
        print(f"    [BAD] 第 {sample['line']} 行：{sample['reason']}｜前一条合法记录 ts={sample['preceding_valid_ts']}"
              f"｜修复后={sample['after_fix']}｜{sample['sample'][:120]}")
    print(f"  耗时 {elapsed} ms；明细 {repo_log}")
    print("=" * 96)

    log_event(repo_log, "func.exit", func="driver_main", inputs={"strategy": args.strategy},
              outputs={"new_lines": new_lines, "bad_after_fix": after["bad_after_fix"], "failures": failures},
              elapsed_ms=elapsed)
    ok = failures == 0 and after["bad_after_fix"] == 0 and new_lines >= expected_new
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """命令行入口：``--role driver|writer`` 或 ``--validate-only``。"""
    parser = argparse.ArgumentParser(description="工单3 日志并发写入压测与非法行校验")
    parser.add_argument("--role", choices=["driver", "writer"], default="driver")
    parser.add_argument("--strategy", choices=["legacy", "single", "atomic", "locked", "product"],
                        default="atomic")
    parser.add_argument("--procs", type=int, default=4, help="并发进程数")
    parser.add_argument("--lines", type=int, default=400, help="每个进程写入行数")
    parser.add_argument("--line-bytes", type=int, default=16384,
                        help="每行目标字节数（默认 16 KB，> 写缓冲 8 KB 才会触发多次 write）")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "部署" / "日志" / "_stress"))
    parser.add_argument("--target", default="", help="写入目标文件（默认 <out-dir>/app.log）")
    parser.add_argument("--tag", default="stress", help="事件名后缀（stress.<tag>），用于事后检索")
    parser.add_argument("--sleep-ms", type=float, default=0.0, help="每行之间的间隔（放大交错概率）")
    parser.add_argument("--since-ts", default="", help="分界时间戳（ISO8601）：之后的非法行记入 bad_after_fix")
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--validate-only", default="", help="只校验该文件，不写入")
    args = parser.parse_args(argv)

    # 可移植性兜底：CP936 等非 UTF-8 终端下，非 ASCII 字符会让 print 抛 UnicodeEncodeError，
    # 使「检查通过却退出码 1」。放宽错误策略（保留原编码，避免中文乱码），并把退出码与打印解耦。
    for _stream in (sys.stdout, sys.stderr):
        _reconfigure = getattr(_stream, "reconfigure", None)
        if callable(_reconfigure):
            try:
                _reconfigure(errors="replace")
            except (ValueError, OSError) as _exc:  # 显式降级：改不了也不阻断
                print(f"[降级] 无法调整 {_stream!r} 的错误策略：{type(_exc).__name__}: {_exc}", flush=True)

    if args.validate_only:
        path = Path(args.validate_only)
        repo_log = path.parent / "_stress_driver.log"
        started = time.perf_counter()
        result = validate(path, since_ts=args.since_ts or None, sample_limit=args.sample_limit, logger_path=repo_log)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"耗时 {round((time.perf_counter() - started) * 1000, 2)} ms（明细 {repo_log}）")
        return 0 if result["bad_after_fix"] == 0 else 1

    if args.role == "writer":
        return writer_main(args)
    return driver_main(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 顶层兜底：打印完整堆栈后以 1 结束（禁止静默）
        traceback.print_exc()
        raise SystemExit(1)
