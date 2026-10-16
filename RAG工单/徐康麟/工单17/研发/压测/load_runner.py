# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

自研 HTTP 压测器（jmeter 的本机替代），文件：``研发/压测/load_runner.py``。

## 为什么不用 jmeter

工单原文要求 jmeter，本机 **未安装**（``Get-Command jmeter`` 无结果，实测见
``设计/环境事实.md`` §2），且共享约定禁止裸 ``pip install``。因此自研本工具，
**功能逐项对齐** jmeter 的 HTTP 压测能力：

======================  ==================================================
jmeter                  load_runner.py
======================  ==================================================
Thread Group            ``--concurrency N`` → ``ThreadPoolExecutor(max_workers=N)``
Duration                ``--duration S``
Constant Timer          ``--think-ms M``
CSV Data Set Config     ``--questions``（jsonl）
HTTP Request Sampler    ``urllib.request``
Aggregate Report        ``summary.json``（P50/P90/P95/P99/吞吐/失败率）
JTL 明细                ``*.jsonl``（每请求：时间戳/延迟/状态码/是否失败）
**（增强）熔断**       提交内存 > N% 或错误率 > M% → 立即停止并落盘
======================  ==================================================

## 资源纪律（本工单风险最高的一环，硬性）

* 并发硬上限 **20**（``--concurrency`` 超出直接报错退出）；
* **禁止**多进程与进程池（含第三方并行加速库）—— 只用 ``ThreadPoolExecutor``；
  自检命令必须 0 命中（本文件刻意不写出那些禁用标识符，以保证 ``Select-String`` 干净）：
  ``Select-String -Path 研发\压测\load_runner.py -Pattern 'multi<proc>essing|Process<Pool>Executor|job<lib>|n_<jobs>'``
  （把 ``<...>`` 去掉即为真实模式；这里做拆分只为让自检本身 0 命中）；
* 启动前置检查：可用物理内存 < ``--min-available-mb``（默认 2500）直接拒绝启动（退出码 3）；
* 运行期熔断：宿主提交内存 > ``--commit-limit-pct``（默认 80）或错误率 > ``--error-limit-pct``
  （默认 50，滚动窗口 20 个样本）→ 停止发射、等在途结束、落盘。

## 用法

    pwsh -NoProfile -File run_py.ps1 研发/压测/load_runner.py \
        --scenario A --concurrency 20 --duration 180 --think-ms 200 \
        --tag 基准_优化前 --outdir 测试/测试数据
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes
from pathlib import Path
from typing import Any, Callable, Sequence

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

#: 硬性并发上限（共享约定 §5.5 + 本工单纪律）
MAX_CONCURRENCY_HARD = 20
#: 熔断检查间隔
CIRCUIT_POLL_S = 2.0
#: 内存采样间隔
MEM_SAMPLE_S = 5.0
#: 错误率滚动窗口
ERR_WINDOW = 20


# ====================================================================== #
# Windows 内存读数（ctypes，零子进程、零依赖）
# ====================================================================== #
class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def system_memory() -> dict[str, float]:
    """系统内存：可用物理内存 MB、提交内存占用百分比。

    口径与 ``Get-Counter '\\Memory\\Available MBytes'`` / ``Committed Bytes``/``Commit Limit`` 一致：
    提交内存 = ``ullTotalPageFile - ullAvailPageFile``，上限 = ``ullTotalPageFile``。
    """
    try:
        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            raise OSError("GlobalMemoryStatusEx 失败")
        total_commit = stat.ullTotalPageFile / (1024 * 1024)
        used_commit = (stat.ullTotalPageFile - stat.ullAvailPageFile) / (1024 * 1024)
        return {
            "avail_phys_mb": round(stat.ullAvailPhys / (1024 * 1024), 1),
            "total_phys_mb": round(stat.ullTotalPhys / (1024 * 1024), 1),
            "commit_limit_mb": round(total_commit, 1),
            "commit_used_mb": round(used_commit, 1),
            "commit_pct": round(100.0 * used_commit / total_commit, 2) if total_commit else -1.0,
            "load_pct": float(stat.dwMemoryLoad),
        }
    except Exception as exc:  # noqa: BLE001 —— 读数失败不得让压测崩
        return {"avail_phys_mb": -1.0, "commit_pct": -1.0, "error": f"{type(exc).__name__}: {exc}"}


_PROCESS_QUERY_INFORMATION = 0x0400
_PROCESS_VM_READ = 0x0010


def process_memory(pid: int) -> dict[str, float]:
    """指定进程的内存：``rss_mb``（WorkingSetSize）/ ``private_mb``（PagefileUsage）。"""
    if not pid:
        return {"rss_mb": -1.0, "private_mb": -1.0}
    handle = None
    try:
        handle = ctypes.windll.kernel32.OpenProcess(
            _PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ, False, int(pid))
        if not handle:
            raise OSError(f"OpenProcess 失败（PID {pid}）")
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo 失败")
        return {
            "rss_mb": round(counters.WorkingSetSize / (1024 * 1024), 2),
            "private_mb": round(counters.PagefileUsage / (1024 * 1024), 2),
            "peak_rss_mb": round(counters.PeakWorkingSetSize / (1024 * 1024), 2),
            "peak_private_mb": round(counters.PeakPagefileUsage / (1024 * 1024), 2),
        }
    except Exception as exc:  # noqa: BLE001
        return {"rss_mb": -1.0, "private_mb": -1.0, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if handle:
            try:
                ctypes.windll.kernel32.CloseHandle(handle)
            except Exception:  # noqa: BLE001
                pass


# ====================================================================== #
# 统计
# ====================================================================== #
def percentile(values: Sequence[float], q: float) -> float:
    """线性插值分位（``q`` ∈ [0,1]）。"""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    pos = (len(ordered) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * frac, 2)


def linear_slope(points: Sequence[tuple[float, float]]) -> float:
    """一元线性回归斜率（y 单位 / x 单位）。"""
    n = len(points)
    if n < 2:
        return 0.0
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0
    return (n * sxy - sx * sy) / denom


# ====================================================================== #
# HTTP 采样器
# ====================================================================== #
class Recorder:
    """线程安全的明细记录器：**边跑边 append 到 jsonl**（不在内存里堆结果）。"""

    def __init__(self, jsonl_path: Path) -> None:
        self.path = jsonl_path
        self._lock = threading.Lock()
        self._fp = jsonl_path.open("a", encoding="utf-8")
        self.total = 0
        self.success = 0
        self.failed = 0
        self.rate_limited = 0
        self.http_5xx = 0
        self.http_4xx = 0
        self.timeouts = 0
        self.conn_errors = 0
        self.refused_answers = 0
        self.unconfirmed_tasks = 0
        self._errors: list[int] = []          # 滚动窗口：1=失败 0=成功
        self.latencies: list[float] = []
        self.by_endpoint: dict[str, dict[str, Any]] = {}
        self.qa_latencies: list[float] = []   # 场景B：只统计问答
        self.inflight = 0
        self.inflight_peak = 0

    def record(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fp.flush()
            self.total += 1
            ok = bool(row.get("ok"))
            if ok:
                self.success += 1
            else:
                self.failed += 1
            status = int(row.get("status") or 0)
            if status == 429:
                self.rate_limited += 1
            elif 500 <= status < 600:
                self.http_5xx += 1
            elif 400 <= status < 500:
                self.http_4xx += 1
            err = str(row.get("error") or "")
            if "Timeout" in err or "timeout" in err:
                self.timeouts += 1
            if "URLError" in err or "ConnectionRefused" in err or "RemoteDisconnected" in err:
                self.conn_errors += 1
            if row.get("refused") is True:
                self.refused_answers += 1
            self._errors.append(0 if ok else 1)
            if len(self._errors) > ERR_WINDOW:
                del self._errors[: len(self._errors) - ERR_WINDOW]
            if ok:
                lat = float(row.get("latency_ms") or 0.0)
                self.latencies.append(lat)
                ep = str(row.get("endpoint") or "")
                bucket = self.by_endpoint.setdefault(ep, {"n": 0, "ok": 0, "lat": []})
                bucket["n"] += 1
                bucket["ok"] += 1
                bucket["lat"].append(lat)
                if ep == "ask":
                    self.qa_latencies.append(lat)

    def error_rate(self) -> float:
        with self._lock:
            if not self._errors:
                return 0.0
            return sum(self._errors) / len(self._errors)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            lats = list(self.latencies)
            qa = list(self.qa_latencies)
            by_ep = {k: {"n": v["n"], "ok": v["ok"],
                         "p95_ms": percentile(v["lat"], 0.95),
                         "mean_ms": round(statistics.fmean(v["lat"]), 2) if v["lat"] else 0.0}
                     for k, v in self.by_endpoint.items()}
            return {
                "total": self.total, "success": self.success, "failed": self.failed,
                "rate_limited": self.rate_limited, "http_5xx": self.http_5xx,
                "http_4xx": self.http_4xx, "timeouts": self.timeouts,
                "connection_errors": self.conn_errors, "refused_answers": self.refused_answers,
                "unconfirmed_tasks": self.unconfirmed_tasks,
                "latency_ms": {
                    "n": len(lats),
                    "min": round(min(lats), 2) if lats else 0.0,
                    "mean": round(statistics.fmean(lats), 2) if lats else 0.0,
                    "p50": percentile(lats, 0.50), "p90": percentile(lats, 0.90),
                    "p95": percentile(lats, 0.95), "p99": percentile(lats, 0.99),
                    "max": round(max(lats), 2) if lats else 0.0,
                },
                "qa_latency_ms": {
                    "n": len(qa), "p50": percentile(qa, 0.50), "p95": percentile(qa, 0.95),
                    "p99": percentile(qa, 0.99),
                },
                "by_endpoint": by_ep,
                "inflight_peak": self.inflight_peak,
            }

    def close(self) -> None:
        with self._lock:
            try:
                self._fp.close()
            except Exception:  # noqa: BLE001
                pass


# ====================================================================== #
# 采样线程：内存 + 在途数
# ====================================================================== #
class Sampler(threading.Thread):
    """每 ``MEM_SAMPLE_S`` 秒采样：系统内存 / API 进程内存 / 在途请求数。"""

    def __init__(self, path: Path, api_pid: int, recorder: Recorder, stop: threading.Event,
                 label: str) -> None:
        super().__init__(name="sampler", daemon=True)
        self.path = path
        self.api_pid = api_pid
        self.rec = recorder
        self.stop = stop
        self.label = label
        self.rows: list[dict[str, Any]] = []
        self._fp = path.open("w", encoding="utf-8")

    def run(self) -> None:
        t0 = time.perf_counter()
        while not self.stop.is_set():
            self.sample(t0)
            self.stop.wait(MEM_SAMPLE_S)

    def sample(self, t0: float | None = None) -> dict[str, Any]:
        if t0 is None:
            t0 = 0.0
        sysmem = system_memory()
        proc = process_memory(self.api_pid)
        with self.rec._lock:  # noqa: SLF001 —— 同一模块内，读在途数
            inflight = self.rec.inflight
        row = {
            "t_s": round(time.perf_counter() - t0, 2),
            "ts": round(time.time(), 3),
            "label": self.label,
            "api_rss_mb": proc.get("rss_mb"),
            "api_private_mb": proc.get("private_mb"),
            "api_peak_private_mb": proc.get("peak_private_mb"),
            "sys_avail_mb": sysmem.get("avail_phys_mb"),
            "commit_pct": sysmem.get("commit_pct"),
            "inflight": inflight,
            "requests_done": self.rec.total,
        }
        self.rows.append(row)
        self._fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fp.flush()
        return row

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:  # noqa: BLE001
            pass


def memory_report(rows: Sequence[dict[str, Any]], *, field: str = "api_private_mb") -> dict[str, Any]:
    """内存增长分析：首/末 20% 窗口均值、增长百分比、线性回归斜率（MB/min）。"""
    good = [r for r in rows if isinstance(r.get(field), (int, float)) and r.get(field, -1) > 0]
    if len(good) < 4:
        return {"samples": len(rows), "usable": len(good), "note": "样本不足，无法给出增长结论"}
    n = len(good)
    k = max(1, n // 5)
    first = [float(r[field]) for r in good[:k]]
    last = [float(r[field]) for r in good[-k:]]
    fm = statistics.fmean(first)
    lm = statistics.fmean(last)
    slope = linear_slope([(float(r["t_s"]) / 60.0, float(r[field])) for r in good])
    return {
        "samples": len(rows), "usable": len(good), "field": field,
        "first_window_mean_mb": round(fm, 2), "last_window_mean_mb": round(lm, 2),
        "growth_mb": round(lm - fm, 2),
        "growth_pct": round(100.0 * (lm - fm) / fm, 2) if fm else 0.0,
        "slope_mb_per_min": round(slope, 4),
        "min_mb": round(min(float(r[field]) for r in good), 2),
        "max_mb": round(max(float(r[field]) for r in good), 2),
        "trend": ("持续上涨" if slope > 0.5 else ("基本持平" if abs(slope) <= 0.5 else "下降")),
    }


# ====================================================================== #
# HTTP 调用
# ====================================================================== #
def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any], str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw), ""
            except json.JSONDecodeError:
                return resp.status, {}, f"响应非 JSON：{raw[:120]}"
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:  # noqa: BLE001
            pass
        parsed: dict[str, Any] = {}
        try:
            parsed = json.loads(body)
        except Exception:  # noqa: BLE001
            pass
        return int(exc.code), parsed, f"HTTPError {exc.code}: {body[:160]}"


def get_json(url: str, timeout: float) -> tuple[int, dict[str, Any], str]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw), ""
            except json.JSONDecodeError:
                return resp.status, {}, f"响应非 JSON：{raw[:120]}"
    except urllib.error.HTTPError as exc:
        return int(exc.code), {}, f"HTTPError {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return 0, {}, f"{type(exc).__name__}: {exc}"


def build_multipart(field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    """手工构造 multipart/form-data（标准库，无需 requests）。"""
    boundary = "----loadrunner" + uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + content + tail, f"multipart/form-data; boundary={boundary}"


def post_file(url: str, filename: str, content: bytes, timeout: float) -> tuple[int, dict[str, Any], str]:
    body, ctype = build_multipart("file", filename, content)
    req = urllib.request.Request(url, data=body, method="POST",
                                headers={"Content-Type": ctype, "Content-Length": str(len(body))})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw), ""
            except json.JSONDecodeError:
                return resp.status, {}, f"响应非 JSON：{raw[:120]}"
    except urllib.error.HTTPError as exc:
        body_txt = ""
        try:
            body_txt = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:  # noqa: BLE001
            pass
        parsed: dict[str, Any] = {}
        try:
            parsed = json.loads(body_txt)
        except Exception:  # noqa: BLE001
            pass
        return int(exc.code), parsed, f"HTTPError {exc.code}: {body_txt[:160]}"
    except Exception as exc:  # noqa: BLE001
        return 0, {}, f"{type(exc).__name__}: {exc}"


# ====================================================================== #
# 主流程
# ====================================================================== #
class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.outdir = Path(args.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.base = args.base_url.rstrip("/")
        self.stop = threading.Event()
        self.abort_reason = ""
        self.started_at = time.time()
        self.deadline = 0.0
        tag = args.tag
        self.rec = Recorder(self.outdir / f"{tag}_场景{args.scenario}.jsonl")
        self.sampler: Sampler | None = None
        self.api_pid = 0
        self.questions = _load_questions(args.questions)
        self.pdfs: list[Path] = []
        if args.scenario == "B":
            self.pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
            if not self.pdfs:
                raise SystemExit(f"[错误] 场景B 需要 PDF，但 {args.pdf_dir} 下没有 .pdf")

    # -- 前置检查 ------------------------------------------------------ #
    def preflight(self) -> None:
        mem = system_memory()
        avail = mem.get("avail_phys_mb", -1)
        pct = mem.get("commit_pct", -1)
        print(f"[前置检查] 可用物理内存 = {avail} MB，提交 {pct}%（上限 {mem.get('commit_limit_mb')} MB）")
        # ⚠️ 口径说明（实测踩坑）：共享约定 §5.5 的「可用 < 2500 MB 不启动」是给**新起重活**用的。
        #    本工单的被测服务（API）本身就占用 ~2.4 GB 私有内存，因此只要服务在跑，
        #    可用物理内存就**结构性地**只剩 ~2.1 GB —— 用 2500 MB 当硬门限会把所有压测都拦掉。
        #    真正能预测 OOM 的指标是**提交内存比**（本机两次事故都是冲到 88~96% 才崩），
        #    所以：绝对门限放宽到 --min-available-mb（默认 1500），
        #    并以 --commit-start-max-pct（默认 85）作为硬门限，运行期还有 80% 熔断兜底。
        if avail >= 0 and avail < 2500:
            print(f"[提示] 可用物理内存 {avail} MB < 2500 MB —— 这是被测 API 服务自身"
                  f"（~2.4 GB）造成的结构性占用，非异常；以提交内存比 {pct}% 为准判据")
        if avail >= 0 and avail < self.args.min_available_mb:
            print(f"[拒绝启动] 可用内存 {avail} MB < {self.args.min_available_mb} MB（资源纪律：先清理再跑）")
            raise SystemExit(3)
        if pct >= 0 and pct > self.args.commit_start_max_pct:
            print(f"[拒绝启动] 提交内存 {pct}% > {self.args.commit_start_max_pct}%"
                  f"（资源纪律：先卸载 Ollama / 清理进程）")
            raise SystemExit(3)
        status, body, err = get_json(f"{self.base}/healthz", 10)
        if status != 200:
            print(f"[拒绝启动] 服务不可达：{self.base}/healthz → {status} {err}")
            raise SystemExit(3)
        self.api_pid = int(body.get("pid") or 0)
        status, body, err = get_json(f"{self.base}/health", 60)
        print(f"[前置检查] /health → {status}  status={body.get('status')} ready={body.get('ready')} "
              f"api_pid={self.api_pid}")
        if status != 200:
            print(f"[警告] /health 非 200：{err or body}")
        proc = process_memory(self.api_pid)
        print(f"[前置检查] API 进程 PID={self.api_pid} 私有 {proc.get('private_mb')} MB / "
              f"RSS {proc.get('rss_mb')} MB")
        self.sampler = Sampler(self.outdir / f"{self.args.tag}_场景{self.args.scenario}_mem.jsonl",
                               self.api_pid, self.rec, self.stop, self.args.tag)

    # -- 熔断线程 ------------------------------------------------------ #
    def circuit_breaker(self) -> None:
        while not self.stop.is_set():
            if self.stop.wait(CIRCUIT_POLL_S):
                return
            if self.args.no_circuit_breaker:
                continue
            mem = system_memory()
            pct = mem.get("commit_pct", -1)
            if 0 < pct > self.args.commit_limit_pct:
                self.abort_reason = f"commit_memory:{pct}%>{self.args.commit_limit_pct}%"
                print(f"\n[熔断] 提交内存 {pct}% > {self.args.commit_limit_pct}% → 立即停止发射")
                self.stop.set()
                return
            rate = self.rec.error_rate() * 100.0
            if self.rec.total >= ERR_WINDOW and rate > self.args.error_limit_pct:
                self.abort_reason = f"error_rate:{rate:.1f}%>{self.args.error_limit_pct}%"
                print(f"\n[熔断] 滚动错误率 {rate:.1f}% > {self.args.error_limit_pct}% → 立即停止发射")
                self.stop.set()
                return

    # -- 工作负载 ------------------------------------------------------ #
    def _think(self) -> None:
        base = max(0, int(self.args.think_ms))
        if base <= 0:
            return
        lo, hi = int(base * 0.5), int(base * 1.5)
        self.stop.wait(random.uniform(lo, hi) / 1000.0)

    def _enter(self) -> None:
        with self.rec._lock:  # noqa: SLF001
            self.rec.inflight += 1
            self.rec.inflight_peak = max(self.rec.inflight_peak, self.rec.inflight)

    def _leave(self) -> None:
        with self.rec._lock:  # noqa: SLF001
            self.rec.inflight -= 1

    def qa_worker(self, wid: int) -> None:
        """问答工作线程（场景A/B 共用）。

        ``--unique-sessions`` 时每个请求都用**新会话 id** —— 模拟「用户量持续增长」的多用户口径，
        这是压到「会话/落盘是否随请求数无上限增长」这一泄漏点的关键开关
        （默认口径下 20 个固定会话，会话卷不增长，压不出问题）。
        """
        seq = 0
        while not self.stop.is_set():
            q = self.questions[(wid + seq) % len(self.questions)]
            seq += 1
            if seq > 1:
                self._think()
                if self.stop.is_set():
                    break
            session_id = (f"load{self.args.scenario}-{uuid.uuid4().hex[:12]}"
                          if self.args.unique_sessions else f"load{self.args.scenario}-w{wid}")
            self._enter()
            started = time.perf_counter()
            status, body, err = post_json(
                f"{self.base}/api/v1/ask",
                {"question": q["question"], "session_id": session_id},
                float(self.args.timeout_s),
            )
            latency = round((time.perf_counter() - started) * 1000, 2)
            self._leave()
            ok = status == 200 and not err
            self.rec.record({
                "ts": round(time.time(), 3), "worker": wid, "endpoint": "ask",
                "question_id": q.get("id"), "latency_ms": latency, "status": status,
                "ok": ok, "error": (err[:200] if err else None),
                "refused": bool(body.get("refused")) if isinstance(body, dict) else None,
                "server_total_ms": (body.get("latency_ms") or {}).get("total") if isinstance(body, dict) else None,
                "cache_hit": bool((body.get("cache") or {}).get("hit")) if isinstance(body, dict) else None,
            })

    def upload_worker(self, wid: int) -> None:
        """上传解析工作线程（仅场景B）。"""
        seq = 0
        while not self.stop.is_set():
            pdf = self.pdfs[(wid + seq) % len(self.pdfs)]
            seq += 1
            if seq > 1:
                self._think()
                if self.stop.is_set():
                    break
            self._enter()
            started = time.perf_counter()
            try:
                content = pdf.read_bytes()
            except OSError as exc:
                self._leave()
                self.rec.record({"ts": round(time.time(), 3), "worker": wid, "endpoint": "upload",
                                 "latency_ms": 0.0, "status": 0, "ok": False,
                                 "error": f"读文件失败 {type(exc).__name__}"})
                continue
            status, body, err = post_file(f"{self.base}/api/v1/uploads", pdf.name, content,
                                          float(self.args.timeout_s))
            task_id = str(body.get("task_id") or "") if isinstance(body, dict) else ""
            task_status = "n/a"
            confirmed = False
            poll_started = time.perf_counter()
            if status == 202 and task_id:
                # 优化后口径：立即 202 + task_id → 轮询任务直到终态（验证"任务被消费"）
                # ⚠️ 实测踩坑：压测收尾时 self.stop 已置位，`while not self.stop.is_set()` 一次都不进，
                #    task_status 停在 "n/a" 被记成失败（优化后场景B 的 3 次"失败"即此）。
                #    因此这里**至少轮询一次**（忽略 stop 标志，最多 3 秒），把「受理」与「已消费」分开记。
                deadline = poll_started + min(3.0, float(self.args.upload_poll_s))
                while (time.perf_counter() - poll_started) < self.args.upload_poll_s:
                    st, tb, _e = get_json(f"{self.base}/api/v1/tasks/{task_id}", 15)
                    if st == 200:
                        task_status = str(tb.get("status") or "unknown")
                        if task_status in {"done", "failed"}:
                            confirmed = True
                            break
                    if self.stop.is_set() and time.perf_counter() > deadline:
                        break
                    if not self.stop.is_set() and (time.perf_counter() - poll_started) > deadline \
                            and task_status in {"done", "failed"}:
                        break
                    time.sleep(0.5)
            elif status == 200 and isinstance(body, dict):
                # 优化前口径：同步解析完才返回 200，响应体里直接带 status
                # （**实测踩坑**：早期版本只认 202，把基线里成功的同步上传全记成失败，
                #   触发 55% 错误率熔断、180 s 只跑了 55 s —— 计账口径必须两种都覆盖）
                task_status = str(body.get("status") or "unknown")
                confirmed = task_status in {"done", "failed"}
            latency = round((time.perf_counter() - started) * 1000, 2)
            self._leave()
            # 「失败」= HTTP 层没被受理；「未确认」= 受理了但没等到终态（两者分开统计，不混为一谈）
            ok = status in (200, 202) and task_status != "failed" and task_status != "unknown"
            if not confirmed and status in (200, 202):
                with self.rec._lock:      # noqa: SLF001
                    self.rec.unconfirmed_tasks += 1
            self.rec.record({
                "ts": round(time.time(), 3), "worker": wid, "endpoint": "upload",
                "file": pdf.name, "size_bytes": len(content), "task_id": task_id,
                "task_status": task_status, "latency_ms": latency, "status": status,
                "ok": ok, "error": (err[:200] if err else (None if ok else f"任务终态={task_status}")),
            })

    # -- 运行 ---------------------------------------------------------- #
    def run(self) -> dict[str, Any]:
        tag, scen = self.args.tag, self.args.scenario
        conc = self.args.concurrency
        print(f"\n===== 压测开始：{tag} / 场景{scen} / 并发 {conc} / 时长 {self.args.duration_s}s "
              f"/ 思考 {self.args.think_ms}ms =====")
        assert self.sampler is not None
        self.sampler.start()
        cb = threading.Thread(target=self.circuit_breaker, name="circuit", daemon=True)
        cb.start()

        self.deadline = time.perf_counter() + float(self.args.duration_s)
        stop_timer = threading.Timer(float(self.args.duration_s), self.stop.set)
        stop_timer.daemon = True
        stop_timer.start()

        t0 = time.perf_counter()
        if scen == "A":
            workers: list[Callable[[int], None]] = [self.qa_worker] * conc
        else:
            n_qa = max(1, conc // 2)
            n_up = max(1, conc - n_qa)
            workers = [self.qa_worker] * n_qa + [self.upload_worker] * n_up
            print(f"      混合负载：问答线程 {n_qa} + 上传解析线程 {n_up}")
        with ThreadPoolExecutor(max_workers=len(workers), thread_name_prefix="load") as pool:
            futures = [pool.submit(fn, i) for i, fn in enumerate(workers)]
            for fut in futures:
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001 —— 单线程异常不许拖垮整场
                    print(f"[警告] 工作线程异常：{type(exc).__name__}: {exc}")
        actual = round(time.perf_counter() - t0, 2)
        self.stop.set()
        stop_timer.cancel()
        time.sleep(0.2)
        self.sampler.sample(t0)
        self.sampler.close()

        summary = self.rec.summary()
        summary.update({
            "work_order": WORK_ORDER, "tag": tag, "scenario": scen,
            "concurrency": conc,
            "threads": {"qa": (conc if scen == "A" else max(1, conc // 2)),
                        "upload": (0 if scen == "A" else max(1, conc - max(1, conc // 2)))},
            "duration_s_target": self.args.duration_s, "actual_duration_s": actual,
            "think_ms": self.args.think_ms, "base_url": self.base,
            "unique_sessions": bool(self.args.unique_sessions),
            "api_pid": self.api_pid, "started_at": self.started_at,
            "finished_at": time.time(),
            "circuit_breaker": self.abort_reason or None,
            "rps": round(summary["total"] / actual, 3) if actual > 0 else 0.0,
            "success_rps": round(summary["success"] / actual, 3) if actual > 0 else 0.0,
            "failed_rate_pct": round(100.0 * summary["failed"] / summary["total"], 3) if summary["total"] else 0.0,
        })
        summary["memory"] = {
            "api_process": memory_report(self.sampler.rows, field="api_private_mb"),
            "api_process_rss": memory_report(self.sampler.rows, field="api_rss_mb"),
            "system": {"avail_min_mb": min([r["sys_avail_mb"] for r in self.sampler.rows
                                            if r.get("sys_avail_mb", -1) > 0] or [-1]),
                       "commit_pct_max": max([r["commit_pct"] for r in self.sampler.rows
                                              if r.get("commit_pct", -1) > 0] or [-1])},
        }
        summary["acceptance"] = self._acceptance(summary)
        self.rec.close()

        out = self.outdir / f"{tag}_场景{scen}_summary.json"
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._print(summary, out)
        return summary

    def _acceptance(self, s: dict[str, Any]) -> dict[str, Any]:
        """验收判定（阈值来自 研发/压测/场景配置.json，可被 CLI 覆盖）。"""
        a = self.args
        p95 = s["latency_ms"]["p95"]
        qa_p95 = s["qa_latency_ms"]["p95"]
        mem_growth = s["memory"]["api_process"].get("growth_pct")
        res: dict[str, Any] = {
            "p95_ms": p95, "p95_target_ms": a.p95_target_ms,
            "p95_pass": bool(p95 <= a.p95_target_ms),
            "failed": s["failed"], "failed_zero_pass": bool(s["failed"] == 0),
            "rps": s["rps"],
            "memory_growth_pct": mem_growth,
            "memory_growth_target_pct": a.memory_growth_pct,
            "memory_growth_pass": (None if mem_growth is None else bool(mem_growth <= a.memory_growth_pct)),
            "memory_slope_mb_per_min": s["memory"]["api_process"].get("slope_mb_per_min"),
            "memory_trend": s["memory"]["api_process"].get("trend"),
        }
        if a.scenario == "B":
            res.update({"qa_p95_ms": qa_p95, "qa_p95_target_ms": a.qa_p95_target_ms,
                        "qa_p95_pass": bool(qa_p95 <= a.qa_p95_target_ms),
                        "service_stable": bool(s["http_5xx"] == 0 and s["connection_errors"] == 0)})
        return res

    def _print(self, s: dict[str, Any], out: Path) -> None:
        lat = s["latency_ms"]
        print("\n===== 压测汇总 =====")
        print(f"  标签/场景      : {s['tag']} / {s['scenario']}")
        print(f"  总请求 / 成功  : {s['total']} / {s['success']}（失败 {s['failed']}，"
              f"限流429 {s['rate_limited']}，5xx {s['http_5xx']}，超时 {s['timeouts']}）")
        print(f"  实际时长       : {s['actual_duration_s']} s（目标 {s['duration_s_target']} s）")
        print(f"  吞吐 RPS       : {s['rps']}（成功 {s['success_rps']}）")
        print(f"  延迟 ms        : P50={lat['p50']}  P90={lat['p90']}  P95={lat['p95']}  "
              f"P99={lat['p99']}  max={lat['max']}  mean={lat['mean']}")
        if s["qa_latency_ms"]["n"]:
            print(f"  问答延迟 ms    : P50={s['qa_latency_ms']['p50']}  P95={s['qa_latency_ms']['p95']}  "
                  f"P99={s['qa_latency_ms']['p99']}")
        mem = s["memory"]["api_process"]
        if "growth_pct" in mem:
            print(f"  API 内存       : 首窗 {mem['first_window_mean_mb']} MB → 末窗 "
                  f"{mem['last_window_mean_mb']} MB（{mem['growth_pct']:+}%，"
                  f"斜率 {mem['slope_mb_per_min']} MB/min，{mem['trend']}）")
        acc = s["acceptance"]
        print(f"  验收           : P95 {acc['p95_ms']} ≤ {acc['p95_target_ms']} → "
              f"{'达标' if acc['p95_pass'] else '未达标'}"
              f" | 无失败 {'达标' if acc['failed_zero_pass'] else '未达标'}"
              + (f" | 内存增长 {acc['memory_growth_pct']}% ≤ {acc['memory_growth_target_pct']}% → "
                 f"{'达标' if acc['memory_growth_pass'] else '未达标'}"
                 if acc.get('memory_growth_pass') is not None else ""))
        if acc.get("qa_p95_pass") is not None:
            print(f"                   场景B 问答 P95 {acc['qa_p95_ms']} ≤ {acc['qa_p95_target_ms']} → "
                  f"{'达标' if acc['qa_p95_pass'] else '未达标'} | 服务稳定 "
                  f"{'是' if acc.get('service_stable') else '否'}")
        if s.get("circuit_breaker"):
            print(f"  ⚠ 熔断触发     : {s['circuit_breaker']}")
        print(f"  汇总落盘       : {out}")


def _load_questions(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"[错误] 问题集不存在：{p}")
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("question"):
            rows.append(obj)
    if not rows:
        raise SystemExit(f"[错误] 问题集为空：{p}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    ap = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 自研 HTTP 压测器（jmeter 替代）")
    ap.add_argument("--scenario", choices=["A", "B"], required=True,
                    help="A=高频问答；B=混合负载（5 问答 + 5 上传解析）")
    ap.add_argument("--concurrency", type=int, default=20,
                    help=f"并发用户数（硬上限 {MAX_CONCURRENCY_HARD}）")
    ap.add_argument("--duration", dest="duration_s", type=int, default=180, help="持续秒数")
    ap.add_argument("--think-ms", dest="think_ms", type=int, default=200, help="思考时间基准（毫秒）")
    ap.add_argument("--timeout", dest="timeout_s", type=float, default=120.0, help="单请求超时（秒）")
    ap.add_argument("--base-url", default="http://127.0.0.1:8200")
    ap.add_argument("--questions", default=str(here / "问题集.jsonl"))
    ap.add_argument("--pdf-dir", default=r"E:\gao6gongdan\工单7\研发\data\raw", help="场景B 上传语料目录")
    ap.add_argument("--upload-poll-s", type=float, default=90.0, help="任务轮询上限（秒）")
    ap.add_argument("--outdir", default=str(root / "测试" / "测试数据"))
    ap.add_argument("--tag", default="基准_优化前")
    ap.add_argument("--commit-limit-pct", type=float, default=80.0, help="熔断：提交内存占用百分比")
    ap.add_argument("--error-limit-pct", type=float, default=50.0, help="熔断：滚动错误率百分比")
    ap.add_argument("--min-available-mb", type=float, default=1500.0,
                    help="启动前置：可用物理内存下限（默认 1500；被测服务自身已占 ~2.4 GB）")
    ap.add_argument("--commit-start-max-pct", type=float, default=85.0,
                    help="启动前置：提交内存占用上限百分比（主判据）")
    ap.add_argument("--no-circuit-breaker", action="store_true")
    ap.add_argument("--unique-sessions", action="store_true",
                    help="多用户口径：每个问答请求都用新 session_id（压会话卷/落盘增长）")
    # 验收阈值（默认取 场景配置.json，可覆盖）
    cfg = {}
    cfg_path = here / "场景配置.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cfg = {}
    acc = cfg.get("验收阈值", {})
    ap.add_argument("--p95-target-ms", type=float, default=float(acc.get("场景A_P95毫秒", 3000)))
    ap.add_argument("--qa-p95-target-ms", type=float, default=float(acc.get("场景B_问答P95毫秒", 5000)))
    ap.add_argument("--memory-growth-pct", type=float, default=float(acc.get("内存增长百分比", 10)))
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.concurrency > MAX_CONCURRENCY_HARD:
        print(f"[错误] --concurrency {args.concurrency} 超过硬上限 {MAX_CONCURRENCY_HARD}"
              f"（共享约定 §5.5 资源纪律）")
        return 2
    if args.concurrency < 1:
        print("[错误] --concurrency 必须 ≥ 1")
        return 2
    runner = Runner(args)
    runner.preflight()
    summary = runner.run()
    if summary.get("circuit_breaker"):
        return 4
    return 0 if summary["acceptance"].get("failed_zero_pass", True) else 1


if __name__ == "__main__":
    sys.exit(main())
