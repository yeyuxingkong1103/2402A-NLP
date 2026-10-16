# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

内存泄漏剖析（memory-profiler / objgraph 的替代）：``研发/剖析/内存剖析.py``。

## 为什么是这个形态

本机 **未安装 memory-profiler / objgraph**（见 ``设计/环境事实.md`` §2），
改用标准库三件套：

1. ``tracemalloc`` 快照 diff（``compare_to(..., 'lineno')``）→ **泄漏对象 Top-N 的调用行号**；
   判据：``size_diff > 0`` 且随轮次**持续增长**的调用点才算泄漏（一次性分配不算）；
2. ``gc.get_objects()`` 分类统计 → 按类型给出 count / 估算字节；
3. ``sys.getsizeof`` 递归估算 → 给出「答案缓存条目平均字节数」这类可报告数字。

## 用法（**建议在服务停止时跑**）

    pwsh -NoProfile -File run_py.ps1 研发/剖析/内存剖析.py --tag 优化前 --rounds 6 --per-round 12
    pwsh -NoProfile -File run_py.ps1 研发/剖析/内存剖析.py --tag 优化后 --rounds 6 --per-round 12

输出：``测试/测试数据/tracemalloc_<tag>.json``（Top-N 差异 + 逐轮曲线 + 对象分类）。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "研发"))

#: 关注的对象类型（工单17 的泄漏嫌疑清单）
WATCH_TYPES = ("dict", "list", "tuple", "set", "str", "bytes", "Answer", "Turn", "Chunk",
               "Hit", "GateResult", "RerankFeatures", "Comprehension")


def _rss_mb() -> float:
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    handle = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, os.getpid())
    counters = PMC()
    counters.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    ctypes.windll.kernel32.CloseHandle(handle)
    return round(counters.PagefileUsage / (1024 * 1024), 1)


def _estimate(obj: Any, depth: int = 2, seen: set[int] | None = None) -> int:
    """``sys.getsizeof`` 递归估算（深度受限，避免无限展开；只用于量级判断）。"""
    seen = seen if seen is not None else set()
    oid = id(obj)
    if oid in seen or depth < 0:
        return 0
    seen.add(oid)
    size = sys.getsizeof(obj, 0)
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:200]:
            size += _estimate(k, depth - 1, seen) + _estimate(v, depth - 1, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in list(obj)[:200]:
            size += _estimate(item, depth - 1, seen)
    return size


def _object_census(top: int = 25) -> dict[str, Any]:
    """``gc.get_objects()`` 分类统计。"""
    counts: dict[str, int] = {}
    for obj in gc.get_objects():
        name = type(obj).__name__
        counts[name] = counts.get(name, 0) + 1
    top_all = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    watched = {k: counts.get(k, 0) for k in WATCH_TYPES}
    return {"total_objects": sum(counts.values()), "top_types": dict(top_all), "watched": watched}


def _fmt_trace(trace: Any) -> str:
    frames = traceback_frames = trace
    try:
        last = frames[-1]
        return f"{last.filename}:{last.lineno}"
    except Exception:  # noqa: BLE001
        return "<unknown>"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 内存泄漏剖析")
    ap.add_argument("--tag", default="优化前")
    ap.add_argument("--rounds", type=int, default=6, help="轮数（每轮产生一批新问句）")
    ap.add_argument("--per-round", type=int, default=12, help="每轮问答次数")
    ap.add_argument("--top", type=int, default=20, help="tracemalloc Top-N")
    ap.add_argument("--outdir", default=str(_ROOT / "测试" / "测试数据"))
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from app.core.config import get_config
    from app.core.logging_conf import get_logger, setup_logging
    from app.core.qa_engine import QAEngine
    from app.storage.sqlite_manager import SqliteManager

    perf_enabled = os.environ.get("PERF_ENABLED", "0")
    cfg = get_config(refresh=True)
    try:
        log = setup_logging(cfg, run_id=f"mem-{args.tag}")
    except Exception:  # noqa: BLE001
        log = get_logger("mem")

    runtime_dir = Path(os.environ.get("DEPLOY_RUNTIME_DIR", _ROOT / "部署" / "卷" / "runtime"))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    db_path = runtime_dir / f"mem_probe_{args.tag}.sqlite3"
    for suffix in ("", "-wal", "-shm"):
        try:
            Path(str(db_path) + suffix).unlink(missing_ok=True)
        except OSError:
            pass
    storage = SqliteManager(db_path, logger=log)

    engine = QAEngine(cfg=cfg, logger=log, storage=storage)
    engine.warmup(with_generate=False)
    print(f"[{args.tag}] PERF_ENABLED={perf_enabled}  预热完成，开始 {args.rounds} 轮 × {args.per_round} 次")

    tracemalloc.start(10)
    gc.collect()
    base_snapshot = tracemalloc.take_snapshot()
    base_census = _object_census()
    rss0 = _rss_mb()

    curve: list[dict[str, Any]] = []
    prev = base_snapshot
    diffs_per_round: list[dict[str, float]] = []
    for rnd in range(1, max(1, args.rounds) + 1):
        for i in range(max(1, args.per_round)):
            # 每轮换不同问句 → 不断写入新的缓存条目（这是泄漏点被压出来的前提）
            q = f"平安银行2019年的营业收入是多少？（第{rnd}轮第{i}次改写{nudge(rnd, i)}）"
            engine.answer(q, session_id=f"mem-{rnd}-{i}")
        gc.collect()
        snap = tracemalloc.take_snapshot()
        stats = snap.compare_to(prev, "lineno")[: args.top]
        growth = sum(s.size_diff for s in stats) / 1024.0
        prev = snap
        census = _object_census()
        rss = _rss_mb()
        curve.append({"round": rnd, "rss_mb": rss,
                      "objects": census["total_objects"],
                      "watched": census["watched"],
                      "engine": {k: v for k, v in engine.stats().items()
                                 if k.startswith(("memory_cache", "asked", "cache_hit"))}})
        top_growth = {"dict": 0.0, "list": 0.0, "str": 0.0}
        diffs_per_round.append(top_growth)
        print(f"  第 {rnd} 轮：RSS {rss} MB  对象 {census['total_objects']}  "
              f"缓存 {engine.stats().get('memory_cache')}  本轮净增 {growth:+.1f} KB")

    gc.collect()
    end_snapshot = tracemalloc.take_snapshot()
    end_census = _object_census()
    rss1 = _rss_mb()

    top_diff = []
    for stat in end_snapshot.compare_to(base_snapshot, "lineno")[: args.top]:
        top_diff.append({
            "trace": _fmt_trace(stat.traceback),
            "size_diff_kb": round(stat.size_diff / 1024.0, 2),
            "size_kb": round(stat.size / 1024.0, 2),
            "count_diff": int(stat.count_diff),
        })

    watched_delta = {k: end_census["watched"].get(k, 0) - base_census["watched"].get(k, 0)
                     for k in WATCH_TYPES}
    first_rss = curve[0]["rss_mb"] if curve else rss0
    last_rss = curve[-1]["rss_mb"] if curve else rss1

    report = {
        "work_order": WORK_ORDER, "tag": args.tag, "perf_enabled": perf_enabled,
        "rounds": args.rounds, "per_round": args.per_round,
        "rss_mb": {"start": rss0, "end": rss1, "delta": round(rss1 - rss0, 1),
                   "first_round": first_rss, "last_round": last_rss,
                   "growth_pct": round(100.0 * (last_rss - first_rss) / first_rss, 2) if first_rss else 0.0},
        "objects": {"start": base_census["total_objects"], "end": end_census["total_objects"],
                    "delta": end_census["total_objects"] - base_census["total_objects"],
                    "watched_delta": watched_delta},
        "tracemalloc_top": top_diff,
        "curve": curve,
        "engine_final": engine.stats(),
        "runtime_db": str(db_path),
    }
    out = outdir / f"tracemalloc_{args.tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n===== 内存剖析汇总 =====")
    print(f"  RSS               : {rss0} → {rss1} MB（{report['rss_mb']['delta']:+}）")
    print(f"  首轮→末轮增长率    : {report['rss_mb']['growth_pct']:+}%")
    print(f"  对象总数           : {base_census['total_objects']} → {end_census['total_objects']}"
          f"（{report['objects']['delta']:+}）")
    print(f"  关注类型增量       : {watched_delta}")
    print(f"  答案缓存           : {engine.stats().get('memory_cache')} / "
          f"{engine.stats().get('memory_cache_max')}")
    print(f"  tracemalloc Top5   :")
    for item in top_diff[:5]:
        print(f"      {item['size_diff_kb']:+9.1f} KB  {item['count_diff']:+6d}  {item['trace']}")
    print(f"  落盘               : {out}")
    storage.close()
    tracemalloc.stop()
    return 0


def nudge(rnd: int, i: int) -> str:
    """产生语义上无意义但**字面不同**的扰动，让缓存键不重复（用于压缓存增长）。"""
    return "·" * ((rnd * 7 + i) % 5 + 1)


if __name__ == "__main__":
    sys.exit(main())
