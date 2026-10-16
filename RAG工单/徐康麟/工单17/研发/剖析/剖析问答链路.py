# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

问答链路进程内剖析（py-spy 的替代）：``研发/剖析/剖析问答链路.py``。

## 为什么是这个形态

本机 **未安装 py-spy**（``Get-Command py-spy`` 无结果，见 ``设计/环境事实.md`` §2），
且共享约定禁止裸 pip 安装。外部采样器不可用，就改用**进程内剖析**：

1. ``cProfile`` 全量剖析一次问答 → ``pstats.sort_stats('cumulative')`` 取 Top-N
   （注意：cProfile 会让耗时膨胀 2~5 倍，**只用来看相对占比，不做绝对性能结论**）；
2. ``urllib.request.urlopen`` **调用计数器**（monkeypatch）→ 直接回答
   「一次问答到底打了几次外部 HTTP、其中几次是多余的 ``/api/tags`` 探测」
   —— 这是工单17 F1/F2 两个修复点的**直接证据**；
3. 分段耗时（retrieve / generate / total）直接来自 ``Answer.latency_ms``（实测值，不估算）。

## 用法（**建议在服务停止时跑**，本进程自身约 2.4 GB，避免与服务叠加）

    pwsh -NoProfile -File run_py.ps1 研发/剖析/剖析问答链路.py --tag 优化前 --n 8
    pwsh -NoProfile -File run_py.ps1 研发/剖析/剖析问答链路.py --tag 优化后 --n 8

``PERF_ENABLED`` 由调用方在环境里设好（与 ``起服务.ps1`` 同一套变量），
脚本只读不写。
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]          # 工单17/
sys.path.insert(0, str(_ROOT / "研发"))

#: urlopen 调用计数（monkeypatch 用）
_CALLS: dict[str, int] = {}
_CALLS_LOCK = threading.Lock()
_ORIGINAL_URLOPEN = urllib.request.urlopen


def _counting_urlopen(url: Any, *args: Any, **kwargs: Any) -> Any:
    """包一层 urlopen：按 URL 路径分桶计数（**不改行为**，只统计）。"""
    target = getattr(url, "full_url", None) or str(url)
    key = target.split("?")[0]
    with _CALLS_LOCK:
        _CALLS[key] = _CALLS.get(key, 0) + 1
    return _ORIGINAL_URLOPEN(url, *args, **kwargs)


def _rss_mb() -> float:
    """本进程 RSS（私有内存口径见 研发/压测/load_runner.py）。"""
    try:
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
    except Exception:  # noqa: BLE001
        return -1.0


def _load_questions(path: Path, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("question"):
            rows.append(obj)
    base = rows or [{"id": f"Q{i}", "question": q} for i, q in enumerate([
        "平安银行2019年的营业收入是多少？",
        "招商银行2019年的营业收入和净利润分别是多少？",
        "平安银行2019年末的不良贷款率是多少？",
        "平安银行2019年末的拨备覆盖率是多少？",
        "招商银行2019年年度报告的董事长是谁？",
        "中信证券2020年年度报告中，公司负责人是谁？",
    ], 1)]
    out: list[dict[str, Any]] = []
    while len(out) < max(1, n):
        out.append(base[len(out) % len(base)])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 问答链路进程内剖析")
    ap.add_argument("--tag", default="优化前")
    ap.add_argument("--n", type=int, default=8, help="跑几次问答（**每次都用新问句**以避开缓存）")
    ap.add_argument("--profile", action="store_true", help="额外跑 cProfile（耗时膨胀 2~5×）")
    ap.add_argument("--outdir", default=str(_ROOT / "测试" / "测试数据"))
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # —— 关键：在导入 QAEngine **之前**打上 urlopen 计数器 ——
    urllib.request.urlopen = _counting_urlopen   # type: ignore[assignment]

    from app.core.config import get_config                      # noqa: E402
    from app.core.logging_conf import get_logger, setup_logging  # noqa: E402
    from app.core.qa_engine import QAEngine                      # noqa: E402

    perf_enabled = os.environ.get("PERF_ENABLED", "0")
    cfg = get_config(refresh=True)
    try:
        log = setup_logging(cfg, run_id=f"profile-{args.tag}")
    except Exception:  # noqa: BLE001
        log = get_logger("profile")

    engine = QAEngine(cfg=cfg, logger=log)
    print(f"[{args.tag}] PERF_ENABLED={perf_enabled}  warmup 开始 …")
    t0 = time.perf_counter()
    warm = engine.warmup(with_generate=False)
    print(f"[{args.tag}] warmup 完成：{round((time.perf_counter() - t0) * 1000, 1)} ms")

    with _CALLS_LOCK:
        _CALLS.clear()
    rss0 = _rss_mb()

    questions = _load_questions(_ROOT / "研发" / "压测" / "问题集.jsonl", args.n)
    results: list[dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        started = time.perf_counter()
        ans = engine.answer(q["question"], session_id=f"profile-{i}")
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        results.append({"i": i, "question": q["question"][:40], "total_ms": elapsed,
                        "refused": ans.refused, "cache_hit": bool(ans.cache.get("hit")),
                        "latency_ms": dict(ans.latency_ms or {})})
        print(f"  [{i}/{len(questions)}] {elapsed:8.1f} ms  refused={ans.refused}  "
              f"cache={ans.cache.get('hit')}")

    rss1 = _rss_mb()
    with _CALLS_LOCK:
        calls = dict(_CALLS)
    stats = engine.stats()

    # —— cProfile（可选）——
    profile_top: list[dict[str, Any]] = []
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        engine.answer(questions[0]["question"], session_id="profile-cprof", use_cache=False)
        pr.disable()
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(30)
        text = buf.getvalue()
        (outdir / f"cProfile_{args.tag}.txt").write_text(text, encoding="utf-8")
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 6 and parts[0].isdigit():
                profile_top.append({"ncalls": parts[0], "tottime": parts[1], "cumtime": parts[3],
                                    "func": " ".join(parts[5:])[:120]})
        print(f"  cProfile 已落盘：cProfile_{args.tag}.txt（Top {len(profile_top)} 行）")

    tags_count = sum(v for k, v in calls.items() if k.endswith("/api/tags"))
    embed_count = sum(v for k, v in calls.items() if k.endswith("/api/embed"))
    gen_count = sum(v for k, v in calls.items() if k.endswith("/api/generate"))
    total_calls = sum(calls.values())
    effective = [r["total_ms"] for r in results if not r["cache_hit"]] or [r["total_ms"] for r in results]

    report = {
        "work_order": WORK_ORDER, "tag": args.tag, "perf_enabled": perf_enabled,
        "n_requests": len(results),
        "warmup": warm,
        "urlopen_calls": {"total": total_calls, "by_url": calls,
                          "api_tags": tags_count, "api_embed": embed_count,
                          "api_generate": gen_count,
                          "per_request_total": round(total_calls / max(1, len(results)), 3),
                          "per_request_api_tags": round(tags_count / max(1, len(results)), 3)},
        "latency_ms": {
            "mean": round(statistics.fmean(effective), 2) if effective else 0.0,
            "min": round(min(effective), 2) if effective else 0.0,
            "max": round(max(effective), 2) if effective else 0.0,
            "non_cache_n": len([r for r in results if not r["cache_hit"]]),
        },
        "stage_ms": {
            "retrieve_mean": round(statistics.fmean([r["latency_ms"].get("retrieve", 0.0) for r in results]), 2),
            "generate_mean": round(statistics.fmean([r["latency_ms"].get("generate", 0.0) for r in results]), 2),
            "total_mean": round(statistics.fmean([r["latency_ms"].get("total", 0.0) for r in results]), 2),
        },
        "rss_mb": {"before": rss0, "after": rss1, "delta": round(rss1 - rss0, 1)},
        "engine_stats": stats,
        "results": results,
        "cprofile_top": profile_top[:20],
    }
    out = outdir / f"剖析问答链路_{args.tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n===== 剖析汇总 =====")
    print(f"  PERF_ENABLED          : {perf_enabled}")
    print(f"  urlopen 总次数        : {total_calls}（每请求 {report['urlopen_calls']['per_request_total']} 次）")
    print(f"    其中 /api/tags 探测 : {tags_count}（每请求 {report['urlopen_calls']['per_request_api_tags']} 次）")
    print(f"    /api/embed          : {embed_count}")
    print(f"    /api/generate       : {gen_count}")
    print(f"  非缓存请求延迟均值    : {report['latency_ms']['mean']} ms"
          f"（min {report['latency_ms']['min']} / max {report['latency_ms']['max']}）")
    print(f"  分段：retrieve {report['stage_ms']['retrieve_mean']} ms / "
          f"generate {report['stage_ms']['generate_mean']} ms / total {report['stage_ms']['total_mean']} ms")
    print(f"  RSS 变化              : {rss0} → {rss1} MB（{report['rss_mb']['delta']:+}）")
    print(f"  答案缓存              : entries={stats.get('memory_cache')} "
          f"max={stats.get('memory_cache_max')} ttl={stats.get('memory_cache_ttl_s')}")
    print(f"  落盘                  : {out}")
    print(f"  落盘统计              : {stats.get('storage')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
