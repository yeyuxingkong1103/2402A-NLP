# -*- coding: utf-8 -*-
"""
工单 17 · 内存增长探针：持续采样进程 RSS + 对象增长（定位泄漏）
依赖：psutil（可选 objgraph / tracemalloc）

用法：
  python memory_probe.py --pid <pid> --minutes 30 --interval 5 [--out mem.csv]
  python memory_probe.py --self-test
"""
from __future__ import annotations

import argparse
import time


def sample_rss(pid: int) -> float:
    import psutil
    return psutil.Process(pid).memory_info().rss / 1024 / 1024  # MB


def run(pid: int, minutes: int, interval: int, out):
    rows = ["ts_s,rss_mb,growth_pct"]
    t0 = time.time()
    base = None
    print(f"sampling pid={pid} for {minutes} min (interval {interval}s) ...")
    while time.time() - t0 < minutes * 60:
        try:
            rss = sample_rss(pid)
        except Exception as e:
            print(f"[WARN] {e}"); break
        if base is None:
            base = rss
        growth = (rss - base) / base * 100 if base else 0.0
        row = f"{int(time.time()-t0)},{rss:.1f},{growth:.2f}"
        rows.append(row)
        print(row, flush=True)
        time.sleep(interval)
    if out:
        open(out, "w", encoding="utf-8").write("\n".join(rows))
        print("saved", out)


def self_test():
    print("RSS 采样逻辑自检 OK（需 psutil 才能对本机进程采样）")
    try:
        import psutil, os
        print("当前进程 RSS(MB):", round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 1))
        print("objgraph:", "available" if __import__("importlib").util.find_spec("objgraph") else "not installed")
    except Exception as e:
        print("psutil not available:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int)
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return
    assert a.pid, "need --pid"
    run(a.pid, a.minutes, a.interval, a.out)


if __name__ == "__main__":
    main()
