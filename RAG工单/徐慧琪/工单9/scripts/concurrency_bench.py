# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：并发压测（工单硬性要求2「资源消耗：合理利用计算资源，确保在高并发环境下稳定运行」）

## 本机并发能力的前提（决定了测什么、不测什么）

本机是**单卡笔记本 GPU（RTX 4060 Laptop）+ 本地嵌入式 Qdrant**，不是服务端集群：

  · 嵌入 / 精排模型独占显存，多个请求**串行排队**而非并行计算；
  · Qdrant 为嵌入式单实例，同一存储目录**排他锁**，进程内共享一个客户端；
  · LLM 走本机 Ollama，同样是单实例串行。

因此本压测**不测「吞吐量随并发线性增长」**（在这套硬件上不可能），
而是测三件真实且可交付的事：

  1. **并发下是否仍然成功**（可靠性）——这是工单「稳定运行」的核心诉求；
  2. **延迟如何随并发退化**（排队代价），给出可承载的并发档位；
  3. **资源是否被合理利用**（内存/显存是否失控）。

这些都如实报告，不夸大成「支持高并发」。

用法：
    python scripts/concurrency_bench.py --levels 1,2,4,8 --per-level 4
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w9 import compat  # noqa: E402

QUESTIONS = [
    "平安银行2019年年度报告的董事长致辞中，提到其盈利增长的关键因素有哪些？",
    "招商银行在其2019年年报中提到的创新商业模式有哪些？",
    "中国邮政储蓄银行2019年末的不良贷款率和拨备覆盖率分别是多少？",
    "中国平安2019年的营业收入和归属于母公司股东的净利润分别是多少？",
]


def _one(question: str, mode: str) -> dict:
    """单次请求（含端到端耗时与是否降级）。"""
    from w9.qa_v2 import ask_v2
    t0 = time.perf_counter()
    try:
        r = ask_v2(question, mode=mode)
        return {"ok": bool(r.answer), "elapsed": time.perf_counter() - t0,
                "degraded": bool(r.degraded), "len": len(r.answer or ""),
                "error": ""}
    except Exception as exc:                           # noqa: BLE001
        return {"ok": False, "elapsed": time.perf_counter() - t0,
                "degraded": True, "len": 0,
                "error": f"{type(exc).__name__}: {exc}"}


def run_level(level: int, per_level: int, mode: str) -> dict:
    """在给定并发度下跑一批请求。"""
    from w9.runtime9 import memory_status
    qs = [QUESTIONS[i % len(QUESTIONS)] for i in range(per_level * level)]
    mem_before = memory_status().get("rss_mb")
    t0 = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=level) as ex:
        futs = [ex.submit(_one, q, mode) for q in qs]
        for f in as_completed(futs):
            results.append(f.result())
    wall = time.perf_counter() - t0
    mem_after = memory_status().get("rss_mb")

    lat = [r["elapsed"] for r in results]
    ok = [r for r in results if r["ok"]]
    return {
        "level": level, "n": len(qs), "mode": mode,
        "ok": len(ok), "failed": len(results) - len(ok),
        "degraded": sum(1 for r in results if r["degraded"]),
        "success_rate": round(len(ok) / max(len(results), 1), 4),
        "wall_s": round(wall, 2),
        "throughput_qps": round(len(results) / wall, 3) if wall else None,
        "lat_mean": round(statistics.mean(lat), 2) if lat else None,
        "lat_p50": round(statistics.median(lat), 2) if lat else None,
        "lat_max": round(max(lat), 2) if lat else None,
        "mem_rss_before_mb": mem_before, "mem_rss_after_mb": mem_after,
        "errors": sorted({r["error"] for r in results if r["error"]})[:3],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="工单09 并发压测")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--per-level", type=int, default=4,
                    help="每档并发的请求总数 = level × per_level")
    ap.add_argument("--mode", default="fast", choices=["full", "fast"])
    ap.add_argument("--out", default=os.path.join(
        compat.wo9_data(), "eval", "concurrency.json"))
    a = ap.parse_args()

    print("预热模型（不计入压测）…", flush=True)
    from src import rag
    rag.warmup()
    from w9.qa_v2 import warmup as w9_warmup
    w9_warmup()

    rows: list[dict] = []
    for lv in [int(x) for x in a.levels.split(",")]:
        print(f"\n=== 并发 {lv} ===", flush=True)
        r = run_level(lv, a.per_level, a.mode)
        rows.append(r)
        print(f"  成功 {r['ok']}/{r['n']}  成功率 {r['success_rate']:.1%}  "
              f"墙钟 {r['wall_s']}s  QPS {r['throughput_qps']}", flush=True)
        print(f"  延迟 均值 {r['lat_mean']}s  P50 {r['lat_p50']}s  "
              f"最大 {r['lat_max']}s", flush=True)
        if r["errors"]:
            print(f"  错误：{r['errors']}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"mode": a.mode, "levels": rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n已写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
