# -*- coding: utf-8 -*-
"""分阶段耗时汇总（baseline vs optimized，取 repeat=0 首查）。"""
import json
from pathlib import Path

d = json.loads((Path(__file__).parent / "results" / "benchmark_results.json")
               .read_text(encoding="utf-8"))
recs = [r for r in d["records"] if r["repeat"] == 0]
for mode in ("baseline", "optimized"):
    ms = [r for r in recs if r["mode"] == mode]
    keys = sorted({k for r in ms for k in r["timings"] if k != "t_total"})
    parts = {k: sum(r["timings"].get(k, 0) for r in ms) / len(ms) for k in keys}
    total = sum(v for k, v in parts.items() if k.startswith("t_"))
    ncand = sum(r["n_candidates"] for r in ms) / len(ms)
    print(f"== {mode} == 平均候选数 {ncand:.1f}  总耗时 {total:.3f}s")
    for k, v in sorted(parts.items(), key=lambda x: -x[1]):
        print(f"  {k:<20} {v:7.3f}s  ({v/total*100:5.1f}%)")
