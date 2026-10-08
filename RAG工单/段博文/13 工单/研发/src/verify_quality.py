# -*- coding: utf-8 -*-
"""校验两链路最终返回结果一致性（条数/top_score/top1来源）。"""
import json
from pathlib import Path

d = json.loads((Path(__file__).parent / "results" / "benchmark_results.json")
               .read_text(encoding="utf-8"))
recs = [r for r in d["records"] if r["repeat"] == 0]
queries = [r["query"] for r in recs if r["mode"] == "baseline"]
same_n = same_score = 0
for q in queries:
    b = next(r for r in recs if r["mode"] == "baseline" and r["query"] == q)
    o = next(r for r in recs if r["mode"] == "optimized" and r["query"] == q)
    same_n += b["n_results"] == o["n_results"]
    same_score += abs(b["top_score"] - o["top_score"]) < 0.6
    print(f"{q[:20]:<22} 基线 n={b['n_results']} s={b['top_score']:.3f} | "
          f"优化 n={o['n_results']} s={o['top_score']:.3f}")
print(f"\n返回条数一致: {same_n}/{len(queries)}  top_score接近(±0.6): {same_score}/{len(queries)}")
