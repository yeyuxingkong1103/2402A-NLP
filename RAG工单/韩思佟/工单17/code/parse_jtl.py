# -*- coding: utf-8 -*-
"""
工单17 —— JMeter 结果(.jtl) 自动汇总脚本

用法：
  python parse_jtl.py result_scenarioA.jtl result_scenarioB.jtl --tag 优化前
输出：
  产出物/压测汇总_<tag>.md / .json（含平均值、P50/P90/P95/P99、吞吐、错误率）
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def percentile(vals, p):
    if not vals:
        return 0.0
    vals = sorted(vals)
    k = max(0, min(len(vals) - 1, int(round(len(vals) * p)) - 1))
    return vals[k]


def parse_jtl(path: Path):
    rows = []
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("elapsed"):
                continue
            rows.append({
                "label": r.get("label", ""),
                "elapsed": float(r["elapsed"]) / 1000.0,
                "success": (r.get("success", "true").lower() == "true"),
                "ts": int(r.get("timeStamp", "0") or 0),
            })
    return rows


def summarize(rows):
    if not rows:
        return {}
    lat = [r["elapsed"] for r in rows]
    ok = [r for r in rows if r["success"]]
    span = (max(r["ts"] for r in rows) - min(r["ts"] for r in rows)) / 1000.0 or 1.0
    return {
        "总请求数": len(rows),
        "成功数": len(ok),
        "错误率": round(1 - len(ok) / len(rows), 4),
        "平均响应s": round(st.mean(lat), 3),
        "P50响应s": round(percentile(lat, 0.50), 3),
        "P90响应s": round(percentile(lat, 0.90), 3),
        "P95响应s": round(percentile(lat, 0.95), 3),
        "P99响应s": round(percentile(lat, 0.99), 3),
        "最大响应s": round(max(lat), 3),
        "吞吐(req/s)": round(len(rows) / span, 2),
        "测试时长s": round(span, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jtls", nargs="+")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    outdir = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "产出物"
    outdir.mkdir(parents=True, exist_ok=True)

    result = {}
    for j in args.jtls:
        p = Path(j)
        if not p.exists():
            print(f"[跳过] {p} 不存在")
            continue
        rows = parse_jtl(p)
        result[p.stem] = summarize(rows)
        # 按 label 细分
        by_label = defaultdict(list)
        for r in rows:
            by_label[r["label"]].append(r)
        result[p.stem + "_分接口"] = {k: summarize(v) for k, v in by_label.items()}

    (outdir / f"压测汇总_{args.tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# 压测汇总（{args.tag}）", ""]
    for scen, s in result.items():
        if scen.endswith("_分接口") or not s:
            continue
        lines += [f"## {scen}", "",
                  "| 指标 | 值 |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in s.items()]
        lines.append("")
    lines += ["## 验收判定", "",
              "- 场景A（20并发问答）：P95 ≤ 3s 且错误率 0 → "
              f"{'通过 ✅' if result.get('result_scenarioA', {}).get('P95响应s', 99) <= 3 and result.get('result_scenarioA', {}).get('错误率', 1) == 0 else '未通过 ❌'}",
              "- 场景B（10并发混合）：P95 ≤ 5s → "
              f"{'通过 ✅' if result.get('result_scenarioB', {}).get('P95响应s', 99) <= 5 else '未通过 ❌'}", ""]
    (outdir / f"压测汇总_{args.tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
