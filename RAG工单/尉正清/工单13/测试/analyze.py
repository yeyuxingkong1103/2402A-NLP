# 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
"""读 perf_log.jsonl，把阶段耗时汇总成表

`api.py` 每处理一个请求就往 `测试/results/perf_log.jsonl` 追加一行，
本脚本负责把它变成报告里的数字。

用法：
    D:/Anaconda/envs/rag_gd/python.exe analyze.py [--tag before] [--log 路径]
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "results" / "perf_log.jsonl"


def pct(vals, p):
    vals = sorted(vals)
    if not vals:
        return 0.0
    k = max(0, min(len(vals) - 1, int(round((p / 100) * len(vals) + 0.5)) - 1))
    return vals[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--tag", default="",
                    help="只看 request_id 以该前缀开头的记录（如 bench-before-）")
    ap.add_argument("--depth", type=int, default=0, help="看第几层的阶段")
    args = ap.parse_args()

    recs = []
    for line in Path(args.log).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if args.tag and not str(r.get("request_id", "")).startswith(args.tag):
            continue
        if r.get("status") != "ok":
            continue
        recs.append(r)

    if not recs:
        print("没有符合条件的记录")
        return 1

    total = [r["total_seconds"] for r in recs]
    print(f"样本 {len(recs)} 条"
          f"{f'（request_id 前缀 {args.tag}）' if args.tag else ''}\n")
    print(f"  端到端  均值 {statistics.mean(total):.3f}s  "
          f"中位 {statistics.median(total):.3f}s  "
          f"P95 {pct(total, 95):.3f}s  最大 {max(total):.3f}s")

    # 按阶段名汇总（可指定看哪一层：0=主阶段，1=子阶段）
    by_stage = defaultdict(list)
    for r in recs:
        for s in r["spans"]:
            if s["depth"] == args.depth:
                by_stage[s["stage"]].append(s["seconds"])

    print(f"\n  {'阶段':<20}{'均值':>9}{'中位':>9}{'P95':>9}{'占比':>8}")
    mean_total = statistics.mean(total)
    for name in sorted(by_stage):
        v = by_stage[name]
        m = statistics.mean(v)
        print(f"  {name:<20}{m:>8.3f}s{statistics.median(v):>8.3f}s"
              f"{pct(v, 95):>8.3f}s{m / mean_total * 100:>7.1f}%")

    # 未归入任何顶层阶段的时间 —— 埋点有没有漏，看这个数
    unacc = [r.get("unaccounted_seconds", 0.0) for r in recs]
    print(f"\n  未归入阶段的时间：均值 {statistics.mean(unacc):.3f}s"
          f"（{statistics.mean(unacc) / mean_total * 100:.1f}%）")

    # 指标分布：召回数、上下文长度、回答长度
    keys = set()
    for r in recs:
        keys |= set(r.get("metrics", {}))
    if keys:
        print("\n  请求指标（均值）:")
        import statistics as st
        for k in sorted(keys):
            vals = [r["metrics"][k] for r in recs
                    if k in r.get("metrics") and isinstance(r["metrics"][k], (int, float))]
            if vals:
                print(f"    {k:<22}{st.mean(vals):>10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
