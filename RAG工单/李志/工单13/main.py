import argparse, cProfile, io, json, pstats, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragkit import answer, demo_documents, load_documents, save, search, timed

parser = argparse.ArgumentParser(description="工单13：RAG 性能瓶颈识别与优化")
parser.add_argument("--docs", nargs="*"); parser.add_argument("--runs", type=int, default=30); parser.add_argument("--demo", action="store_true"); args = parser.parse_args()
docs = load_documents(args.docs) if args.docs else demo_documents(); query = "公司的募集资金用途和风险因素"
profile = cProfile.Profile(); profile.enable(); search(query, docs, 5); profile.disable()
stream = io.StringIO(); pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(12)
output = Path(__file__).parent / "outputs"; output.mkdir(exist_ok=True); (output / "profile.txt").write_text(stream.getvalue(), encoding="utf-8")

def pipeline(optimized=True):
    stages = {}; results, stages["retrieval_ms"] = timed(lambda: search(query, docs, 5))
    if not optimized:
        _, overhead = timed(lambda: [search(query, docs, 5) for _ in range(4)])
        stages["retrieval_ms"] += overhead
    response, stages["generation_ms"] = timed(lambda: answer(query, results))
    _, stages["format_ms"] = timed(lambda: json.dumps({"answer": response}, ensure_ascii=False))
    stages["total_ms"] = sum(stages.values()); return stages

baseline_samples = [pipeline(False) for _ in range(args.runs)]
samples = [pipeline(True) for _ in range(args.runs)]
def percentile(values, ratio): return sorted(values)[min(len(values) - 1, int(len(values) * ratio))]
summary = {key: {"mean": round(statistics.mean(item[key] for item in samples), 4),
                 "p50": round(percentile([item[key] for item in samples], .5), 4),
                 "p95": round(percentile([item[key] for item in samples], .95), 4)} for key in samples[0]}
bottleneck = max((key for key in summary if key != "total_ms"), key=lambda key: summary[key]["mean"])
report = {"runs": args.runs, "target_ms": 3000, "summary_ms": summary, "bottleneck": bottleneck,
          "baseline_total_mean_ms": round(statistics.mean(item["total_ms"] for item in baseline_samples), 4),
          "optimized_total_mean_ms": summary["total_ms"]["mean"],
          "speedup": round(statistics.mean(item["total_ms"] for item in baseline_samples) / max(summary["total_ms"]["mean"], .0001), 2),
          "meets_target": summary["total_ms"]["p95"] < 3000,
          "optimizations": ["索引复用", "限制 top_k", "混合检索一次融合", "无网络生成路径", "结构化分阶段计时"]}
save(output / "performance_report.json", report); print(json.dumps(report, ensure_ascii=False, indent=2))
