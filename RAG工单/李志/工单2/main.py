import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragkit import answer, demo_documents, load_documents, save, search, timed

parser = argparse.ArgumentParser(description="工单2：RAG 准确率优化")
parser.add_argument("--docs", nargs="*"); parser.add_argument("--query", default="募集资金用于哪些项目？")
parser.add_argument("--demo", action="store_true"); args = parser.parse_args()
docs = load_documents(args.docs) if args.docs else demo_documents()
baseline, baseline_ms = timed(lambda: search(args.query, docs, 3, "vector"))
optimized, optimized_ms = timed(lambda: search(args.query, docs, 5, "hybrid"))
report = {"query": args.query, "baseline": answer(args.query, baseline), "optimized": answer(args.query, optimized),
          "baseline_top_score": baseline[0]["score"] if baseline else 0,
          "optimized_top_score": optimized[0]["score"] if optimized else 0,
          "baseline_ms": baseline_ms, "optimized_ms": optimized_ms, "method": "BM25+向量融合、扩大候选集"}
save(Path(__file__).parent / "outputs/comparison.json", report); print(json.dumps(report, ensure_ascii=False, indent=2))
