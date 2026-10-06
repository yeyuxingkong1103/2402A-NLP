import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragkit import demo_documents, load_documents, save, search

parser = argparse.ArgumentParser(description="工单6：BM25/向量/混合检索")
parser.add_argument("--docs", nargs="*"); parser.add_argument("--query", default="研发中心和技术风险")
parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid")
parser.add_argument("--demo", action="store_true"); args = parser.parse_args()
docs = load_documents(args.docs) if args.docs else demo_documents()
payload = {"mode": args.mode, "query": args.query, "results": search(args.query, docs, 5, args.mode)}
save(Path(__file__).parent / "outputs/retrieval.json", payload); print(json.dumps(payload, ensure_ascii=False, indent=2))
