import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragkit import answer, demo_documents, load_documents, save, search, timed

parser = argparse.ArgumentParser(description="工单1：PDF RAG 问答")
parser.add_argument("--docs", nargs="*"); parser.add_argument("--query", default="公司的主营业务是什么？")
parser.add_argument("--demo", action="store_true"); args = parser.parse_args()
docs = load_documents(args.docs) if args.docs else demo_documents()
results, elapsed = timed(lambda: search(args.query, docs, 4))
payload = {"question": args.query, "answer": answer(args.query, results), "sources": results, "elapsed_ms": elapsed}
save(Path(__file__).parent / "outputs/result.json", payload)
print(payload["answer"]); print(f"检索耗时: {elapsed:.2f} ms")
