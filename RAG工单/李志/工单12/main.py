import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragkit import demo_documents, graph_expand, graph_from_documents, load_documents, save, search, tokens

parser = argparse.ArgumentParser(description="工单12：LightRAG 双层检索与增量更新")
parser.add_argument("--docs", nargs="*"); parser.add_argument("--query", default="销售部和主营业务有什么关系？")
parser.add_argument("--mode", choices=["rag", "lightrag", "both"], default="both"); parser.add_argument("--demo", action="store_true"); args = parser.parse_args()
docs = load_documents(args.docs) if args.docs else demo_documents(); output = Path(__file__).parent / "outputs"
old_graph_path = output / "graph.json"; started = time.perf_counter(); graph = graph_from_documents(docs)
old_nodes = 0
if old_graph_path.exists(): old_nodes = len(json.loads(old_graph_path.read_text(encoding="utf-8"))["nodes"])
save(old_graph_path, graph); update_ms = (time.perf_counter() - started) * 1000
plain = search(args.query, docs, 5); light = graph_expand(args.query, docs, graph)
query_terms = set(tokens(args.query))
precision = lambda rows: round(sum(bool(query_terms & set(tokens(row["text"]))) for row in rows) / max(1, len(rows)), 3)
report = {"query": args.query, "mode": args.mode, "incremental_update_ms": update_ms,
          "previous_nodes": old_nodes, "current_nodes": len(graph["nodes"]), "graph_edges": len(graph["edges"]),
          "rag": {"context_precision_proxy": precision(plain), "results": plain},
          "lightrag": {"context_precision_proxy": precision(light), "results": light}}
save(output / "comparison.json", report); print(json.dumps({k: v for k, v in report.items() if k not in ("rag", "lightrag")}, ensure_ascii=False, indent=2))
