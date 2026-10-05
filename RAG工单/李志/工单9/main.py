import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragkit import demo_documents, graph_expand, graph_from_documents, load_documents, save, search, tokens

questions = [("主营业务是什么", "电子元器件 代理分销 技术服务"), ("募集资金用途", "研发中心 营销网络 流动资金"),
             ("公司风险", "供应链 市场竞争 汇率 技术")]
parser = argparse.ArgumentParser(description="工单9：GraphRAG 优化与指标对比")
parser.add_argument("--docs", nargs="*"); parser.add_argument("--demo", action="store_true"); args = parser.parse_args()
docs = load_documents(args.docs) if args.docs else demo_documents(); graph = graph_from_documents(docs)
rows = []
ragas_rows = []
for query, expected in questions:
    expected_terms = set(tokens(expected)); base = search(query, docs, 5); optimized = graph_expand(query, docs, graph)
    metric = lambda results: len(expected_terms & set(tokens(" ".join(x["text"] for x in results)))) / max(1, len(expected_terms))
    relevant = lambda results: sum(bool(expected_terms & set(tokens(x["text"]))) for x in results) / max(1, len(results))
    optimized = [item for item in optimized if item["score"] >= 0.3][:3] or optimized[:1]
    rows.append({"query": query, "baseline_context_recall": round(metric(base), 3),
                 "optimized_context_recall": round(metric(optimized), 3),
                 "context_precision": round(relevant(optimized), 3)})
    ragas_rows.append({"user_input": query, "response": "根据检索内容回答：" + optimized[0]["text"],
                       "retrieved_contexts": [item["text"] for item in optimized], "reference": expected})
report = {"target": {"context_precision": 0.8, "context_recall": 0.9}, "results": rows,
          "passed": all(row["context_precision"] >= 0.8 and row["optimized_context_recall"] >= 0.9 for row in rows),
          "evaluator": "deterministic lexical evaluator",
          "note": "这是无外部 LLM 的确定性评估框架；如配置 RAGAS，可用其同名指标复核。"}
save(Path(__file__).parent / "outputs/graphrag_evaluation.json", report); print(json.dumps(report, ensure_ascii=False, indent=2))
save(Path(__file__).parent / "outputs/ragas_dataset.json", ragas_rows)
