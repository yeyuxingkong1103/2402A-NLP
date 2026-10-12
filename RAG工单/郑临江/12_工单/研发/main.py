# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-LightRAG优化
主程序：构建知识图谱 → RAG / LightRAG 检索 → 检索结果对比 →
RAGAS 指标对比 → 输出知识图谱与结果。
"""
import json
import statistics

import config
from pdf_parser import load_documents
from rag import RAG
from lightrag import LightRAG
from evaluation import evaluate


def main():
    docs = load_documents([config.PDF1, config.PDF2])
    print(f"[main] 加载文档块 {len(docs)} 个")

    # 知识图谱构建
    lr = LightRAG(docs)
    print("[main] 知识图谱统计:", lr.graph.stats())

    # 传统 RAG
    rag = RAG(docs)

    results = []
    for q in config.QUESTIONS:
        r_rag = rag.answer(q["question"])
        r_light = lr.query(q["question"])
        results.append({
            "id": q["id"],
            "question": q["question"],
            "rag_answer": r_rag["answer"],
            "lightrag_answer": r_light["answer"],
            "rag_metrics": evaluate(r_rag, q["question"]),
            "lightrag_metrics": evaluate(r_light, q["question"]),
        })
        print(f"[main] id={q['id']} RAG={results[-1]['rag_metrics']} "
              f"LightRAG={results[-1]['lightrag_metrics']}")

    # 输出知识图谱
    with open("knowledge_graph.json", "w", encoding="utf-8") as f:
        json.dump(lr.graph.export(), f, ensure_ascii=False, indent=2)

    # 输出检索结果对比
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # RAGAS 平均指标对比
    def avg(metric):
        r = statistics.mean(x["rag_metrics"][metric] for x in results)
        l = statistics.mean(x["lightrag_metrics"][metric] for x in results)
        return round(r, 3), round(l, 3)

    print("\n===== RAGAS 平均指标对比 (RAG vs LightRAG) =====")
    for m in ["faithfulness", "answer_relevance", "context_precision", "context_recall"]:
        r, l = avg(m)
        print(f"{m}: RAG={r}  LightRAG={l}")

    print("\n已生成 knowledge_graph.json、results.json")


if __name__ == "__main__":
    main()
