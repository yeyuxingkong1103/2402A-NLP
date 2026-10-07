# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化 - 检索细分计时"""

import time
import json
from rag_qa_system import RAGQASystem
from rag_profiler import get_profiler


QUESTIONS = [
    {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"},
    {"id": 5, "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成？"},
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
]


def main():
    print("=" * 80)
    print("检索阶段细分分析（5 题）")
    print("=" * 80)

    system = RAGQASystem(["./data/招股说明书1.pdf", "./data/招股说明书2.pdf"])
    retriever = system.retriever
    reranker = retriever.reranker if hasattr(retriever, "reranker") else None

    # 手动细分计时
    timings = {
        "vector_retrieval": [],
        "bm25_retrieval": [],
        "rrf_fusion": [],
        "reranker": [],
        "graph": [],
    }

    for q in QUESTIONS:
        question = q["question"]

        # 1. 向量检索
        t0 = time.time()
        vec_results = retriever.vector_retriever.retrieve(question, top_k=20)
        timings["vector_retrieval"].append(time.time() - t0)

        # 2. BM25
        t0 = time.time()
        bm25_results = retriever.fulltext_searcher.search(question, top_k=20)
        timings["bm25_retrieval"].append(time.time() - t0)

        # 3. Reranker
        t0 = time.time()
        if reranker:
            pairs = [[question, c["content"]] for c in vec_results[:10]]
            scores = reranker.model.predict(pairs)
        timings["reranker"].append(time.time() - t0)

        # 4. Graph
        t0 = time.time()
        graph_triples = system.graph_retrieve(question, top_k=5)
        timings["graph"].append(time.time() - t0)

        print(f"  [{q['id']}] done")

    # 汇总
    print()
    print("=" * 80)
    print("细分阶段耗时")
    print("=" * 80)
    print(f"{'阶段':<25} {'平均':<12} {'最小':<12} {'最大':<12}")
    print("-" * 80)
    for stage, times in timings.items():
        if times:
            print(f"{stage:<25} {sum(times)/len(times):<12.4f} "
                  f"{min(times):<12.4f} {max(times):<12.4f}")

    # 保存
    with open("perf_detail.json", "w", encoding="utf-8") as f:
        json.dump({k: {"avg": sum(v)/len(v), "min": min(v), "max": max(v)} 
                    for k, v in timings.items() if v}, f, ensure_ascii=False, indent=2)
    print("\n✅ 已保存：perf_detail.json")


if __name__ == "__main__":
    main()
