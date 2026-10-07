import time
import json
from rag_qa_system import RAGQASystem
from optimized_retriever import OptimizedRetriever

QUESTIONS = [
    {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"},
    {"id": 5, "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成？"},
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
]

def main():
    print("=" * 80)
    print("优化后性能测试（5 题）")
    print("=" * 80)

    system = RAGQASystem(["./data/招股说明书1.pdf", "./data/招股说明书2.pdf"])
    opt = OptimizedRetriever(system.retriever)

    timings = {"vector": [], "bm25": [], "rrf": [], "rerank": [], "total": []}

    for q in QUESTIONS:
        t0 = time.time()
        results = opt.retrieve(q["question"], top_k=3)
        total = time.time() - t0
        timings["total"].append(total)
        if results and "_timing" in results[0]:
            t = results[0]["_timing"]
            for k in ["vector", "bm25", "rrf", "rerank"]:
                timings[k].append(t[k])
        print(f"  [{q['id']}] {total:.4f}s")

    print()
    print("=" * 80)
    print("优化后细分耗时")
    print("=" * 80)
    print(f"{'阶段':<20} {'平均':<12} {'最小':<12} {'最大':<12}")
    print("-" * 80)
    for stage, times in timings.items():
        if times:
            print(f"{stage:<20} {sum(times)/len(times):<12.4f} "
                  f"{min(times):<12.4f} {max(times):<12.4f}")

    with open("perf_after.json", "w", encoding="utf-8") as f:
        json.dump({k: {"avg": sum(v)/len(v)} for k, v in timings.items() if v}, 
                  f, ensure_ascii=False, indent=2)
    print("\\n✅ 已保存：perf_after.json")

if __name__ == "__main__":
    main()
