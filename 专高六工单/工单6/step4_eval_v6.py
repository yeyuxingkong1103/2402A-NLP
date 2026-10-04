# 工单编号：人工智能 NLP-RAG-混合检索任务
import json, time
from step3_rag_v6 import ask_rag, DialogState

DIALOG = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "他参与的哪个工程荣获了国家科技进步一等奖？",
    "这个公司的法定代表人是谁？",
    "那武汉力源信息技术股份有限公司呢？",
    "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
]

MODES = [
    ("vector", "bge"),
    ("fulltext", "bge"),
    ("hybrid", "bge"),
    ("hybrid", "tfidf"),
    ("hybrid", "llm"),
    ("hybrid", "adaptive"),
]

if __name__ == "__main__":
    all_results = {}
    for mode, reranker in MODES:
        print(f"\n===== 模式：{mode} / 重排：{reranker} =====")
        state = DialogState()
        results = []
        for i, q in enumerate(DIALOG):
            t0 = time.time()
            ans, ctx, rewritten = ask_rag(q, state, mode=mode, reranker=reranker)
            t1 = time.time()
            results.append({
                "turn": i+1, "question": q, "rewritten": rewritten,
                "answer": ans,
                "contexts": [(m["source"], m["page"], m["type"]) for m, _ in ctx],
                "time": round(t1-t0, 2)
            })
            print(f"  [轮{i+1}] {ans[:80]}... ({results[-1]['time']}s)")
        all_results[f"{mode}_{reranker}"] = results

    with open("eval_results_v6.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\ndone")