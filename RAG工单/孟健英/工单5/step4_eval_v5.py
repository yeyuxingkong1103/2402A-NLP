# 工单编号：人工智能 NLP-RAG-Query 理解优化任务
import json, time
from step3_rag_v5 import ask_rag, DialogState

DIALOG = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "他参与的哪个工程荣获了国家科技进步一等奖？",
    "这个公司的法定代表人是谁？",
    "那武汉力源信息技术股份有限公司呢？",
    "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
]

if __name__ == "__main__":
    state = DialogState()
    results = []
    for i, q in enumerate(DIALOG):
        t0 = time.time()
        ans, ctx, rewritten = ask_rag(q, state)
        t1 = time.time()
        results.append({
            "turn": i + 1,
            "question": q,
            "rewritten": rewritten,
            "answer": ans,
            "contexts": [(m["source"], m["page"], m["type"]) for m, _ in ctx],
            "time": round(t1 - t0, 2)
        })
        print(f"[轮 {i+1}] {q}")
        print(f"  重写：{rewritten}")
        print(f"  回答：{ans[:150]}")
        print(f"  耗时：{results[-1]['time']}s")
        print("-" * 70)
    with open("eval_results_v5.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("done")