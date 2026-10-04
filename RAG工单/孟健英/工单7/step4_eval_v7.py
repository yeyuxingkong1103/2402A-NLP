# 工单编号：人工智能 NLP-RAG-功能测试及评估
import json, time
from step3_rag_v7 import ask_rag
from config_v7 import QUESTION_FILE

if __name__ == "__main__":
    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for q in questions:
        t0 = time.time()
        ans, ctx = ask_rag(q["question"])
        t1 = time.time()
        results.append({
            "id": q["id"],
            "question": q["question"],
            "answer": ans,
            "contexts": [
                {"source": m["source"], "page": m["page"], "type": m["type"], "text": c[:300]}
                for m, c in ctx
            ],
            "time": round(t1 - t0, 2)
        })
        print(f"[{q['id']}] {q['question']}")
        print(f"  A: {ans[:120]}")
        print(f"  T: {results[-1]['time']}s")
        print("-" * 70)

    with open("eval_results_v7.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("done")