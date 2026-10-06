# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
import json, time
from step3_rag_v2 import ask_rag, ask_llm

QUESTIONS = [
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"},
]

if __name__ == "__main__":
    results = []
    for q in QUESTIONS:
        t0 = time.time()
        rag_ans, ctx = ask_rag(q["question"])
        t1 = time.time()
        llm_ans = ask_llm(q["question"])
        results.append({
            "id": q["id"],
            "question": q["question"],
            "rag_answer_v2": rag_ans,
            "llm_answer": llm_ans,
            "rag_context_pages": [c[0] for c in ctx],
            "time": round(t1 - t0, 2)
        })
        print(q["id"], "time:", results[-1]["time"])
    with open("eval_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("done")