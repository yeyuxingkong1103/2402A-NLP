# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
模块：10 个问题 Graph RAG 测试
功能：跑 10 个问题，对比工单7 结果
"""

import json
import time
from rag_qa_system import RAGQASystem

# 与工单7 相同的 10 个问题
QUESTIONS = [
    {"id": 1, "question": "平安银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？", "doc": "2019年_平安银行"},
    {"id": 2, "question": "招商银行在其2019年年报中提到的创新商业模式有哪些？", "doc": "2019年_招商银行"},
    {"id": 3, "question": "平安银行的风险管理体系如何体现其应对宏观经济周期波动的能力？", "doc": "2019年_平安银行"},
    {"id": 4, "question": "分析这些银行和保险公司在面对经济周期波动时的共同策略与差异化策略。", "doc": "多个年报"},
    {"id": 5, "question": "平安银行2019年营业收入是多少？", "doc": "2019年_平安银行"},
    {"id": 6, "question": "中国平安2019年归属于母公司股东的净利润是多少？", "doc": "2019年_中国平安"},
    {"id": 7, "question": "招商银行2019年不良贷款率是多少？", "doc": "2019年_招商银行"},
    {"id": 8, "question": "中信证券2020年营业收入是多少？", "doc": "2020年_中信证券"},
    {"id": 9, "question": "中国人寿2020年保费收入是多少？", "doc": "2020年_中国人寿"},
    {"id": 10, "question": "国泰君安2021年营业收入是多少？", "doc": "2021年_国泰君安"},
]

if __name__ == "__main__":
    # 只加载 2 个招股书（graph_test.json 的图谱来自它们）
    system = RAGQASystem([
        "./data/招股说明书1.pdf",
        "./data/招股说明书2.pdf",
    ])

    results = []
    print("\n" + "=" * 70)
    print("工单8：10 个问题 Graph RAG 测试")
    print("=" * 70)

    for q in QUESTIONS:
        r = system.answer(q["question"], use_rag=True)
        graph_triples = r.get("graph_triples", [])
        print(f"\n【ID:{q['id']}】{q['question']}")
        print(f"  答案：{r['answer'][:200]}...")
        print(f"  图谱三元组：{len(graph_triples)} 条")
        print(f"  响应：{r['response_time']}s")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "answer": r["answer"],
            "graph_triples_count": len(graph_triples),
            "graph_triples": graph_triples[:5],
            "response_time": r["response_time"],
        })

    with open("工单8结果.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n✅ 已保存到 工单8结果.json")
