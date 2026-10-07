# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化 - 性能测试"""

import json
from rag_qa_system import RAGQASystem
from rag_profiler import get_profiler


QUESTIONS = [
    {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"},
    {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"},
    {"id": 3, "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁？"},
    {"id": 4, "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？"},
    {"id": 5, "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成？"},
    {"id": 6, "question": "武汉力源信息技术股份有限公司招股意向书中，IC市场增长率最快的是哪个行业？"},
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"},
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
]


def main():
    print("=" * 80)
    print("RAG 性能瓶颈分析")
    print("=" * 80)

    system = RAGQASystem(["./data/招股说明书1.pdf", "./data/招股说明书2.pdf"])
    profiler = get_profiler()

    print(f"\n正在跑 {len(QUESTIONS)} 个问题...")
    results = []
    for q in QUESTIONS:
        r = system.answer(q["question"], use_rag=True)
        results.append({"id": q["id"], "response_time": r["response_time"]})
        print(f"  [{q['id']}] {r['response_time']}s")

    print()
    profiler.print_summary()
    profiler.save("rag_profiler_before.json")

    with open("perf_before.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = sum(r["response_time"] for r in results)
    avg = total / len(results)
    print(f"\n📊 总计：{total:.2f}s，平均：{avg:.4f}s/题")


if __name__ == "__main__":
    main()
