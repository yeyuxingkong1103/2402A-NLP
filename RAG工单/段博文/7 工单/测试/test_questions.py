# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
"""
RAG 功能测试及评估脚本（10 题 + RAGAS 评估）

基于 ccf_competition.zip 中的 9 份年报 PDF 设计 10 道测试题，
使用 RAGAS 框架评估检索质量（faithfulness / context_precision / context_recall / answer_relevancy）。

测试题覆盖：文本事实、表格数据、多文档对比、推理计算。
"""

import json
import time
import requests
from typing import List, Dict

BASE = "http://127.0.0.1:8000"

# ==================== 10 道测试题 ====================
QUESTIONS = [
    {
        "id": 1,
        "question": "平安银行2019年年度报告中的董事长是谁？",
        "expected_answer": "谢永林",
        "keywords": ["谢永林"],
        "type": "文本事实",
        "source_doc": "平安银行2019年年报",
    },
    {
        "id": 2,
        "question": "招商银行2019年的净利润是多少亿元？",
        "expected_answer": "928.67亿元",
        "keywords": ["928", "净利润"],
        "type": "表格数据",
        "source_doc": "招商银行2019年年报",
    },
    {
        "id": 3,
        "question": "邮储银行2019年年报中提到的三大风险是什么？",
        "expected_answer": "信用风险、市场风险、流动性风险",
        "keywords": ["信用风险", "市场风险", "流动性风险"],
        "type": "文本事实",
        "source_doc": "邮储银行2019年年报",
    },
    {
        "id": 4,
        "question": "中信证券2020年的营业收入比2019年增长了多少？",
        "expected_answer": "营业收入543.83亿元，同比增长26.06%",
        "keywords": ["543", "26"],
        "type": "表格数据+计算",
        "source_doc": "中信证券2020年年报",
    },
    {
        "id": 5,
        "question": "中国人寿2020年的保费收入是多少？",
        "expected_answer": "保费收入6122.65亿元",
        "keywords": ["6122", "保费"],
        "type": "表格数据",
        "source_doc": "中国人寿2020年年报",
    },
    {
        "id": 6,
        "question": "平安银行2019年和招商银行2019年，哪家银行的净利润更高？高多少？",
        "expected_answer": "招商银行净利润928.67亿元，平安银行净利润281.95亿元，招商银行高646.72亿元",
        "keywords": ["招商银行", "928", "平安", "281"],
        "type": "多文档对比",
        "source_doc": "平安银行+招商银行",
    },
    {
        "id": 7,
        "question": "国泰君安2021年的法定代表人是谁？",
        "expected_answer": "贺青",
        "keywords": ["贺青"],
        "type": "文本事实",
        "source_doc": "国泰君安2021年年报",
    },
    {
        "id": 8,
        "question": "中国太保2021年年报中提到的数字化转型战略是什么？",
        "expected_answer": "中国太保推进数字化转型2.0战略",
        "keywords": ["数字化", "转型"],
        "type": "文本事实",
        "source_doc": "中国太保2021年年报",
    },
    {
        "id": 9,
        "question": "招商证券2021年的总资产是多少亿元？",
        "expected_answer": "总资产5597.94亿元",
        "keywords": ["5597", "总资产"],
        "type": "表格数据",
        "source_doc": "招商证券2021年年报",
    },
    {
        "id": 10,
        "question": "中国平安2019年年报中，保险资金投资组合规模是多少？",
        "expected_answer": "保险资金投资组合规模3.21万亿元",
        "keywords": ["3.21", "保险资金"],
        "type": "表格数据",
        "source_doc": "中国平安2019年年报",
    },
]


def run_test(strategy: str = "hybrid", fusion: str = "rrf", reranker: str = "cross_encoder"):
    """运行 10 题测试，返回检索结果和评估数据。"""
    results = []

    print(f"\n{'='*60}")
    print(f"RAG 功能测试及评估（{strategy}/{fusion}/{reranker}）")
    print(f"{'='*60}")

    for q in QUESTIONS:
        print(f"\n[Q{q['id']}] {q['question'][:50]}...")

        # 检索
        t0 = time.time()
        r = requests.get(
            f"{BASE}/api/search",
            params={"query": q["question"], "top_k": 5, "strategy": strategy, "fusion": fusion, "reranker": reranker},
            timeout=60,
        )
        search_data = r.json()
        t_search = time.time() - t0

        contexts = [r["page_content"] for r in search_data.get("results", [])]
        retrieval_ok = any(k in " ".join(contexts) for k in q["keywords"])

        # 问答
        t0 = time.time()
        r = requests.post(
            f"{BASE}/api/chat",
            json={"query": q["question"], "top_k": 3, "stream": False, "use_rewrite": False,
                  "strategy": strategy, "fusion": fusion, "reranker": reranker},
            timeout=120,
        )
        chat_data = r.json()
        t_chat = time.time() - t0

        answer = chat_data.get("answer", "")
        answer_ok = any(k in answer for k in q["keywords"])

        result = {
            "id": q["id"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "answer": answer,
            "contexts": contexts,
            "retrieval_ok": retrieval_ok,
            "answer_ok": answer_ok,
            "search_time": round(t_search, 2),
            "chat_time": round(t_chat, 2),
            "type": q["type"],
        }
        results.append(result)

        print(f"  检索: {'✓' if retrieval_ok else '✗'} | 答案: {'✓' if answer_ok else '✗'} | "
              f"检索{t_search:.1f}s 问答{t_chat:.1f}s")

    return results


def analyze_problems(results: List[Dict]) -> Dict:
    """分析检索结果存在的问题。"""
    problems = {
        "retrieval_failures": [],
        "answer_failures": [],
        "slow_queries": [],
        "type_stats": {},
    }

    for r in results:
        if not r["retrieval_ok"]:
            problems["retrieval_failures"].append({
                "id": r["id"],
                "question": r["question"],
                "reason": "检索结果未包含期望关键词",
            })
        if not r["answer_ok"]:
            problems["answer_failures"].append({
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"][:100],
                "reason": "LLM 答案未包含期望关键词",
            })
        if r["search_time"] > 5:
            problems["slow_queries"].append({
                "id": r["id"],
                "search_time": r["search_time"],
                "reason": "检索响应超过 5 秒",
            })

        t = r["type"]
        if t not in problems["type_stats"]:
            problems["type_stats"][t] = {"total": 0, "ok": 0}
        problems["type_stats"][t]["total"] += 1
        if r["retrieval_ok"] and r["answer_ok"]:
            problems["type_stats"][t]["ok"] += 1

    return problems


def ragas_evaluate(results: List[Dict]) -> Dict:
    """使用 RAGAS 框架评估（简化版：基于关键词匹配计算指标）。"""
    # 简化版 RAGAS 指标计算
    n = len(results)

    # Context Precision: 检索结果中包含答案关键词的比例
    context_precision = sum(1 for r in results if r["retrieval_ok"]) / n

    # Context Recall: 检索结果覆盖期望答案的程度（简化为关键词命中）
    context_recall = context_precision  # 简化：与 precision 相同

    # Faithfulness: 答案与检索内容的一致性（答案关键词在上下文中出现）
    faithfulness = sum(
        1 for r in results
        if r["answer_ok"] and any(k in " ".join(r["contexts"]) for k in [r["expected_answer"].split("，")[0][:10]])
    ) / n

    # Answer Relevancy: 答案与问题的相关性（简化为答案非空且含关键词）
    answer_relevancy = sum(1 for r in results if r["answer_ok"]) / n

    return {
        "context_precision": round(context_precision, 4),
        "context_recall": round(context_recall, 4),
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(answer_relevancy, 4),
        "overall_score": round((context_precision + faithfulness + answer_relevancy) / 3, 4),
    }


def main():
    # 运行测试
    results = run_test(strategy="hybrid", fusion="rrf", reranker="cross_encoder")

    # 问题分析
    problems = analyze_problems(results)

    # RAGAS 评估
    ragas_scores = ragas_evaluate(results)

    # 汇总
    n = len(results)
    retrieval_ok = sum(1 for r in results if r["retrieval_ok"])
    answer_ok = sum(1 for r in results if r["answer_ok"])
    avg_search = sum(r["search_time"] for r in results) / n
    avg_chat = sum(r["chat_time"] for r in results) / n

    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    print(f"总题数: {n}")
    print(f"检索成功率: {retrieval_ok}/{n} = {retrieval_ok/n*100:.1f}%")
    print(f"答案成功率: {answer_ok}/{n} = {answer_ok/n*100:.1f}%")
    print(f"平均检索时间: {avg_search:.2f}s")
    print(f"平均问答时间: {avg_chat:.2f}s")

    print(f"\nRAGAS 评估指标:")
    for k, v in ragas_scores.items():
        print(f"  {k}: {v}")

    print(f"\n问题分析:")
    print(f"  检索失败: {len(problems['retrieval_failures'])} 题")
    print(f"  答案失败: {len(problems['answer_failures'])} 题")
    print(f"  慢查询(>5s): {len(problems['slow_queries'])} 题")
    for t, s in problems["type_stats"].items():
        print(f"  {t}: {s['ok']}/{s['total']} 通过")

    # 保存结果
    output = {
        "config": {"strategy": "hybrid", "fusion": "rrf", "reranker": "cross_encoder"},
        "questions": QUESTIONS,
        "results": results,
        "problems": problems,
        "ragas_scores": ragas_scores,
        "summary": {
            "total": n,
            "retrieval_ok": retrieval_ok,
            "answer_ok": answer_ok,
            "retrieval_rate": round(retrieval_ok / n * 100, 1),
            "answer_rate": round(answer_ok / n * 100, 1),
            "avg_search_time": round(avg_search, 2),
            "avg_chat_time": round(avg_chat, 2),
        },
    }

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到 test_results.json")


if __name__ == "__main__":
    main()
