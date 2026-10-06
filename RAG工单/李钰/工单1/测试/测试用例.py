# -*- coding: utf-8 -*-
"""
测试用例 - 工单验收问题集
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

说明:
    1. 包含工单中明确要求的 10 个验收问题
    2. 对每个问题运行 RAG 与纯 LLM 两种模式
    3. 检查响应时间是否 ≤ 3 秒
    4. 输出对比结果与统计信息
"""
import os
import sys
import json
import time
import logging
from typing import List, Dict

# 引入研发目录
_DEV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "研发")
sys.path.insert(0, _DEV_DIR)

import qa_engine
import config

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# 工单验收的 10 个问题
TEST_QUESTIONS: List[Dict] = [
    {"id": 260, "question": "报告期内, 武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少?"},
    {"id": 95,  "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准?"},
    {"id": 33,  "question": "报告期内, 武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少?"},
    {"id": 34,  "question": "根据武汉兴图新科电子股份有限公司招股意向书, 电子信息行业的上游涉及哪些企业?"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商?"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书, 电子信息行业的下游主要包括哪些行业?"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖?"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少?"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁?"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金?"},
]


def run_one(item: Dict) -> Dict:
    """运行单条用例"""
    q = item["question"]
    print(f"\n[Q{item['id']}] {q}")
    try:
        result = qa_engine.answer_question(q)
        print(f"  响应时间: {result['response_time']}s "
              f"(限 {config.MAX_RESPONSE_TIME}s) "
              f"{'OK' if result['within_time_limit'] else 'OVER'}")
        print(f"  检索片段数: {len(result['retrieval'])}")
        print(f"  RAG 答案: {result['rag_answer'][:120]}")
        print(f"  纯 LLM 答案: {result['llm_answer'][:120]}")
        return {
            "id": item["id"],
            "question": q,
            "rag_answer": result["rag_answer"],
            "llm_answer": result["llm_answer"],
            "response_time": result["response_time"],
            "within_time_limit": result["within_time_limit"],
            "retrieval_count": len(result["retrieval"]),
            "top_score": result["retrieval"][0]["score"] if result["retrieval"] else 0.0,
            "status": "ok",
        }
    except Exception as e:
        print(f"  [异常] {e}")
        return {
            "id": item["id"],
            "question": q,
            "status": "error",
            "error": str(e),
        }


def run_all(questions: List[Dict] = None) -> List[Dict]:
    """运行全部用例"""
    questions = questions or TEST_QUESTIONS
    results = []
    print(f"========== 共 {len(questions)} 个验收问题 ==========")
    for item in questions:
        results.append(run_one(item))
    print("\n========== 全部运行完毕 ==========")
    _summary(results)
    return results


def _summary(results: List[Dict]):
    """统计结果"""
    total = len(results)
    ok = [r for r in results if r.get("status") == "ok"]
    err = [r for r in results if r.get("status") == "error"]
    in_time = [r for r in ok if r.get("within_time_limit")]
    avg_time = sum(r["response_time"] for r in ok) / len(ok) if ok else 0
    max_time = max((r["response_time"] for r in ok), default=0)
    print(f"总数: {total} | 成功: {len(ok)} | 失败: {len(err)}")
    print(f"响应达标 (≤3s): {len(in_time)}/{len(ok)}")
    print(f"平均响应: {avg_time:.3f}s | 最大响应: {max_time:.3f}s")


def save_results(results: List[Dict], path: str = None):
    """保存测试结果到 JSON"""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "测试结果.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {path}")


if __name__ == "__main__":
    results = run_all()
    save_results(results)
