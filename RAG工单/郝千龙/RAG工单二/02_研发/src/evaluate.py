# -*- coding: utf-8 -*-
# 【批量评测脚本 · evaluate.py】对工单指定的10个问题做优化后回归，输出准确率与逐题明细
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

"""评测命令行（对应验收：准确率≥90%、响应≤3秒）：

    python evaluate.py --index ../index_store --out ../03_测试/评测结果.csv

标准答案来自招股书原文事实（页码可溯源），评测分两级：
- hit@1：Top-1 证据块是否命中答案关键事实（人工复核列）；
- 答案正确率：抽取/生成答案是否包含标准答案关键数字/实体。
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from embeddings import create_embedder
from qa_engine import QAEngine
from vector_store import IndexStore

# 工单二指定的 10 个问题及招股书原文标准答案（关键事实用于自动判分）
QUESTIONS = [
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
     "gold": ["6,464.51", "14,414.16", "18,780.67", "4,627.14"]},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
     "gold": ["视频指挥系统技术标准", "技术规范"]},
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
     "gold": ["82.10%", "97.31%", "94.84%", "94.34%"]},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
     "gold": ["电子元器件", "金属壳体"]},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
     "gold": ["视频指挥"]},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
     "gold": ["军队", "政府", "能源"]},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
     "gold": ["情报、指挥、控制与通信网络一体化工程", "C4ISR"]},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？",
     "gold": ["5,520"]},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？",
     "gold": ["程家明"]},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
     "gold": ["15,000.00", "1.5亿"]},
]


def judge(answer: str, gold: list) -> bool:
    """判分：答案命中任一标准答案关键事实即视为正确。

    :param answer: 系统答案
    :param gold: 标准答案关键事实列表（OR 关系）
    :return: 是否正确
    """
    if not answer:
        return False
    if len(gold) >= 3:  # 多数字题：要求命中绝大多数关键数字
        hit = sum(1 for g in gold if g in answer)
        return hit >= len(gold) - 1
    return all(g in answer for g in gold[:1]) if gold else False


def main() -> None:
    """加载索引并对 10 个问题批量问答，落盘 CSV 明细。"""
    parser = argparse.ArgumentParser(description="工单二十问评测脚本")
    parser.add_argument("--index", default=CONFIG.index_dir)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(CONFIG.base_dir), "03_测试", "评测结果.csv"))
    args = parser.parse_args()

    embedder = create_embedder()
    store = IndexStore.load(args.index, embedder)
    engine = QAEngine(__import__("retriever").Retriever(store))

    rows, correct, within_sla = [], 0, 0
    for item in QUESTIONS:
        result = engine.answer(item["question"])
        ok = judge(result.answer, item["gold"])
        sla = result.latency_s <= CONFIG.response_timeout_s
        correct += int(ok)
        within_sla += int(sla)
        top1 = result.evidences[0] if result.evidences else None
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "answer": re.sub(r"\s+", " ", result.answer)[:300],
            "gold": "；".join(item["gold"]),
            "is_correct": ok,
            "top1_page": top1.page_no if top1 else "",
            "top1_score": round(top1.score, 4) if top1 else "",
            "dense_rank": top1.dense_rank if top1 else "",
            "sparse_rank": top1.sparse_rank if top1 else "",
            "latency_s": round(result.latency_s, 3),
            "within_3s": sla,
            "mode": result.mode,
        })
        print(f"[{'✓' if ok else '✗'}] id={item['id']} "
              f"{result.latency_s:.2f}s  {result.answer[:60]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(QUESTIONS)
    print("\n========== 评测汇总 ==========")
    print(f"答案正确率：{correct}/{total} = {correct / total:.0%}（验收线 90%）")
    print(f"响应≤3s 占比：{within_sla}/{total} = {within_sla / total:.0%}")
    print(f"明细已写出：{args.out}")


if __name__ == "__main__":
    main()
