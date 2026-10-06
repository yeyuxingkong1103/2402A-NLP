# -*- coding: utf-8 -*-
# 【批量评测脚本 · evaluate.py】对工单四16个图文问题批量回归，输出准确率/耗时/降级明细
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化

"""评测命令行（在02_研发/src目录下执行）：

    python evaluate.py

标准答案取自招股书原文与图表实测（页码可溯源），判分两级：
- 多数字/枚举题要求命中绝大多数关键事实；
- 同义实体题（any）命中任一标准答案即可。
输出03_测试/评测结果.csv（utf-8-sig，Excel可直接打开）。
"""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from multimodal_index import create_embedder, IndexStore
from qa_engine import QAEngine
from retriever import Retriever

# 工单PDF两处问题列表的并集中的16个问题（顺序按PDF首次出现顺序）
QUESTIONS = [
    {"id": 5, "company": "力源信息", "type": "矢量组织结构图",
     "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？",
     "gold": ["4个部门", "6个销售处"], "match": "all"},
    {"id": 6, "company": "力源信息", "type": "位图增长率图表",
     "question": "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？",
     "gold": ["汽车", "14.0", "IC卡", "-2.0"], "match": "most"},
    {"id": 1, "company": "力源信息", "type": "文本数字",
     "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
     "gold": ["1,670", "25.04"], "match": "all"},
    {"id": 2, "company": "力源信息", "type": "表格枚举",
     "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？",
     "gold": ["仓储及物流中心", "研发中心", "电子商务平台",
              "扩充产品种类和数量", "其他与主营业务相关的营运资金"],
     "match": "most"},
    {"id": 3, "company": "力源信息", "type": "关联方表格",
     "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？",
     "gold": ["赵马克", "42.35", "控股股东"], "match": "most"},
    {"id": 4, "company": "力源信息", "type": "关联方表格枚举",
     "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
     "gold": ["融冰投资", "武汉博润", "上海博润", "听音投资",
              "联众聚源", "力源贸易", "普芯达"], "match": "most"},
    {"id": 260, "company": "兴图新科", "type": "多数字枚举",
     "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
     "gold": ["6,464.51", "14,414.16", "18,780.67", "4,627.14"],
     "match": "most"},
    {"id": 95, "company": "兴图新科", "type": "实体",
     "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
     "gold": ["视频指挥系统技术标准", "技术规范"], "match": "any"},
    {"id": 33, "company": "兴图新科", "type": "多数字枚举",
     "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
     "gold": ["82.10%", "97.31%", "94.84%", "94.34%"], "match": "most"},
    {"id": 34, "company": "兴图新科", "type": "列表",
     "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
     "gold": ["电子元器件", "金属壳体"], "match": "all"},
    {"id": 957, "company": "兴图新科", "type": "实体",
     "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
     "gold": ["视频指挥"], "match": "any"},
    {"id": 793, "company": "兴图新科", "type": "列表",
     "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
     "gold": ["军队", "政府", "能源"], "match": "most"},
    {"id": 795, "company": "兴图新科", "type": "实体",
     "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
     "gold": ["情报、指挥、控制与通信网络一体化工程", "C4ISR"], "match": "any"},
    {"id": 543, "company": "兴图新科", "type": "槽位数字",
     "question": "武汉兴图新科电子股份有限公司注册资本是多少？",
     "gold": ["5,520"], "match": "any"},
    {"id": 531, "company": "兴图新科", "type": "槽位人名",
     "question": "武汉兴图新科电子股份有限公司法定代表人是谁？",
     "gold": ["程家明"], "match": "any"},
    {"id": 207, "company": "兴图新科", "type": "槽位数字",
     "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
     "gold": ["15,000.00", "15,000", "1.5亿"], "match": "any"},
]


def judge(answer: str, gold: list, match: str) -> bool:
    """判分：按match策略校验答案是否覆盖标准答案关键事实。

    :param answer: 系统答案
    :param gold: 标准答案关键事实
    :param match: all=全部命中 / most=命中除1个外全部 / any=命中任一
    :return: 正确返回True
    """
    if not answer:
        return False
    hits = [g for g in gold if g in answer]
    if match == "all":
        return len(hits) == len(gold)
    if match == "most":
        return len(hits) >= len(gold) - 1
    return bool(hits)


def main() -> None:
    """加载索引批量问答，落盘CSV并打印汇总。"""
    embedder = create_embedder()
    store = IndexStore.load(CONFIG.index_dir, embedder)
    engine = QAEngine(Retriever(store))

    rows, correct, within_sla, degraded_n = [], 0, 0, 0
    for item in QUESTIONS:
        result = engine.answer(item["question"])
        ok = judge(result.answer, item["gold"], item["match"])
        sla = result.latency_s <= CONFIG.response_timeout_s
        correct += int(ok)
        within_sla += int(sla)
        degraded_n += int(result.degraded)
        top1 = result.evidences[0] if result.evidences else None
        rows.append({
            "id": item["id"],
            "company": item["company"],
            "question_type": item["type"],
            "question": item["question"],
            "answer": re.sub(r"\s+", " ", result.answer)[:400],
            "gold": "；".join(item["gold"]),
            "is_correct": int(ok),
            "top1_evidence_type": top1.chunk_type if top1 else "",
            "top1_doc": top1.doc if top1 else "",
            "top1_page": top1.page_no if top1 else "",
            "top1_score": round(top1.score, 4) if top1 else "",
            "latency_s": round(result.latency_s, 3),
            "within_3s": int(sla),
            "degraded": int(result.degraded),
        })
        print(f"[{'正确' if ok else '错误'}] id={item['id']:<4} "
              f"{result.latency_s:0.2f}s  {result.answer[:70]}")

    out_dir = os.path.join(CONFIG.root_dir, "03_测试")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "评测结果.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(QUESTIONS)
    print("\n========== 评测汇总 ==========")
    print(f"答案正确率：{correct}/{total} = {correct / total:.1%}（验收线90%）")
    print(f"响应≤3s占比：{within_sla}/{total} = {within_sla / total:.0%}")
    print(f"降级答案数：{degraded_n}/{total}")
    print(f"明细已写出：{out_path}")


if __name__ == "__main__":
    main()
