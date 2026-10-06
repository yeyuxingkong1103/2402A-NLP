# -*- coding: utf-8 -*-
# 【多轮评测脚本 · evaluate.py】对工单指定的5组多轮对话做回归，每轮+整组双判分
# 工单编号：人工智能NLP-RAG-Query理解优化任务

"""评测命令行（对应验收：整组多轮准确率≥90%、每轮≤3秒）：

    python evaluate.py --index ../index_store --out ../../03_测试/评测结果.csv

评测分两级：
- 每轮判定：改写后检索答案是否命中标准答案关键事实；
- 整组判定：一组内所有轮全部正确才算整组通过；
- 总体准确率 = 正确轮数 / 总轮数（验收线 90%）。
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from dialogue_manager import DialogueManager, DialogueState
from qa_engine import QAEngine
from retriever import Retriever
from vector_store import IndexStore

# 5 组多轮对话测试题（每组覆盖指代词、省略、实体切换等）
# gold：标准答案关键事实列表（OR 关系，命中任一即算该事实点命中）
DIALOGUE_SETS = [
    {
        "set_id": 1,
        "name": "工单指定5轮（兴图→奖项→法代→力源法代→组织结构图）",
        "turns": [
            {"q": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
             "gold": ["6,464.51", "14,414.16", "18,780.67", "4,627.14"]},
            {"q": "他参与的哪个工程荣获了国家科技进步一等奖？",
             "gold": ["情报、指挥、控制与通信网络一体化工程", "C4ISR"]},
            {"q": "这个公司的法定代表人是谁？",
             "gold": ["程家明"]},
            {"q": "那武汉力源信息技术股份有限公司呢？",
             "gold": ["赵马克"]},
            {"q": "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
             "gold": ["大客户销售部", "珠海销售处", "深圳销售处", "北京销售处", "武汉销售处", "广州销售处", "成都销售处"]},
        ],
    },
    {
        "set_id": 2,
        "name": "注册资本跨公司切换",
        "turns": [
            {"q": "武汉兴图新科电子股份有限公司的注册资本是多少？",
             "gold": ["5,520"]},
            {"q": "那武汉力源信息技术股份有限公司呢？",
             "gold": ["5,000"]},
            {"q": "武汉力源信息技术股份有限公司的英文名称是什么？",
             "gold": ["Wuhan P&S Information Technology"]},
            {"q": "那这个公司的法定代表人是谁？",
             "gold": ["赵马克"]},
            {"q": "武汉兴图新科电子股份有限公司的英文名是什么？",
             "gold": ["Wuhan Xingtu Xinke Electronics"]},
        ],
    },
    {
        "set_id": 3,
        "name": "技术标准→领域→组织结构图",
        "turns": [
            {"q": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
             "gold": ["视频指挥系统技术标准", "视频技术规范"]},
            {"q": "这个公司在哪个领域已经成为重要供应商？",
             "gold": ["视频指挥"]},
            {"q": "武汉力源信息技术股份有限公司有哪些销售处？",
             "gold": ["珠海销售处", "深圳销售处", "北京销售处", "武汉销售处", "广州销售处", "成都销售处"]},
            {"q": "这个公司的法定代表人是谁？",
             "gold": ["赵马克"]},
        ],
    },
    {
        "set_id": 4,
        "name": "军用占比→法代→力源注册资本",
        "turns": [
            {"q": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
             "gold": ["82.10%", "97.31%", "94.84%", "94.34%"]},
            {"q": "这个公司的法定代表人是谁？",
             "gold": ["程家明"]},
            {"q": "那武汉力源信息技术股份有限公司的注册资本呢？",
             "gold": ["5,000"]},
            {"q": "武汉力源信息技术股份有限公司组织结构图中销售处最多的销售部是哪个？",
             "gold": ["大客户销售部"]},
        ],
    },
    {
        "set_id": 5,
        "name": "法代→力源法代→销售处数量→兴图注册资本",
        "turns": [
            {"q": "武汉兴图新科电子股份有限公司法定代表人是谁？",
             "gold": ["程家明"]},
            {"q": "那武汉力源信息技术股份有限公司呢？",
             "gold": ["赵马克"]},
            {"q": "这个公司一共有多少个销售处？",
             "gold": ["6"]},
            {"q": "那武汉兴图新科电子股份有限公司的注册资本呢？",
             "gold": ["5,520"]},
            {"q": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
             "gold": ["情报、指挥、控制与通信网络一体化工程", "C4ISR"]},
        ],
    },
]


def judge(answer: str, gold: list) -> bool:
    """判分：答案命中 gold 中绝大多数关键事实即视为正确。

    :param answer: 系统答案
    :param gold: 标准答案关键事实列表
    :return: 是否正确
    """
    if not answer:
        return False
    if len(gold) >= 4:  # 多事实题：要求命中绝大多数（允许漏1个）
        hit = sum(1 for g in gold if g in answer)
        return hit >= len(gold) - 1
    # 少事实题：命中任一即可
    return any(g in answer for g in gold)


def main() -> None:
    parser = argparse.ArgumentParser(description="工单五多轮对话评测脚本")
    parser.add_argument("--index", default=CONFIG.index_dir)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(CONFIG.base_dir), "03_测试", "评测结果.csv"))
    args = parser.parse_args()

    store = IndexStore.load(args.index)
    engine = QAEngine(Retriever(store))
    dm = DialogueManager()

    rows = []
    total_turns = 0
    correct_turns = 0
    within_sla_turns = 0
    correct_sets = 0
    coref_success = 0  # 指代消解成功（有指代且本轮正确）

    for ds in DIALOGUE_SETS:
        state = DialogueState()
        set_all_correct = True
        for idx, turn in enumerate(ds["turns"], start=1):
            parsed = dm.process_turn(state, turn["q"])
            result = engine.answer(turn["q"], parsed.rewritten)
            dm.record_answer(state, parsed, result.answer, result.latency_s)

            ok = judge(result.answer, turn["gold"])
            sla = result.latency_s <= CONFIG.response_timeout_s
            total_turns += 1
            correct_turns += int(ok)
            within_sla_turns += int(sla)
            if not ok:
                set_all_correct = False
            if parsed.has_coreference and ok:
                coref_success += 1

            rows.append({
                "组号": ds["set_id"],
                "组名": ds["name"],
                "轮次": idx,
                "原始问题": turn["q"],
                "改写后查询": parsed.rewritten,
                "指代消解": "是" if parsed.has_coreference else "否",
                "省略补全": "是" if parsed.has_ellipsis else "否",
                "当前实体": parsed.entity,
                "系统答案": result.answer[:300],
                "标准答案": "；".join(turn["gold"]),
                "是否正确": "是" if ok else "否",
                "耗时(秒)": round(result.latency_s, 3),
                "≤3秒": "是" if sla else "否",
                "Top1页码": result.evidences[0].page_no if result.evidences else "",
            })
            flag = "✓" if ok else "✗"
            print(f"[组{ds['set_id']} 轮{idx}] {flag} {result.latency_s:.2f}s "
                  f"| 改写: {parsed.rewritten[:50]}")
        if set_all_correct:
            correct_sets += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    acc = correct_turns / total_turns
    sla_rate = within_sla_turns / total_turns
    set_pass = correct_sets / len(DIALOGUE_SETS)
    print("\n========== 评测汇总 ==========")
    print(f"总轮数：{total_turns}")
    print(f"正确轮数：{correct_turns}")
    print(f"总体准确率：{correct_turns}/{total_turns} = {acc:.0%}（验收线 90%）")
    print(f"≤3秒占比：{within_sla_turns}/{total_turns} = {sla_rate:.0%}")
    print(f"整组通过：{correct_sets}/{len(DIALOGUE_SETS)} = {set_pass:.0%}")
    print(f"指代消解成功轮数：{coref_success}")
    print(f"明细已写出：{args.out}")


if __name__ == "__main__":
    main()
