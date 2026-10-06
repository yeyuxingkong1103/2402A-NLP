# -*- coding: utf-8 -*-
# 【评测入口 · evaluate.py】14道标准问题离线评测：判分、页码、耗时统计并输出评测结果CSV
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

"""离线评测：14 道标准问题（兴图新科 10 题 + 力源信息 4 题），输出
《03_测试/评测结果.csv》，字段含问题/答案/gold/是否正确/页码/耗时。

加 --baseline 时加载无表格基线索引（定长滑窗、关闭公司路由，
表格精排加权因无表格块自然失效；答案槽位沿用工单二同层逻辑，
以隔离“表格解析与分块”单一变量），
输出《03_测试/评测结果_基线.csv》，供 04 优化报告做真实前后对比。

判分规则（与工单二一致）：
- gold 项 ≥3 个的枚举/多数字题：答案命中至少 len(gold)-1 个判正确；
- 其余题：命中 gold 第 1 项即判正确；
- 比对前统一去除空白与标点（数字保留小数点与百分号）。
"""
import argparse
import csv
import os
import re
import sys
import time

from config import CONFIG
from vector_store import IndexStore
from retriever import Retriever
from qa_engine import QAEngine

# (编号, 公司, 问题, gold答案项)
QUESTIONS = [
    ("id1", "力源信息",
     "力源信息本次发行股数是多少？占发行后总股本的比例是多少？",
     ["1,670", "25.04"]),
    ("id2", "力源信息",
     "力源信息本次募集资金拟投资哪些项目？",
     ["仓储及物流中心", "研发中心", "电子商务平台", "扩充产品种类和数量",
      "其他与主营业务相关的营运资金"]),
    ("id3", "力源信息",
     "力源信息与公司存在控制关系的关联方是谁？持股比例及与本公司的关系如何？",
     ["赵马克", "42.35", "控股股东"]),
    ("id4", "力源信息",
     "力源信息与公司不存在控制关系的关联方企业有哪些？",
     ["融冰投资", "武汉博润", "上海博润", "听音投资", "联众聚源",
      "力源贸易", "普芯达"]),
    ("id260", "兴图新科",
     "兴图新科来自军用领域的收入分别是多少？",
     ["6,464.51", "14,414.16", "18,780.67", "4,627.14"]),
    ("id95", "兴图新科",
     "兴图新科参与制定了什么样的技术标准？",
     ["视频指挥系统技术标准"]),
    ("id33", "兴图新科",
     "兴图新科军用产品收入占主营业务收入的比重分别是多少？",
     ["82.10", "97.31", "94.84", "94.34"]),
    ("id34", "兴图新科",
     "兴图新科所处电子信息行业的上游涉及哪些企业？",
     ["电子元器件", "金属壳体"]),
    ("id957", "兴图新科",
     "兴图新科目前已经成为哪个领域的重要供应商？",
     ["视频指挥"]),
    ("id793", "兴图新科",
     "兴图新科所处电子信息行业下游主要包括哪些行业？",
     ["军队", "政府", "能源"]),
    ("id795", "兴图新科",
     "兴图新科参与的哪个工程获得国家科技进步一等奖？",
     ["情报、指挥、控制与通信网络一体化工程"]),
    ("id543", "兴图新科",
     "兴图新科的注册资本是多少？",
     ["5,520"]),
    ("id531", "兴图新科",
     "兴图新科的法定代表人是谁？",
     ["程家明"]),
    ("id207", "兴图新科",
     "兴图新科募集资金中用于补充流动资金的金额是多少？",
     ["15,000.00"]),
]

# 比对前剔除的标点（保留小数点、百分号、数字、字母与汉字）
_PUNCT_RE = re.compile(
    r"[\s，,。、；;：:？?！!（）()【】\[\]“”\"'’‘《》<>·\-—_~`@#$^&*]+")


def normalize(text: str) -> str:
    """归一化文本：去空白与标点，便于子串命中。

    :param text: 原文
    :return: 归一化字符串
    """
    return _PUNCT_RE.sub("", text or "").lower()


def judge(gold, answer: str):
    """按统一判分规则判分。

    :param gold: gold 答案项列表
    :param answer: 系统答案
    :return: (是否正确, 命中项数, 缺失项列表)
    """
    norm_answer = normalize(answer)
    hits = [g for g in gold if normalize(g) in norm_answer]
    need = len(gold) - 1 if len(gold) >= 3 else 1
    ok = len(hits) >= need
    return ok, len(hits), [g for g in gold if g not in hits]


def run(baseline: bool = False) -> None:
    """加载索引并跑完 14 题，输出 CSV 与汇总。

    :param baseline: 是否评测基线链路
    """
    index_dir = CONFIG.index_dir + ("_baseline" if baseline else "")
    if not os.path.isdir(index_dir):
        print(f"[评测][错误] 索引不存在：{index_dir}，请先运行 "
              f"python build_index.py{' --baseline' if baseline else ''}")
        sys.exit(1)

    print(f"[评测] 加载{'基线' if baseline else '优化'}索引：{index_dir}")
    t0 = time.perf_counter()
    store = IndexStore.load(index_dir)
    retriever = Retriever(store)
    # 基线只关闭公司路由（无表格行/整表块，表格精排加权自然不生效）；
    # 答案抽取槽位沿用工单二同层逻辑，以隔离“表格解析与分块”单一变量
    engine = QAEngine(retriever, use_routing=not baseline, use_slots=True)
    print(f"[评测] 索引加载耗时 {time.perf_counter() - t0:.1f} 秒，"
          f"共 {len(store.chunks)} 块")

    rows = []
    correct = 0
    within_3s = 0
    latencies = []
    for qid, company, question, gold in QUESTIONS:
        result = engine.answer(question)
        ok, hit_n, missing = judge(gold, result.answer)
        correct += int(ok)
        within_3s += int(result.latency_s <= CONFIG.response_timeout_s)
        latencies.append(result.latency_s)
        pages = "、".join(str(e.page_no) for e in result.evidences[:3])
        rows.append({
            "编号": qid,
            "公司": company,
            "问题": question,
            "答案": result.answer,
            "gold": "；".join(gold),
            "是否正确": "是" if ok else "否",
            "命中项数": f"{hit_n}/{len(gold)}",
            "缺失项": "；".join(missing),
            "页码": pages,
            "耗时(秒)": f"{result.latency_s:.3f}",
        })
        flag = "✓" if ok else "✗"
        print(f"  {flag} {qid:>5} {result.latency_s:5.2f}s "
              f"命中{hit_n}/{len(gold)}  P{pages}  {question[:22]}")

    total = len(QUESTIONS)
    acc = correct / total
    avg_lat = sum(latencies) / total
    max_lat = max(latencies)
    print("\n" + "=" * 60)
    print(f"{'基线' if baseline else '优化'}链路：准确率 {correct}/{total}"
          f" = {acc:.1%}；≤3秒 {within_3s}/{total}；"
          f"平均 {avg_lat:.2f}s；最大 {max_lat:.2f}s")
    print("=" * 60)

    out_dir = os.path.join(os.path.dirname(CONFIG.base_dir), "03_测试")
    os.makedirs(out_dir, exist_ok=True)
    csv_name = "评测结果_基线.csv" if baseline else "评测结果.csv"
    csv_path = os.path.join(out_dir, csv_name)
    fields = ["编号", "公司", "问题", "答案", "gold", "是否正确",
              "命中项数", "缺失项", "页码", "耗时(秒)"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[评测] CSV已写出：{csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="14题离线评测")
    parser.add_argument("--baseline", action="store_true",
                        help="评测无表格基线链路")
    args = parser.parse_args()
    run(baseline=args.baseline)
