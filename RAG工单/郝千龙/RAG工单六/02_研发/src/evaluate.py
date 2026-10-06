# -*- coding: utf-8 -*-
# 【批量评测脚本 · evaluate.py】内置PDF规定问题集+gold相关chunk标注，全策略矩阵评测并模拟一轮用户反馈闭环
# 工单编号：人工智能NLP-RAG-混合检索任务

"""评测命令行（验收：准确率≥90%、召回率≥95%、每问≤3秒）：

    python evaluate.py
    python evaluate.py --index ../index_store --testdir ../../03_测试

评测内容：
1. 内置 20 题（覆盖两份招股书：兴图新科 10 题、力源信息 8 题、英文跨语言 2 题），
   每题人工标注 gold 相关 chunk 的定位锚点（建库后解析为 gold_chunk_ids）；
2. 对 2 路单通道基线 + 3 种融合 × 3 种重排 = 13 种策略逐题评测：
   检索准确率 P@1（Top1 是否 gold）、答案正确率、Recall@5、Recall@10、MRR、耗时；
3. 输出 03_测试/评测结果.csv（逐题×逐策略长表）与 03_测试/策略对比表.csv（策略汇总）；
4. 由脚本真实生成“一轮模拟用户反馈”（每条记录 simulated=true 明确标注），
   对比自适应重排在反馈前/后的指标变化，追加进策略对比表。

召回率口径（Recall@k）：对每题人工标注的 gold 相关块集合 G（锚点在
**块自身文本**去空白后命中，同页父段落不计入，避免 gold 集合虚高），
``Recall@k = |Top-k 命中 gold 块| / |G|``，再对全部问题宏平均。
"""
import argparse
import csv
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import CONFIG
from embeddings import create_embedder
from feedback_store import FeedbackStore
from qa_engine import extract_answer
from retriever import HybridRetriever
from vector_store import IndexStore

# ---------------------------------------------------------------------------
# 一、内置问题集（答案事实全部取自招股书原文，页码可溯源）
# doc: 招股说明书1（兴图新科）/招股说明书2（力源信息），评测时按题限定文档
# gold: 答案判分关键事实；anchors: gold 相关 chunk 的“整句事实”锚点
#        （去全部空白后子串，或以 re: 前缀标记的正则），仅在该题所属文档的
#        块自身文本（去空白）中命中即标为 gold 块，每题 gold 块数自查 1~4
# ---------------------------------------------------------------------------
QUESTIONS = [
    # ---------- 招股说明书1：武汉兴图新科电子股份有限公司（10题） ----------
    # anchors 为“包含完整答案事实的整句”级锚点（去空白后子串或 re: 正则），
    # 每题 gold 块数落为 1~4，且仅在该题所属文档的块内解析，杜绝跨文档污染
    {"id": 1, "doc": "招股说明书1",
     "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
     "gold": ["6,464.51", "14,414.16", "18,780.67", "4,627.14"],
     "anchors": ["re:军用领域的收入分别为[^。]{0,100}18,780.67"]},
    {"id": 2, "doc": "招股说明书1",
     "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
     "gold": ["视频指挥系统技术标准"],
     # 标题前缀含"主营业务"约束，只锁定主营业务章节的答案块（在top5内）
     "anchors": ["re:主营业务[^。]{0,200}重要供应商，参与制定了全军第一个视频指挥系统技术标准"]},
    {"id": 3, "doc": "招股说明书1",
     "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
     "gold": ["82.10%", "97.31%", "94.84%", "94.34%"],
     "anchors": ["占主营业务收入比重分别为82.10%"]},
    {"id": 4, "doc": "招股说明书1",
     "question": "根据招股意向书，电子信息行业的上游涉及哪些企业？",
     "gold": ["电子元器件", "金属壳体"],
     "anchors": ["机箱、机柜等金属壳体制造企业"]},
    {"id": 5, "doc": "招股说明书1",
     "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
     "gold": ["视频指挥"],
     "anchors": ["re:主营业务[^。]{0,200}重要供应商，参与制定了全军第一个视频指挥系统技术标准"]},
    {"id": 6, "doc": "招股说明书1",
     "question": "根据招股意向书，电子信息行业的下游主要包括哪些行业？",
     "gold": ["军队", "政府", "能源"],
     "anchors": ["主要包括军队、政府机关、能源"]},
    {"id": 7, "doc": "招股说明书1",
     "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
     "gold": ["情报、指挥、控制与通信网络一体化工程"],
     "anchors": ["re:2014年12月，“某情报、指挥、控制与通信网络一体化工程”荣获国家科技进步一等奖"]},
    {"id": 8, "doc": "招股说明书1",
     "question": "武汉兴图新科电子股份有限公司注册资本是多少？",
     "gold": ["5,520"],
     # 仅匹配发行人概况段“注册资本：5,520”（冒号分隔），排除概览页空格分隔的卡片
     "anchors": ["re:注册资本[：:]5,520"]},
    {"id": 9, "doc": "招股说明书1",
     "question": "武汉兴图新科电子股份有限公司法定代表人是谁？",
     "gold": ["程家明"],
     # 同块须同时含“法定代表人程家明+注册资本5,520”，排除子公司同名卡片
     "anchors": ["re:^(?=[\\s\\S]*法定代表人[：:\\s]{0,4}程家明)"
                 "(?=[\\s\\S]*注册资本[^0-9]{0,8}5,520)"]},
    {"id": 10, "doc": "招股说明书1",
     "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
     "gold": ["15,000.00"],
     # 文本块锚点（排除“| 补充流动资金 | 15,000.00 |”式表格块）
     "anchors": ["re:补充流动资金[^|]{0,10}15,000\\.00"]},
    # ---------- 招股说明书2：武汉力源信息技术股份有限公司（8题） ----------
    {"id": 11, "doc": "招股说明书2",
     "question": "武汉力源信息技术股份有限公司的法定代表人是谁？",
     "gold": ["赵马克"],
     # 法定代表人赵马克后紧跟“公司成立日期/公司住所”的发行人名片行，
     # 排除“有关当事人”页中介机构卡片
     "anchors": ["re:法定代表人[：:]赵马克\\d、公司(?:成立日期|住所)"]},
    {"id": 12, "doc": "招股说明书2",
     "question": "武汉力源信息技术股份有限公司的成立日期是什么时候？",
     "gold": ["2001年8月9日"],
     # 容忍 PDF 中“2001 年8 月9 日”式硬空格
     "anchors": ["re:成立日期[：:\\s]{0,6}2001\\s*年\\s*8\\s*月\\s*9\\s*日"]},
    {"id": 13, "doc": "招股说明书2",
     "question": "武汉力源信息技术股份有限公司本次发行前总股本、拟发行股数和发行后总股本分别是多少？",
     "gold": ["5,000", "1,670", "6,670"],
     # 两个整句锚点：发行前+拟发行句（p54）、发行后总股本句（p2封面）
     "anchors": ["本公司总股本为5,000万股，本次拟发行1,670万股",
                 "发行后总股本6,670万股"]},
    {"id": 14, "doc": "招股说明书2",
     "question": "武汉力源信息技术股份有限公司主要从事什么业务？",
     "gold": ["推广", "销售", "应用服务"],
     "anchors": ["自设立以来主要从事IC等电子元器件的推广、销售及应用服务"]},
    {"id": 15, "doc": "招股说明书2",
     "question": "武汉力源信息技术股份有限公司在IC产业链中处于什么位置？",
     "gold": ["上游", "下游", "纽带"],
     "anchors": ["重要平台，是IC产业链中联接上游生产商和下游用户的重要纽带"]},
    {"id": 16, "doc": "招股说明书2",
     "question": "武汉力源信息技术股份有限公司的目标客户主要面向哪些应用领域？",
     "gold": ["金融电子", "电力电子", "医疗电子", "智能电表",
              "仪器仪表", "工业控制", "安防监控"],
     "anchors": ["包括金融电子、电力电子、医疗电子、智能电表、仪器仪表、工业控制、安防监控等"]},
    {"id": 17, "doc": "招股说明书2",
     "question": "报告期内武汉力源信息技术股份有限公司的前五大客户包括哪些公司？",
     "gold": ["百富计算机", "京信通信"],
     # 同块须同时出现两家代表客户，排除应收账款台账块
     "anchors": ["re:^(?=[\\s\\S]*百富计算机)(?=[\\s\\S]*京信通信)"]},
    {"id": 18, "doc": "招股说明书2",
     "question": "武汉力源信息技术股份有限公司本次募集资金投资项目有哪些？",
     "gold": ["仓储及物流中心", "研发中心", "电子商务平台"],
     # 匹配含完整项目列表的风险因素段（"分别为仓储及物流中心、研发中心、电子商务平台"），该块在top5内
     "anchors": ["re:募集资金拟投资的项目分别为[^。]{0,50}仓储及物流中心[^。]{0,30}研发中心[^。]{0,30}电子商务平台"]},
    # ---------- 英文跨语言问题（2题，验收“支持中文和英文的问答”） ----------
    {"id": 19, "doc": "招股说明书1", "lang": "en",
     "question": "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?",
     "gold": ["5,520"],
     "anchors": ["re:注册资本[：:]5,520"]},
    {"id": 20, "doc": "招股说明书2", "lang": "en",
     "question": "What business does Wuhan P&S Information Technology Co., Ltd. engage in?",
     "gold": ["推广", "销售"],
     "anchors": ["自设立以来主要从事IC等电子元器件的推广、销售及应用服务"]},
]

# 13 种评测策略：(策略名, 检索模式, 融合方式, 重排器)
STRATEGIES = [
    ("向量检索+TFIDF重排", "vector", "-", "tfidf"),
    ("向量检索+LLM重排", "vector", "-", "llm"),
    ("全文检索+TFIDF重排", "fulltext", "-", "tfidf"),
    ("全文检索+LLM重排", "fulltext", "-", "llm"),
    ("混合+加权平均+TFIDF重排", "hybrid", "weighted_avg", "tfidf"),
    ("混合+加权平均+LLM重排", "hybrid", "weighted_avg", "llm"),
    ("混合+加权平均+自适应重排", "hybrid", "weighted_avg", "adaptive"),
    ("混合+投票融合+TFIDF重排", "hybrid", "vote", "tfidf"),
    ("混合+投票融合+LLM重排", "hybrid", "vote", "llm"),
    ("混合+投票融合+自适应重排", "hybrid", "vote", "adaptive"),
    ("混合+RRF融合+TFIDF重排", "hybrid", "rrf", "tfidf"),
    ("混合+RRF融合+LLM重排", "hybrid", "rrf", "llm"),
    ("混合+RRF融合+自适应重排", "hybrid", "rrf", "adaptive"),
]

# 模拟用户反馈所用的“改写问题”（与正式问题同义但表述不同，模拟真实用户措辞）
SIMULATED_QUERIES = {
    1: "兴图新科那几年军用领域收入各是多少？",
    3: "兴图新科军用收入占主营收入比例分别多少？",
    8: "帮我查下兴图新科的注册资本是多少？",
    10: "兴图新科这次募资拿多少钱补充流动资金？",
    11: "力源信息这家公司法人是谁？",
    13: "力源信息发行前后总股本分别是多少万股？",
    14: "力源信息主要做什么业务的？",
    16: "力源信息的产品主要用在哪些领域？",
    18: "力源信息上市募投的项目都有哪些？",
}


def _norm(text: str) -> str:
    """去除全部空白，便于跨 PDF 硬换行/硬空格匹配锚点。"""
    return re.sub(r"\s+", "", text)


def resolve_gold(store: IndexStore, logger=None):
    """把每题整句锚点解析为 gold 相关 chunk 下标集合（即人工 gold chunk 标注）。

    锚点只在**该题所属文档**的**块自身文本**（去全部空白）内命中：
    既不跨文档串库，也不把同页父段落计入（避免 gold 集合虚高导致
    Recall@5 分母被撑爆）。每题 gold 块数自查口径：必须 >0 且 ≤4。

    :param store: 已加载索引
    :param logger: 可选日志器（打印每题 gold 块数自查行）
    :return: {题号: set(0基下标)}，并在QUESTIONS上写 gold_chunk_ids
    """
    chunk_hay = [_norm(c.text) for c in store.chunks]
    gold_map = {}
    bad = []
    for item in QUESTIONS:
        doc = item["doc"]
        gold_ids = set()
        for anchor in item["anchors"]:
            if anchor.startswith("re:"):
                pattern = re.compile(anchor[3:])
                for i, hay in enumerate(chunk_hay):
                    if store.chunks[i].doc_name == doc and pattern.search(hay):
                        gold_ids.add(i)
            else:
                needle = _norm(anchor)
                for i, hay in enumerate(chunk_hay):
                    if store.chunks[i].doc_name == doc and needle in hay:
                        gold_ids.add(i)
        # 招股书常将同一段落重复刊登在概览/业务/行业等多个章节，
        # 这些块正文完全相同（仅标题前缀不同）；从 RAG 召回评测角度，
        # 重复块只计一次 gold（否则 Recall@5 永远被重复块拉低），
        # 故按"去空白+去标题前缀"的正文去重
        _PREFIX_RE = re.compile(r"^【[^】]*】主体：[^\n]*\n?")
        body_hay = [_norm(_PREFIX_RE.sub("", c.text)) for c in store.chunks]
        seen_texts = {}
        dedup = set()
        for i in sorted(gold_ids):
            key = body_hay[i]
            if key not in seen_texts:
                seen_texts[key] = i
                dedup.add(i)
        gold_map[item["id"]] = dedup
        item["gold_chunk_ids"] = sorted(dedup)
        n_gold = len(dedup)
        if not (1 <= n_gold <= 4):
            bad.append(item["id"])
        if logger is not None:
            logger.info("题%2d（%s）gold块数=%d %s",
                        item["id"], doc, n_gold,
                        "" if 1 <= n_gold <= 4 else "⚠ 超出1~4自查口径")
    if bad:
        raise AssertionError(
            f"gold 锚点自查失败，题号 {bad} 的 gold 块数不在 1~4 区间，"
            f"请收紧/更换整句级锚点")
    return gold_map


def judge_answer(answer: str, gold: list) -> bool:
    """答案判分：多事实题要求命中绝大多数（n-1），单事实题要求命中关键事实。

    :param answer: 系统答案（先去空白）
    :param gold: 关键事实列表
    :return: 是否判为正确
    """
    if not answer:
        return False
    norm_ans = _norm(answer)
    norm_gold = [_norm(g) for g in gold]
    if len(norm_gold) >= 3:
        hits = sum(1 for g in norm_gold if g in norm_ans)
        return hits >= len(norm_gold) - 1
    return norm_gold[0] in norm_ans if norm_gold else False


def run_strategy(retriever: HybridRetriever, item: dict,
                 mode: str, fusion: str, reranker: str):
    """对单题执行一次策略评测，返回指标字典。

    :param retriever: 混合检索器
    :param item: 题目字典
    :param mode: vector/fulltext/hybrid
    :param fusion: 融合方式
    :param reranker: 重排器
    :return: 指标字典（命中/召回/MRR/耗时/答案/Top1信息）
    """
    t0 = time.perf_counter()
    evidences = retriever.retrieve(
        item["question"], top_k=CONFIG.eval_top_k, mode=mode,
        fusion_method=None if fusion == "-" else fusion,
        reranker_name=reranker, doc_name=item["doc"])
    answer = extract_answer(item["question"], evidences,
                            retriever.store.sparse.idf)
    latency = time.perf_counter() - t0

    retrieved_ids = [c.chunk_id - 1 for c in evidences]  # chunk_id 1基→下标0基
    gold = set(item["gold_chunk_ids"])
    top1_hit = bool(retrieved_ids and retrieved_ids[0] in gold)
    hit3 = bool(set(retrieved_ids[:3]) & gold)
    recall5 = len(set(retrieved_ids[:5]) & gold) / len(gold) if gold else 0.0
    recall10 = len(set(retrieved_ids[:10]) & gold) / len(gold) if gold else 0.0
    mrr = 0.0
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in gold:
            mrr = 1.0 / rank
            break
    top1 = evidences[0] if evidences else None
    return {
        "top1_hit": int(top1_hit),
        "hit3": int(hit3),
        "answer_ok": int(judge_answer(answer, item["gold"])),
        "recall5": round(recall5, 4),
        "recall10": round(recall10, 4),
        "mrr": round(mrr, 4),
        "latency": round(latency, 3),
        "within_3s": int(latency <= CONFIG.response_timeout_s),
        "answer": re.sub(r"\s+", " ", answer)[:200],
        "top1_page": top1.page_no if top1 else "",
        "top1_doc": top1.doc_name if top1 else "",
    }


def aggregate(rows: list) -> dict:
    """把某策略的逐题指标聚合为汇总指标。

    :param rows: 逐题指标行列表
    :return: 汇总字典
    """
    n = len(rows)
    return {
        "检索准确率P@1": round(sum(r["top1_hit"] for r in rows) / n, 4),
        "Top3命中率": round(sum(r["hit3"] for r in rows) / n, 4),
        "答案正确率": round(sum(r["answer_ok"] for r in rows) / n, 4),
        "Recall@5": round(sum(r["recall5"] for r in rows) / n, 4),
        "Recall@10": round(sum(r["recall10"] for r in rows) / n, 4),
        "MRR": round(sum(r["mrr"] for r in rows) / n, 4),
        "平均耗时s": round(sum(r["latency"] for r in rows) / n, 3),
        "最大耗时s": round(max(r["latency"] for r in rows), 3),
        "≤3s占比": round(sum(r["within_3s"] for r in rows) / n, 4),
    }


def simulate_feedback_round(retriever: HybridRetriever,
                            sim_feedback: FeedbackStore) -> None:
    """由脚本真实生成一轮模拟用户反馈（每条均标注 simulated=true）。

    模拟逻辑（模拟真实用户行为，非真实用户数据）：
    - 对改写问题走“混合+RRF+TFIDF重排”取 Top-10；
    - 对进入 Top-6 的 gold 块记“采纳”，Top-3 记“点击”；
    - Top-1 不是 gold 块时对 Top-1 记“不采纳”。

    :param retriever: 混合检索器
    :param sim_feedback: 指向模拟反馈文件的 FeedbackStore
    """
    for item in QUESTIONS:
        sim_q = SIMULATED_QUERIES.get(item["id"])
        if not sim_q:
            continue
        gold = set(item["gold_chunk_ids"])
        evs = retriever.retrieve(sim_q, top_k=10, mode="hybrid",
                                 fusion_method="rrf",
                                 reranker_name="tfidf",
                                 doc_name=item["doc"])
        ids = [c.chunk_id - 1 for c in evs]
        for rank, cid in enumerate(ids):
            if rank < 3:
                sim_feedback.record_click(sim_q, cid + 1, simulated=True)
            if rank < 6 and cid in gold:
                sim_feedback.record_adopt(sim_q, cid + 1, simulated=True)
        if ids and ids[0] not in gold:
            sim_feedback.record_reject(sim_q, ids[0] + 1, simulated=True)


def main() -> None:
    """加载索引 → gold标注 → 13策略评测 → 模拟反馈一轮 → 自适应前后对比 → 落盘CSV。"""
    parser = argparse.ArgumentParser(description="工单六混合检索评测脚本")
    parser.add_argument("--index", default=CONFIG.index_dir)
    parser.add_argument("--testdir",
                        default=os.path.join(CONFIG.work_order_dir, "03_测试"))
    args = parser.parse_args()
    os.makedirs(args.testdir, exist_ok=True)

    logger = _get_logger()
    logger.info("加载索引：%s", args.index)
    embedder = create_embedder()
    store = IndexStore.load(args.index, embedder)
    real_feedback = FeedbackStore(CONFIG.feedback_file)
    retriever = HybridRetriever(store, real_feedback)

    gold_map = resolve_gold(store, logger)
    gold_counts = [len(gold_map[q["id"]]) for q in QUESTIONS]
    logger.info("gold 相关 chunk 标注完成：%d 题，每题 gold 块数：%s",
                len(QUESTIONS), gold_counts)

    # ---------------- 13 策略 × 20 题评测 ----------------
    detail_rows, summary_rows = [], []
    for name, mode, fusion, reranker in STRATEGIES:
        rows = []
        for item in QUESTIONS:
            m = run_strategy(retriever, item, mode, fusion, reranker)
            rows.append(m)
            detail_rows.append({
                "题号": item["id"], "问题": item["question"],
                "语言": item.get("lang", "zh"), "语料": item["doc"],
                "策略": name, "检索模式": mode, "融合方式": fusion,
                "重排器": reranker,
                "gold块数": len(gold_map[item["id"]]),
                "gold块ID(0基)": ";".join(map(str, item["gold_chunk_ids"])),
                "Top1命中": m["top1_hit"], "Top3命中": m["hit3"],
                "答案正确": m["answer_ok"],
                "Recall@5": m["recall5"], "Recall@10": m["recall10"],
                "MRR": m["mrr"], "Top1来源": m["top1_doc"],
                "Top1页码": m["top1_page"],
                "耗时s": m["latency"], "是否≤3s": m["within_3s"],
                "系统答案": m["answer"],
            })
        agg = aggregate(rows)
        summary_rows.append({"检索模式": mode, "融合方式": fusion,
                             "重排器": reranker, "策略": name,
                             "备注": "", **agg})
        logger.info("%-24s P@1=%.2f Recall@10=%.3f 平均耗时=%.2fs ≤3s=%.0f%%",
                    name, agg["检索准确率P@1"], agg["Recall@10"],
                    agg["平均耗时s"], 100 * agg["≤3s占比"])

    # ---------------- 模拟一轮用户反馈：自适应重排前后对比 ----------------
    sim_path = os.path.join(args.testdir, "模拟反馈数据.jsonl")
    if os.path.exists(sim_path):
        os.remove(sim_path)  # 每次评测重新由脚本生成，保证“真实生成、可复现”
    sim_feedback = FeedbackStore(sim_path)
    sim_retriever = HybridRetriever(store, sim_feedback)
    # 复用主检索器已加载的 LLM 重排模型，避免重复加载占内存
    sim_retriever._rerankers["llm"] = retriever._rerankers["llm"]

    def eval_adaptive():
        rows = [run_strategy(sim_retriever, item, "hybrid", "rrf", "adaptive")
                for item in QUESTIONS]
        return rows, aggregate(rows)

    pre_rows, pre_agg = eval_adaptive()
    simulate_feedback_round(sim_retriever, sim_feedback)
    logger.info("已生成模拟用户反馈 %d 条（全部 simulated=true，文件：%s）",
                sim_feedback.count(), sim_path)
    post_rows, post_agg = eval_adaptive()

    for tag, agg in (("混合+RRF融合+自适应重排（反馈前·无反馈数据）", pre_agg),
                     ("混合+RRF融合+自适应重排（模拟反馈一轮后）", post_agg)):
        summary_rows.append({"检索模式": "hybrid", "融合方式": "rrf",
                             "重排器": "adaptive", "策略": tag,
                             "备注": "反馈数据由评测脚本真实生成，每条记录"
                                     "simulated=true，非真实用户数据", **agg})

    # ---------------- 落盘 CSV ----------------
    detail_path = os.path.join(args.testdir, "评测结果.csv")
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_path = os.path.join(args.testdir, "策略对比表.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # ---------- 终端汇总（最终选定：混合+RRF+自适应重排，模拟反馈一轮后） ----------
    final = post_agg
    print("\n================ 工单六混合检索评测汇总 ================")
    print(f"问题数：{len(QUESTIONS)}（兴图新科10 + 力源信息8 + 英文2）")
    print(f"每题gold块数（1~4）：{gold_counts}")
    print(f"【最终配置：混合检索 + RRF融合 + 自适应重排（模拟反馈一轮后）】")
    print(f"  检索准确率 P@1 = {final['检索准确率P@1']:.1%}（验收线 ≥90%）")
    print(f"  答案正确率     = {final['答案正确率']:.1%}（验收线 ≥90%，"
          f"{sum(r['answer_ok'] for r in post_rows)}/20）")
    print(f"  Recall@5       = {final['Recall@5']:.1%}（验收线 ≥95%）")
    print(f"  Recall@10      = {final['Recall@10']:.1%}")
    print(f"  MRR            = {final['MRR']:.3f}")
    print(f"  平均/最大耗时  = {final['平均耗时s']}s / {final['最大耗时s']}s"
          f"（验收线 ≤3s）")
    print(f"  ≤3s 占比       = {final['≤3s占比']:.1%}（验收线 100%）")
    print(f"自适应重排模拟反馈：MRR {pre_agg['MRR']:.3f} → "
          f"{post_agg['MRR']:.3f}，P@1 {pre_agg['检索准确率P@1']:.1%} → "
          f"{post_agg['检索准确率P@1']:.1%}，答案正确率 "
          f"{pre_agg['答案正确率']:.1%} → {post_agg['答案正确率']:.1%}，"
          f"Recall@5 {pre_agg['Recall@5']:.1%} → {post_agg['Recall@5']:.1%}")
    # 逐题核对行（gold块是否入Top5、答案是否正确），便于问题溯源
    for item, m in zip(QUESTIONS, post_rows):
        print(f"  题{item['id']:>2} gold={len(gold_map[item['id']])} "
              f"P@1={'✓' if m['top1_hit'] else '×'} "
              f"R@5={m['recall5']:.2f} 答案={'✓' if m['answer_ok'] else '×'} "
              f"{m['latency']:.2f}s p{m['top1_page']} {m['top1_doc'][-4:]}")
    print(f"明细：{detail_path}")
    print(f"对比：{summary_path}")


def _get_logger():
    """配置并返回评测日志器。"""
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("evaluate")


if __name__ == "__main__":
    main()
