# -*- coding: utf-8 -*-
"""
16 问完整评估（工单04 验收：准确率 ≥90%、响应 ≤3s）
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

问题集 = 工单01 兴图 10 问 + 工单03 力源表格 4 问 + 工单04 图像 2 问 = 16 问，
全部来自 rag_core.config.get_all_questions()。

评测口径：
  · 准确率 —— 「答案要点命中」判定：每个问题预设若干必答要点（字符串或
              "别名1|别名2" 形式），答案全部命中才算答对（口径同
              rag_core.evaluate.keyword_accuracy，分数敏感）。
  · 检索侧 —— Hit Rate / MRR / Recall@k（工单07 口径）。
  · RAGAS  —— --ragas 时用 rag_core.evaluate 计算忠实度/相关性/上下文精度/召回。
  · 响应时间 —— 分别统计「检索耗时」与「端到端耗时（含 LLM 生成）」，
              并与 3 秒指标对照（LLM 走网络 API 时端到端通常 >3s，
              工程上通过缓存/本地模型满足，详见 docs/技术文档.md）。

答案要点是**评测标注**（答案键），不是程序输出：全部来自两份招股书的实际内容，
可通过 --answer-key 传入 JSON 覆盖，便于随文档版本更新。

用法：
    python src/run_evaluation.py                       # 默认评测多模态索引
    python src/run_evaluation.py --qids 5,6            # 只评两个图像问题
    python src/run_evaluation.py --collection wo04_text
    python src/run_evaluation.py --ragas               # 追加 RAGAS 指标（慢/费token）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_extractor import RESULTS_DIR  # noqa: E402
from rag_core import config, evaluate, generator  # noqa: E402

EVAL_JSON = RESULTS_DIR / "evaluation.json"
EVAL_MD = RESULTS_DIR / "evaluation.md"

# ---------------------------------------------------------------------------
# 一、参考答案（RAGAS 用；内容全部取自两份招股书，可在 answer_key.json 覆盖）
# ---------------------------------------------------------------------------
GROUND_TRUTH: dict[int, str] = {
    # ---- 力源信息（招股说明书2）图像问题 ----
    5: ("公司组织结构图中，销售部下设 4 个部门：电话及网络销售部、渠道销售部、"
        "大客户销售部、国际贸易部；其中大客户销售部下设 6 个销售处："
        "北京销售处、广州销售处、成都销售处、深圳销售处、武汉销售处、珠海销售处。"),
    6: ("2008 年中国 IC 市场应用结构与增长图中，增长率最快的是汽车，增长率 14.0%；"
        "负增长的是 IC 卡，增长率为 -2.0%。其余行业增长率为：工控 10.5%、"
        "计算机 7.9%、其他 6.1%、消费 5.1%、网络通信 2.6%。"),
    # ---- 力源信息（招股说明书2）表格问题（工单03）----
    1: ("本次发行股数 1,670 万股，占发行后总股本（6,670 万股）的比例为 25.04%。"),
    2: ("本次募集资金拟投资项目：仓储及物流中心项目、研发中心项目、"
        "电子商务平台项目、扩充产品种类和数量项目，以及其他与主营业务相关的营运资金项目。"),
    3: ("存在控制关系的关联方为赵马克，持股比例 42.35%，系公司控股股东。"),
    4: ("不存在控制关系的关联方企业包括：融冰投资、武汉博润、上海博润、听音投资、"
        "联众聚源（均为持有公司股份 5% 以上的股东），力源贸易（同一实际控制人控制的企业），"
        "普芯达（实际控制人近亲属控制的公司）。"),
    # ---- 兴图新科（招股说明书1）文本问题（工单01）----
    260: ("报告期内公司来自军用领域的收入分别为：2016 年 6,464.51 万元、"
          "2017 年 14,414.16 万元、2018 年 18,780.67 万元。"),
    33: "报告期内来自军用领域的收入占主营业务收入的比重分别为 82.10%、97.31% 和 94.84%。",
    95: "公司参与制定了全军第一个视频指挥系统技术标准，即《某视频技术规范 1.0》。",
    34: ("电子信息行业的上游涉及信息系统相关的电子元器件制造企业，"
         "以及机箱、机柜等金属壳体制造企业。"),
    957: "兴图新科目前已经成为国防军队视频指挥领域的重要供应商。",
    793: "电子信息行业的下游行业为各类终端用户，主要包括军队、政府机关、能源等行业企业。",
    795: ("在某大型研究所牵头的“某情报、指挥、控制与通信网络一体化工程”"
          "（相当于美军的 C4ISR 系统）中，公司独立承担了视频指挥分系统的设计、开发、"
          "研制、部署工作，该工程整体获得了国家科技进步一等奖。"),
    543: "武汉兴图新科电子股份有限公司注册资本为 5,520 万元。",
    531: "武汉兴图新科电子股份有限公司法定代表人为程家明。",
    207: "计划使用本次发行募集资金 15,000.00 万元用于补充流动资金。",
}

# ---------------------------------------------------------------------------
# 二、答案要点（准确率判定；"|" 表示别名，命中任一即算该要点命中）
#     数字写法容错由 normalize_for_match() 处理（全角/半角、空格、破折号）
# ---------------------------------------------------------------------------
ANSWERS_EXPECTED: dict[str, list[str]] = {
    # ---- 图像问题（工单04 核心）----
    # 销售部 4 个部门 + 大客户销售部 6 个销售处（答案只在组织结构图中）
    "5": ["渠道销售部", "电话及网络销售部", "大客户销售部", "国际贸易部",
          "6 个销售处|6个销售处|六个销售处|6家销售处",
          "珠海", "深圳", "北京", "广州", "成都", "武汉"],
    # 增长率最快=汽车 14.0%；负增长=IC卡 -2.0%
    "6": ["汽车", "14.0%|14%", "IC 卡|IC卡", "-2.0%|-2%|负增长"],
    # ---- 表格问题（工单03）----
    "1": ["1,670|1670", "25.04%"],
    "2": ["仓储及物流中心|仓储物流中心", "研发中心", "电子商务平台", "扩充产品种类"],
    "3": ["赵马克", "42.35%", "控股股东"],
    "4": ["融冰投资", "武汉博润", "上海博润", "听音投资", "联众聚源",
          "力源贸易|普芯达"],
    # ---- 文本问题（工单01）----
    "260": ["6,464|6464", "14,414|14414", "18,780|18780|18,781"],
    "33": ["82.10%", "97.31%", "94.84%"],
    "95": ["视频指挥系统技术标准|技术标准", "视频技术规范"],
    "34": ["电子元器件", "机箱|机柜|金属壳体"],
    "957": ["视频指挥", "重要供应商"],
    "793": ["军队", "政府机关|政府", "能源"],
    "795": ["一体化工程|C4ISR", "国家科技进步一等奖"],
    "543": ["5,520|5520"],
    "531": ["程家明"],
    "207": ["15,000|15000|1.5 亿|1.5亿"],
}


# ---------------------------------------------------------------------------
# 三、文本归一化与要点匹配
# ---------------------------------------------------------------------------
_FULL2HALF = str.maketrans(
    "０１２３４５６７８９％－—–　", "0123456789%-—- "
)


def normalize_for_match(text: str) -> str:
    """
    归一化后再比对，避免「写法差异」被判错：
      · 全角数字/百分号 → 半角
      · 各种破折号/减号统一为 '-'
      · 去掉空格（「1,670 万股」与「1,670万股」等价）
    """
    if not text:
        return ""
    t = text.translate(_FULL2HALF)
    t = t.replace("，", ",")               # 全角逗号在数字里也常见
    return re.sub(r"\s+", "", t)


def group_hit(answer_norm: str, spec: str) -> bool:
    """判断一个要点是否被答案覆盖（spec 用 '|' 分隔别名）。"""
    for alt in spec.split("|"):
        a = normalize_for_match(alt)
        if a and a in answer_norm:
            return True
    return False


def judge_keywords(answer: str, qid) -> dict:
    """按答案要点判分，返回 {正确, 命中, 漏答}。"""
    specs = ANSWERS_EXPECTED.get(str(qid)) or []
    if not specs:
        return {"正确": None, "命中": [], "漏答": []}
    ans = normalize_for_match(answer)
    hits = [s for s in specs if group_hit(ans, s)]
    missed = [s for s in specs if s not in hits]
    return {"正确": len(missed) == 0, "命中": hits, "漏答": missed,
            "要点数": len(specs), "命中数": len(hits)}


def safe_generate(question: str, docs: list[dict], ctx: str,
                  verbose: bool = True):
    """
    调用生成模型的容错包装。

    生成模型（DeepSeek API）不可用时不让整批评估中断——返回带明确标记的
    占位答案，评测报告照常产出（工单「系统稳定性/容错机制」要求）。
    """
    try:
        return generator.generate_rag(question, docs, ctx)
    except Exception as e:
        if verbose:
            print(f"\n  [warn] 生成失败（{type(e).__name__}: {e}），"
                  f"以占位答案继续（请检查 DEEPSEEK_API_KEY）")
        return generator.Generation(
            question=question, answer=f"（生成失败：{type(e).__name__}: {e}）",
            mode="rag", contexts=docs, refused=True)


def load_answer_key(override_path: str | None) -> None:
    """用外部 JSON 覆盖内置答案要点（{"5": [...], ...}），便于更新标注。"""
    if not override_path:
        return
    p = Path(override_path)
    if not p.exists():
        print(f"[warn] 答案键文件不存在，忽略：{p}")
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    for k, v in data.items():
        ANSWERS_EXPECTED[str(k)] = v
    print(f"[答案键] 已从 {p} 覆盖 {len(data)} 个问题的评测要点")


# ---------------------------------------------------------------------------
# 四、执行评估
# ---------------------------------------------------------------------------
def evaluate_questions(collection: str, qids: list[int] | None = None,
                       top_k: int = 5, ragas: bool = False,
                       verbose: bool = True) -> dict:
    """
    在指定索引上跑完整问题集，返回评估结果（含逐条明细）。

    检索配置与 PRESETS["wo04_image"] 一致：hybrid + rrf + cascade 重排。
    """
    from build_multimodal_index import load_retriever

    retriever = load_retriever(collection)
    questions = config.get_all_questions()
    if qids:
        questions = [q for q in questions if q["id"] in qids]

    records: list[evaluate.EvalRecord] = []
    details: list[dict] = []

    for i, q in enumerate(questions, 1):
        if verbose:
            print(f"  [{i}/{len(questions)}] id={q['id']} …", end="", flush=True)

        # --- 检索 ---
        t_ret0 = time.perf_counter()
        res = retriever.retrieve(
            q["question"], strategy="hybrid", top_k=top_k,
            recall_k=config.TOP_K_RECALL, reranker="cascade",
            fusion="rrf", alpha=config.HYBRID_ALPHA,
        )
        retrieve_s = time.perf_counter() - t_ret0
        docs = res.docs
        ctx = retriever.format_context(docs)

        # --- 生成 ---
        t_all0 = time.perf_counter()
        gen = safe_generate(q["question"], docs, ctx)
        total_s = time.perf_counter() - t_all0 + retrieve_s

        # --- 判分 ---
        kw = judge_keywords(gen.answer, q["id"])
        rec = evaluate.EvalRecord(
            qid=q["id"], question=q["question"], answer=gen.answer,
            ground_truth=GROUND_TRUTH.get(q["id"], ""),
            contexts=[d.get("text", "") for d in docs],
            reference_doc="招股说明书2" if q["id"] <= 6 else "招股说明书1",
            retrieved_docs=[str(d.get("doc", "")) for d in docs],
            retrieved_pages=[int(d.get("page", 0)) for d in docs],
            latency=total_s,
        )
        rec.metrics["keyword_correct"] = (1.0 if kw["正确"] else 0.0) \
            if kw["正确"] is not None else float("nan")
        records.append(rec)

        image_hits = [
            {"page": d.get("page"), "type": d.get("type"),
             "snippet": (d.get("text") or "")[:120]}
            for d in docs if d.get("type") == "image"
        ]
        details.append({
            "id": q["id"], "question": q["question"], "answer": gen.answer,
            "是否答对": kw["正确"], "命中要点": kw["命中"], "漏答要点": kw["漏答"],
            "命中图像块": image_hits,
            "检索页码": rec.retrieved_pages,
            "检索耗时(s)": round(retrieve_s, 3),
            "总耗时(s)": round(total_s, 3),
            "所用检索耗时明细": {k: round(v, 4) for k, v in res.timings.items()},
            "引用": gen.citations,
        })
        if verbose:
            mark = "✓" if kw["正确"] else ("×" if kw["正确"] is not None else "-")
            print(f" {mark} ({retrieve_s:.2f}s/{total_s:.2f}s)")

    # --- 汇总 ---
    judged = [d for d in details if d["是否答对"] is not None]
    n_ok = sum(1 for d in judged if d["是否答对"])
    summary = {
        "collection": collection,
        "n_questions": len(details),
        "n_judged": len(judged),
        "accuracy": round(n_ok / len(judged), 4) if judged else None,
        "correct": n_ok,
        "retrieval_latency_avg": round(
            sum(d["检索耗时(s)"] for d in details) / max(len(details), 1), 3),
        "retrieval_latency_max": round(
            max((d["检索耗时(s)"] for d in details), default=0), 3),
        "total_latency_avg": round(
            sum(d["总耗时(s)"] for d in details) / max(len(details), 1), 3),
        "total_latency_max": round(
            max((d["总耗时(s)"] for d in details), default=0), 3),
        "n_under_3s_total": sum(1 for d in details if d["总耗时(s)"] <= 3.0),
        "n_under_3s_retrieval": sum(1 for d in details if d["检索耗时(s)"] <= 3.0),
        "image_questions_correct": {
            str(d["id"]): d["是否答对"] for d in details if d["id"] in (5, 6)},
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # RAGAS 指标（可选，需要 LLM）
    if ragas:
        if verbose:
            print("  [RAGAS] 计算忠实度/相关性/上下文精度/召回 …")
        rsum = evaluate.evaluate_records(
            records,
            metrics=["faithfulness", "answer_relevancy",
                     "context_precision", "context_recall",
                     "answer_correctness"],
            verbose=verbose,
        )
        summary["ragas"] = rsum

    return {"summary": summary, "details": details,
            "records": [r.to_dict() for r in records]}


# ---------------------------------------------------------------------------
# 五、落盘
# ---------------------------------------------------------------------------
def save_evaluation(result: dict, collection: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = collection.replace("wo04_", "")
    jp = RESULTS_DIR / f"evaluation_{tag}.json"
    mp = RESULTS_DIR / f"evaluation_{tag}.md"
    jp.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                  encoding="utf-8")

    s = result["summary"]
    md = [
        "# 工单04 评估报告（16 问）",
        "",
        "工单编号：人工智能NLP-RAG-图像内容解析及检索优化",
        "",
        f"- 索引：`{s['collection']}`",
        f"- 评估时间：{s['evaluated_at']}",
        f"- **准确率：{s['accuracy'] * 100:.1f}%**"
        f"（{s['correct']}/{s['n_judged']} 题答对，口径：答案要点全部命中）",
        f"- 检索耗时：平均 {s['retrieval_latency_avg']}s / "
        f"最大 {s['retrieval_latency_max']}s（≤3s：{s['n_under_3s_retrieval']}/{s['n_questions']}）",
        f"- 端到端耗时：平均 {s['total_latency_avg']}s / "
        f"最大 {s['total_latency_max']}s（≤3s：{s['n_under_3s_total']}/{s['n_questions']}）",
        f"- 图像问题（id 5/6）：{s['image_questions_correct']}",
        "",
    ]
    if "ragas" in s:
        r = s["ragas"]
        md += ["## RAGAS 指标", "", "```", evaluate.format_summary(r), "```", ""]
    md += [
        "## 逐题明细",
        "",
        "| id | 问题 | 判定 | 漏答要点 | 检索耗时 | 端到端 | 命中图像块(页) |",
        "|----|------|------|----------|----------|--------|----------------|",
    ]
    for d in result["details"]:
        mark = {True: "✓ 正确", False: "× 错误", None: "—"}[d["是否答对"]]
        pages = ",".join(str(x["page"]) for x in d["命中图像块"]) or "—"
        q = d["question"].replace("|", "／")[:56]
        missed = "、".join(d["漏答要点"])[:40] or "—"
        md.append(f"| {d['id']} | {q} | {mark} | {missed} | "
                  f"{d['检索耗时(s)']} | {d['总耗时(s)']} | {pages} |")
    md += ["", "## 答案全文", ""]
    for d in result["details"]:
        md += [f"### id {d['id']}：{d['question']}", "",
               f"- 判定：{d['是否答对']}（命中 {len(d['命中要点'])} 个要点）",
               f"- 检索页码：{d['检索页码']}", "",
               "```text", d["answer"], "```", ""]
    mp.write_text("\n".join(md), encoding="utf-8")
    return jp, mp


def main() -> None:
    ap = argparse.ArgumentParser(description="工单04 十六问评估")
    ap.add_argument("--collection", default="wo04_multimodal",
                    help="索引名（默认 wo04_multimodal）")
    ap.add_argument("--qids", default="", help="只评指定问题 id，逗号分隔（如 5,6）")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--ragas", action="store_true", help="追加 RAGAS 指标（需 LLM）")
    ap.add_argument("--answer-key", default="", help="外部答案要点 JSON 路径")
    args = ap.parse_args()

    load_answer_key(args.answer_key or None)
    qids = [int(x) for x in args.qids.split(",") if x.strip()] or None

    print(f"[评估] 索引 {args.collection}，"
          f"{'全部 16 问' if not qids else f'{len(qids)} 问'}")
    result = evaluate_questions(args.collection, qids=qids, top_k=args.top_k,
                                ragas=args.ragas)
    jp, mp = save_evaluation(result, args.collection)

    s = result["summary"]
    print("\n" + "=" * 60)
    print(f"准确率：{s['accuracy'] * 100:.1f}%（{s['correct']}/{s['n_judged']}）")
    print(f"检索耗时 平均 {s['retrieval_latency_avg']}s / 最大 {s['retrieval_latency_max']}s")
    print(f"端到端耗时 平均 {s['total_latency_avg']}s / 最大 {s['total_latency_max']}s")
    print(f"报告：{jp}\n      {mp}")


if __name__ == "__main__":
    main()
