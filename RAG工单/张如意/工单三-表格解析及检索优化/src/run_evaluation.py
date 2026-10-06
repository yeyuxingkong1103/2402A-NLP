# -*- coding: utf-8 -*-
"""
14 问 RAG 评估（4 个表格问 + 10 个文本问）
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

评估分两层，缺一不可：

  检索层（确定性，不依赖 LLM，可复现）
    · Hit Rate / MRR / Recall@k   —— rag_core.evaluate.retrieval_metrics
    · 要点覆盖率                   —— 检索上下文里出现了多少答案要点
    · 关键词准确率                 —— rag_core.evaluate.keyword_accuracy

  生成层（RAGAS 方法论，需要 LLM；无 API Key 时自动跳过并明确标注）
    · Faithfulness        忠实度（抗幻觉）
    · Answer Relevancy    答案相关性
    · Context Precision   上下文精度
    · Context Recall      上下文召回
    · Answer Correctness  答案正确性

数字归一化：LLM 常把「1,670万股」写成「1670 万股」，直接子串匹配会误判，
因此判分前统一做「去千分位逗号 / 全角转半角 / 去空白」，见 table_qa.normalize_text。

用法：
    python run_evaluation.py                # 有 API Key 时跑全量指标
    python run_evaluation.py --no-llm       # 只跑检索层指标
    python run_evaluation.py --fast         # 只跑 answer_correctness + context_recall
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import config, evaluate              # noqa: E402
import build_index_with_tables as bi               # noqa: E402
import table_qa as tq                              # noqa: E402

RESULTS_DIR = ROOT / "工单03-表格解析及检索优化" / "results"


# ---------------------------------------------------------------------------
# 1. 文本类问题（兴图新科）答案基准
#    要点均逐字核对《招股说明书1.pdf》原文后固化；
#    must 全部命中才算答对，bonus 为加分项。
# ---------------------------------------------------------------------------
TEXT_GROUND_TRUTH: dict[int, dict] = {
    260: {
        "must": ["6,464.51", "14,414.16", "18,780.67", "4,627.14"],
        "bonus": ["万元"],
        "pages": [4, 129, 343],
        "reference": "报告期内，来自军用领域的收入分别为6,464.51万元、14,414.16万元、"
                     "18,780.67万元和4,627.14万元。",
    },
    95: {
        "must": ["视频指挥系统技术标准"],
        "bonus": ["全军第一个", "1.0"],
        "pages": [26, 27],
        "reference": "参与制定了全军第一个视频指挥系统技术标准（即《某视频技术规范1.0》）。",
    },
    33: {
        "must": ["82.10%", "97.31%", "94.84%", "94.34%"],
        "bonus": ["主营业务收入"],
        "pages": [4, 129, 343],
        "reference": "来自军用领域的收入占主营业务收入的比重分别为82.10%、97.31%、"
                     "94.84%和94.34%。",
    },
    34: {
        "must": ["电子元器件", "机箱", "金属壳体"],
        "bonus": ["竞争充分"],
        "pages": [152],
        "reference": "电子信息行业的上游涉及信息系统相关的电子元器件制造企业，"
                     "以及机箱、机柜等金属壳体制造企业。",
    },
    957: {
        "must": ["重要供应商"],
        "bonus": ["国防军队视频指挥领域", "军队视频指挥领域"],
        "pages": [26, 27, 154],
        "reference": "兴图新科目前已经成为国防军队视频指挥领域的重要供应商。",
    },
    793: {
        "must": ["军队", "政府机关", "能源"],
        "bonus": ["覆盖范围广泛"],
        "pages": [152],
        "reference": "电子信息行业的下游为各类终端用户，覆盖范围广泛，"
                     "主要包括军队、政府机关、能源等行业企业。",
    },
    795: {
        "must": ["国家科技进步一等奖"],
        "bonus": ["一体化工程", "C4ISR"],
        "pages": [27, 94, 96, 155],
        "reference": "2014年12月，某大型研究所牵头承担的「某情报、指挥、控制与通信"
                     "网络一体化工程」（即相当于美军的C4ISR系统）荣获国家科技进步一等奖，"
                     "兴图新科是该工程网络化视频指挥系统的唯一参与者。",
    },
    543: {
        "must": ["5,520"],
        "bonus": ["万元"],
        "pages": [22, 52],
        "reference": "注册资本为人民币5,520.00万元。",
    },
    531: {
        "must": ["程家明"],
        "bonus": [],
        "pages": [22, 52],
        "reference": "法定代表人为程家明。",
    },
    207: {
        "must": ["15,000"],
        "bonus": ["补充流动资金"],
        "pages": [30, 479],
        "reference": "计划使用本次发行募集资金15,000.00万元用于补充流动资金"
                     "（募集资金投资项目合计40,584.83万元）。",
    },
}


def all_questions() -> list[dict]:
    """14 个问题的统一结构：{group, id, question, ground_truth, must, pages}。"""
    out = []
    for q in tq.TABLE_QUESTIONS:
        gt = tq.TABLE_GROUND_TRUTH[q["id"]]
        out.append({"group": "table", "id": q["id"], "question": q["question"],
                    "ground_truth": gt["reference"], "must": gt["must"],
                    "reference_doc": "招股说明书2", "pages": gt["pages"]})
    for q in config.QUESTIONS_XINGTU:
        gt = TEXT_GROUND_TRUTH[q["id"]]
        out.append({"group": "text", "id": q["id"], "question": q["question"],
                    "ground_truth": gt["reference"], "must": gt["must"],
                    "reference_doc": "招股说明书1", "pages": gt["pages"]})
    return out


# ---------------------------------------------------------------------------
# 2. 跑问答，构造 EvalRecord
# ---------------------------------------------------------------------------
def collect_records(collection: str = bi.COLLECTION_WITH_TABLES,
                    top_k: int = 5, use_llm: bool = True,
                    verbose: bool = True) -> list[evaluate.EvalRecord]:
    retriever = bi.load_retriever(collection)
    records: list[evaluate.EvalRecord] = []

    for q in all_questions():
        t0 = time.perf_counter()
        res = bi.answer_with_tables(q["question"], retriever, top_k=top_k,
                                    use_llm=use_llm)
        latency = time.perf_counter() - t0
        docs = res["docs"]
        rec = evaluate.EvalRecord(
            qid=q["id"], question=q["question"], answer=res["answer"],
            ground_truth=q["ground_truth"],
            contexts=[d.get("text", "") for d in docs],
            reference_doc=q["reference_doc"],
            retrieved_docs=[d.get("doc", "") for d in docs],
            retrieved_pages=[d.get("page", 0) for d in docs],
            latency=latency,
        )
        records.append(rec)
        if verbose:
            print(f"  [{len(records):>2}/{len(all_questions())}] "
                  f"id={q['id']:<4} 检索 {res['timings']['retrieve'] * 1000:.0f}ms "
                  f"总 {latency:.2f}s")
    return records


# ---------------------------------------------------------------------------
# 3. 检索层指标（确定性）+ 关键词准确率
# ---------------------------------------------------------------------------
def retrieval_layer(records: list[evaluate.EvalRecord]) -> dict:
    """
    计算不依赖 LLM 的确定性指标：
      · Hit Rate / MRR / Recall@k（正确文档是否被召回）
      · 要点覆盖率（答案要点在检索上下文中的覆盖比例）
      · 命中页命中率（是否召回了包含答案的那一页）
      · 平均/最大检索耗时
    """
    q_map = {q["id"]: q for q in all_questions()}

    per_q, hit_page, cover = [], 0, 0.0
    lat = []
    for r in records:
        q = q_map[r.qid]
        ctx = "\n".join(r.contexts)
        hit = tq.match_points(ctx, q["must"])
        cov = len(hit) / max(len(q["must"]), 1)
        cover += cov
        page_ok = any(p in r.retrieved_pages for p in q["pages"])
        hit_page += int(page_ok)
        lat.append(r.latency)
        per_q.append({
            "id": r.qid, "group": q["group"], "keypoint_coverage": round(cov, 4),
            "missing": [p for p in q["must"] if p not in hit],
            "page_hit": page_ok,
        })

    n = max(len(records), 1)
    out = {
        "n": len(records),
        "keypoint_coverage": round(cover / n, 4),
        "page_hit_rate": round(hit_page / n, 4),
        "latency_avg_s": round(sum(lat) / n, 3),
        "latency_max_s": round(max(lat, default=0), 3),
        "latency_under_3s": round(sum(1 for x in lat if x <= 3.0) / n, 4),
        "per_question": per_q,
    }
    out.update(evaluate.retrieval_metrics(records))
    return out


def keyword_layer(records: list[evaluate.EvalRecord]) -> dict:
    """
    关键词式「准确率」（工单要求的 90% 口径）：
    每条问题的 must 要点必须全部出现在答案里才算答对。
    两边都先做数字归一化，避免格式差异造成的假阴性。
    """
    expected = {str(q["id"]): [tq.normalize_text(p) for p in q["must"]]
                for q in all_questions()}
    norm_records = [
        evaluate.EvalRecord(qid=r.qid, question=r.question,
                            answer=tq.normalize_text(r.answer))
        for r in records
    ]
    return evaluate.keyword_accuracy(norm_records, expected)


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
def run_evaluation(collection: str = bi.COLLECTION_WITH_TABLES,
                   top_k: int = 5, use_llm: bool = True,
                   metrics: list[str] | None = None,
                   out_dir: Path | None = None,
                   verbose: bool = True) -> dict:
    out_dir = Path(out_dir or RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if verbose:
        print(f"[1/3] 检索问答（collection={collection}，use_llm={use_llm}）…")
    records = collect_records(collection, top_k=top_k, use_llm=use_llm,
                              verbose=verbose)

    if verbose:
        print("[2/3] 检索层指标…")
    ret_layer = retrieval_layer(records)
    kw_layer = keyword_layer(records)

    ragas_summary = None
    if use_llm:
        metrics = metrics or ["faithfulness", "answer_relevancy",
                              "context_precision", "context_recall",
                              "answer_correctness"]
        if verbose:
            print(f"[3/3] RAGAS 指标：{'、'.join(metrics)}…")
        try:
            ragas_summary = evaluate.evaluate_records(records, metrics=metrics,
                                                      verbose=verbose)
        except Exception as e:
            print(f"  [warn] RAGAS 评估失败（{type(e).__name__}: {e}），仅保留检索层结果")
    elif verbose:
        print("[3/3] 未启用 LLM，跳过 RAGAS 指标（检索层结果不受影响）")

    summary = {
        "n_questions": len(records),
        "collection": collection,
        "top_k": top_k,
        "use_llm": use_llm,
        "keyword_accuracy": kw_layer["accuracy"],
        "keyword_correct": kw_layer["correct"],
        "keypoint_coverage": ret_layer["keypoint_coverage"],
        "page_hit_rate": ret_layer["page_hit_rate"],
        "hit_rate": ret_layer.get("hit_rate"),
        "mrr": ret_layer.get("mrr"),
        "latency_avg_s": ret_layer["latency_avg_s"],
        "latency_max_s": ret_layer["latency_max_s"],
        "latency_under_3s_ratio": ret_layer["latency_under_3s"],
        "ragas": ragas_summary,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }

    payload = {
        "meta": {
            "工单": "人工智能NLP-RAG-PDF文档的表格解析及检索优化",
            "问题构成": {"表格类(id 1-4)": 4, "文本类(兴图新科)": 10, "合计": len(records)},
            "判分口径": "归一化关键词匹配；must 要点全中即判对",
            "归一化规则": "去千分位逗号 + 全角转半角 + 去空白",
        },
        "summary": summary,
        "retrieval_layer": ret_layer,
        "keyword_details": kw_layer["details"],
        "records": [r.to_dict() for r in records],
    }

    (out_dir / "evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, out_dir / "evaluation.md")

    if verbose:
        print(f"\n准确率(要点全中) {summary['keyword_accuracy']:.2%}"
              f"（{summary['keyword_correct']}/{summary['n_questions']}）"
              f" | 要点覆盖率 {summary['keypoint_coverage']:.2%}"
              f" | 命中页率 {summary['page_hit_rate']:.2%}"
              f" | 平均耗时 {summary['latency_avg_s']}s"
              f" | 3 秒内占比 {summary['latency_under_3s_ratio']:.0%}")
        if ragas_summary:
            print(evaluate.format_summary(ragas_summary))
        print(f"报告：{out_dir / 'evaluation.md'}")
    return payload


def _write_markdown(p: dict, path: Path) -> None:
    s = p["summary"]
    rl = p["retrieval_layer"]
    lines = [
        "# 14 问 RAG 评估报告",
        "",
        "> 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化",
        "",
        "## 一、评估口径",
        "",
        "- **准确率**：每条问题预设 must 要点，答案全部命中才判对；",
        "  判分前做数字归一化（`1,670万股` 与 `1670万股` 视为相同），避免格式假阴性。",
        "- **检索层指标**：要点覆盖率、命中页率、Hit Rate / MRR / Recall@k，"
        "全部确定性计算，不依赖 LLM。",
        "- **生成层指标**：RAGAS 四大指标 + 答案正确性"
        f"（{'已启用' if s['use_llm'] else '未启用（无 API Key，已跳过）'}）。",
        "",
        "## 二、总体结果",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 问题数 | {s['n_questions']}（表格类 4 + 文本类 10） |",
        f"| **准确率（要点全中）** | **{s['keyword_accuracy']:.2%}**"
        f"（{s['keyword_correct']}/{s['n_questions']}） |",
        f"| 检索要点覆盖率 | {s['keypoint_coverage']:.2%} |",
        f"| 命中页率（召回包含答案的页） | {s['page_hit_rate']:.2%} |",
        f"| Hit Rate | {s['hit_rate']} |",
        f"| MRR | {s['mrr']} |",
        f"| 平均端到端耗时 | {s['latency_avg_s']} s |",
        f"| 最大端到端耗时 | {s['latency_max_s']} s |",
        f"| 3 秒内完成占比 | {s['latency_under_3s_ratio']:.2%} |",
        "",
    ]
    if s.get("ragas"):
        lines += [
            "### RAGAS 生成层指标",
            "",
            "```",
            evaluate.format_summary(s["ragas"]),
            "```",
            "",
        ]
    lines += [
        "## 三、逐题明细",
        "",
        "| 分组 | id | 问题（截断） | 要点覆盖 | 命中页 | 是否全中 | 缺失要点 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    per_q = {x["id"]: x for x in rl["per_question"]}
    kw = {d["id"]: d for d in p["keyword_details"]}
    for r in p["records"]:
        g = per_q.get(r["id"], {})
        k = kw.get(r["id"], {})
        lines.append(
            f"| {g.get('group', '')} | {r['id']} | {r['question'][:30]}… "
            f"| {g.get('keypoint_coverage', 0):.0%} "
            f"| {'✓' if g.get('page_hit') else '✗'} "
            f"| {'✓' if k.get('正确') else '✗'} "
            f"| {'、'.join(k.get('漏答', [])) or '—'} |")
    lines += [
        "",
        "## 四、结论",
        "",
        f"1. 14 问准确率 {s['keyword_accuracy']:.2%}，"
        f"{'满足' if s['keyword_accuracy'] >= 0.9 else '未达到'}"
        f"工单「准确率 90% 以上」的要求。",
        f"2. 检索要点覆盖率 {s['keypoint_coverage']:.2%}、命中页率 {s['page_hit_rate']:.2%}，"
        f"说明答案所需信息在检索阶段已被召回，剩余误差来自生成阶段的表述。",
        f"3. 3 秒内完成占比 {s['latency_under_3s_ratio']:.2%}，"
        f"检索侧耗时（毫秒级）远低于约束，端到端耗时由生成模型决定。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="工单03 十四问 RAG 评估")
    ap.add_argument("--collection", default=bi.COLLECTION_WITH_TABLES)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-llm", action="store_true",
                    help="不调用生成模型，且跳过 RAGAS 指标")
    ap.add_argument("--fast", action="store_true",
                    help="只评估 answer_correctness + context_recall（省时省钱）")
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args()

    metrics = ["answer_correctness", "context_recall"] if args.fast else None
    run_evaluation(collection=args.collection, top_k=args.top_k,
                   use_llm=not args.no_llm, metrics=metrics,
                   out_dir=Path(args.out))


if __name__ == "__main__":
    main()
