# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估

RAG 评估脚本 —— 用 rag_core.evaluate 的评估框架给 10 个问题打分。

评估分三层：

  1. 生成质量层（RAGAS 四大指标 + 答案正确性）
       faithfulness      忠实度      答案的论断有多少能被检索上下文支撑（抗幻觉）
       answer_relevancy  答案相关性  答案是否切题（由答案反推问题再算相似度）
       context_precision 上下文精度  有用的片段是否排在前面
       context_recall    上下文召回  参考答案的信息有多少能从检索上下文里找到
       answer_correctness 答案正确性 0.7×事实重叠 + 0.3×语义相似

  2. 检索层（纯计算，不依赖 LLM）
       hit_rate / mrr / recall_at_k（evaluate.retrieval_metrics）
       + 文档覆盖率 doc_coverage（多文档题专用）
       + 期望页码命中率 page_hit_rate（答案所在页是否被召回）

  3. 关键词判准层（answers_expected）
       每题的「答案必须包含的关键信息点」是否全部命中（数值题尤其有效）

逐题判定规则 PASS_RULES（三档，全部满足才算通过）：
       · 检索档：命中主责文档
       · 召回档：context_recall ≥ 0.60
       · 生成档：answer_correctness ≥ 0.50 且 faithfulness ≥ 0.70

用法：
    python run_evaluation.py                    # 完整评估（默认内置实现）
    python run_evaluation.py --use-ragas        # 装了官方 ragas 时改走官方实现
    python run_evaluation.py --retrieval-only   # 只算检索层与关键词，不调 LLM
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import config, evaluate  # noqa: E402
from rag_core.evaluate import (EvalRecord, evaluate_records, format_summary,  # noqa: E402
                               keyword_accuracy, retrieval_metrics)
from prepare_corpus import WO_NO, ensure_results_dir  # noqa: E402

#: 逐题判定阈值（判读标准见 docs/评估方法说明.md）
PASS_RULES = {
    "require_hit_reference_doc": True,
    "min_context_recall": 0.60,
    "min_answer_correctness": 0.50,
    "min_faithfulness": 0.70,
}

DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision",
                   "context_recall", "answer_correctness"]


# ---------------------------------------------------------------------------
# 组装 EvalRecord
# ---------------------------------------------------------------------------
def build_records(test_payload: dict, rag_payload: dict) -> list[EvalRecord]:
    """把「测试用例」与「RAG 测试结果」合并成评估记录。"""
    cases = {c["id"]: c for c in test_payload["cases"]}
    records: list[EvalRecord] = []
    for r in rag_payload["results"]:
        c = cases.get(r["id"], {})
        contexts = [d.get("text", "") for d in r.get("retrieved", []) if d.get("text")]
        rec = EvalRecord(
            qid=r["id"],
            question=r["question"],
            answer=r.get("answer", ""),
            ground_truth=c.get("ground_truth", ""),
            contexts=contexts,
            reference_doc=r.get("reference_doc", ""),
            retrieved_docs=r.get("retrieved_docs", []),
            retrieved_pages=r.get("retrieved_pages", []),
            latency=float(r.get("latency", 0.0)),
        )
        # 把用例侧信息挂到 extra 上（EvalRecord 允许动态属性）
        rec.extra = {                       # type: ignore[attr-defined]
            "question_type": r.get("question_type", ""),
            "difficulty": r.get("difficulty", ""),
            "origin": r.get("origin", ""),
            "answer_keywords": c.get("answer_keywords", []),
            "expected_pages": r.get("expected_pages", []),
            "page_hits": r.get("page_hits", []),
            "doc_coverage": r.get("doc_coverage", 0.0),
            "n_docs_covered": r.get("n_docs_covered", 0),
            "n_docs_expected": r.get("n_docs_expected", 1),
            "n_table_chunks": r.get("n_table_chunks", 0),
            "hit_reference_doc": r.get("hit_reference_doc", False),
            "rank_of_reference_doc": r.get("rank_of_reference_doc", 0),
            "refused": r.get("refused", False),
            "need_table": r.get("need_table", False),
        }
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# 逐题判定
# ---------------------------------------------------------------------------
def judge(rec: EvalRecord) -> tuple[bool, list[str]]:
    """按 PASS_RULES 判定单题是否通过，返回 (是否通过, 未通过原因)。"""
    reasons: list[str] = []
    extra = getattr(rec, "extra", {})
    m = rec.metrics

    if PASS_RULES["require_hit_reference_doc"] and not extra.get("hit_reference_doc"):
        reasons.append("未检索到主责文档")
    cr = m.get("context_recall")
    if cr is None or (isinstance(cr, float) and math.isnan(cr)) or cr < PASS_RULES["min_context_recall"]:
        reasons.append(f"上下文召回 {_fmt(cr)} < {PASS_RULES['min_context_recall']}")
    ac = m.get("answer_correctness")
    if ac is None or (isinstance(ac, float) and math.isnan(ac)) or ac < PASS_RULES["min_answer_correctness"]:
        reasons.append(f"答案正确性 {_fmt(ac)} < {PASS_RULES['min_answer_correctness']}")
    fa = m.get("faithfulness")
    if fa is not None and not (isinstance(fa, float) and math.isnan(fa)) and fa < PASS_RULES["min_faithfulness"]:
        reasons.append(f"忠实度 {_fmt(fa)} < {PASS_RULES['min_faithfulness']}")
    if extra.get("refused"):
        reasons.append("模型拒答")
    return (len(reasons) == 0), reasons


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return f"{v:.3f}"


# ---------------------------------------------------------------------------
# 分组统计
# ---------------------------------------------------------------------------
def group_stats(records: list[EvalRecord], metrics: list[str]) -> dict:
    """按题型 / 难度分组统计指标均值。"""
    out: dict[str, dict] = {}
    for key in ("question_type", "difficulty"):
        buckets: dict[str, list[EvalRecord]] = {}
        for rec in records:
            g = getattr(rec, "extra", {}).get(key, "未知")
            buckets.setdefault(g, []).append(rec)
        out[key] = {}
        for g, rs in buckets.items():
            row = {"n": len(rs)}
            for m in metrics:
                vals = [r.metrics.get(m) for r in rs]
                vals = [v for v in vals if v is not None
                        and not (isinstance(v, float) and math.isnan(v))]
                row[m] = round(sum(vals) / len(vals), 4) if vals else None
            row["hit_rate"] = round(
                sum(1 for r in rs if getattr(r, "extra", {}).get("hit_reference_doc"))
                / len(rs), 4)
            out[key][g] = row
    return out


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def build_payload(records: list[EvalRecord], summary: dict, kw: dict,
                  metrics: list[str], retrieval_only: bool) -> dict:
    judged = []
    for r in records:
        ok, reasons = judge(r)
        extra = getattr(r, "extra", {})
        judged.append({
            "id": r.qid, "question": r.question, "question_type": extra.get("question_type"),
            "difficulty": extra.get("difficulty"), "origin": extra.get("origin"),
            "reference_doc": r.reference_doc, "reference_docs_note": (
                f"覆盖 {extra.get('n_docs_covered')}/{extra.get('n_docs_expected')} 份文档"),
            "hit_reference_doc": extra.get("hit_reference_doc"),
            "rank_of_reference_doc": extra.get("rank_of_reference_doc"),
            "page_hits": extra.get("page_hits"), "expected_pages": extra.get("expected_pages"),
            "n_table_chunks": extra.get("n_table_chunks"), "need_table": extra.get("need_table"),
            "answer": r.answer, "ground_truth": r.ground_truth,
            "metrics": {k: (round(v, 4) if isinstance(v, float) and not math.isnan(v) else None)
                        for k, v in r.metrics.items()},
            "keywords": next((d for d in kw.get("details", []) if d["id"] == r.qid), None),
            "passed": ok, "fail_reasons": reasons,
        })
    n_pass = sum(1 for j in judged if j["passed"])
    return {
        "wo_no": WO_NO,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "llm_available": bool(config.DEEPSEEK_API_KEY),
        "retrieval_only": retrieval_only,
        "metrics_requested": metrics,
        "pass_rules": PASS_RULES,
        "summary": summary,
        "keyword_accuracy": {k: v for k, v in kw.items() if k != "details"},
        "group_stats": group_stats(records, metrics),
        "pass": {"n_pass": n_pass, "n_total": len(judged),
                 "pass_rate": round(n_pass / max(len(judged), 1), 4)},
        "records": judged,
    }


def render_md(payload: dict) -> str:
    s = payload["summary"]
    lines = [
        f"# RAG 评估结果（{s['n']} 个问题）",
        "",
        f"> 工单编号：{WO_NO}　生成时间：{payload['generated_at']}",
        f"> 评估器：`{s.get('evaluator', 'builtin')}`　"
        f"LLM 可用：{'是' if payload['llm_available'] else '**否（已自动降级为规则近似实现）**'}",
        "",
        "## 一、评估指标汇总（RAGAS 四大指标 + 正确性 + 检索层）",
        "",
        "```",
        format_summary(s),
        "```",
        "",
        f"**关键词判准准确率**：{payload['keyword_accuracy'].get('accuracy', 0):.2%}"
        f"（{payload['keyword_accuracy'].get('correct', 0)}/{payload['keyword_accuracy'].get('total', 0)} 题关键信息点全中）",
        "",
        f"**综合判定通过率**：{payload['pass']['pass_rate']:.2%}"
        f"（{payload['pass']['n_pass']}/{payload['pass']['n_total']}）",
        "",
        "**判定规则**：" + "；".join(f"{k}={v}" for k, v in payload["pass_rules"].items()),
        "",
        "## 二、逐题评估结果",
        "",
        "| 编号 | 题型 | 命中 | Faith. | Ans.Rel. | Ctx.Prec. | Ctx.Rec. | Ans.Corr. | 关键词 | 判定 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in payload["records"]:
        m = r["metrics"]
        kw = r["keywords"] or {}
        kw_s = f"{len(kw.get('命中', []))}/{len(kw.get('命中', [])) + len(kw.get('漏答', []))}"
        lines.append(
            f"| {r['id']} | {r['question_type']} | "
            f"{'✓' if r['hit_reference_doc'] else '✗'} | "
            f"{_f(m.get('faithfulness'))} | {_f(m.get('answer_relevancy'))} | "
            f"{_f(m.get('context_precision'))} | {_f(m.get('context_recall'))} | "
            f"{_f(m.get('answer_correctness'))} | {kw_s} | "
            f"{'通过' if r['passed'] else '**未通过**'} |")

    lines += ["", "## 三、分组统计", ""]
    for key, label in (("question_type", "按题型"), ("difficulty", "按难度")):
        lines += [f"### {label}", "",
                  "| 分组 | n | Faith. | Ans.Rel. | Ctx.Prec. | Ctx.Rec. | Ans.Corr. | 检索命中率 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for g, row in payload["group_stats"][key].items():
            lines.append(
                f"| {g} | {row['n']} | {_f(row.get('faithfulness'))} | "
                f"{_f(row.get('answer_relevancy'))} | {_f(row.get('context_precision'))} | "
                f"{_f(row.get('context_recall'))} | {_f(row.get('answer_correctness'))} | "
                f"{row['hit_rate']:.2%} |")
        lines.append("")

    lines += ["## 四、未通过题目与原因", ""]
    bad = [r for r in payload["records"] if not r["passed"]]
    if not bad:
        lines.append("全部题目通过。")
    else:
        for r in bad:
            lines.append(f"- **{r['id']}**（{r['question_type']}）："
                         f"{'；'.join(r['fail_reasons'])}")
    lines.append("")
    return "\n".join(lines)


def _f(v) -> str:
    return "—" if v is None else f"{v:.3f}"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=f"RAG 评估（{WO_NO}）")
    ap.add_argument("--test-cases", default="results/test_cases.json")
    ap.add_argument("--rag-results", default="results/rag_test_results.json")
    ap.add_argument("--use-ragas", action="store_true", help="使用官方 ragas 包（需已安装）")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="只算检索层指标与关键词命中，跳过需要 LLM/嵌入的指标")
    ap.add_argument("--metrics", nargs="*", default=None)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    cases_path = base / args.test_cases
    rag_path = base / args.rag_results
    for p in (cases_path, rag_path):
        if not p.exists():
            raise SystemExit(f"缺少输入文件：{p}\n请先依次运行 build_questions.py、run_rag_test.py")

    test_payload = json.loads(cases_path.read_text(encoding="utf-8"))
    rag_payload = json.loads(rag_path.read_text(encoding="utf-8"))

    print(f"===== {WO_NO} · RAG 评估 =====")
    records = build_records(test_payload, rag_payload)
    print(f"载入 {len(records)} 条问答记录")

    metrics = args.metrics or DEFAULT_METRICS
    if args.retrieval_only:
        print("[模式] 仅检索层指标（不调用 LLM）")
        summary = {"n": len(records), "evaluator": "retrieval-only"}
        summary.update(retrieval_metrics(records))
        lat = [r.latency for r in records if r.latency > 0]
        if lat:
            summary["latency_avg"] = round(sum(lat) / len(lat), 3)
            summary["latency_max"] = round(max(lat), 3)
    else:
        if not config.DEEPSEEK_API_KEY:
            print("[warn] 未检测到 DEEPSEEK_API_KEY，评估将自动降级为规则近似实现"
                  "（论断分解回退分句、支撑判断回退关键词覆盖）")
        print(f"[评估] 指标：{'、'.join(metrics)}"
              f"{'（官方 ragas）' if args.use_ragas else '（内置实现）'}")
        summary = evaluate_records(records, metrics=metrics,
                                   use_ragas=args.use_ragas, verbose=True)

    # 关键词判准（自动「准确率」口径）
    expected = {c["id"]: c.get("answer_keywords", []) for c in test_payload["cases"]}
    kw = keyword_accuracy(records, expected)

    payload = build_payload(records, summary, kw, metrics, args.retrieval_only)

    out_dir = ensure_results_dir()
    # evaluation.json = 汇总 + 逐题判定 + 分组统计 + 评估框架原始记录（便于二次分析）
    (out_dir / "evaluation.json").write_text(
        json.dumps({**payload, "raw_summary": summary,
                    "records_raw": [r.to_dict() for r in records]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "evaluation.md").write_text(render_md(payload), encoding="utf-8")

    print("\n" + format_summary(summary))
    print(f"\n关键词判准准确率：{kw['accuracy']:.2%}（{kw['correct']}/{kw['total']}）")
    print(f"综合判定通过率：{payload['pass']['pass_rate']:.2%}"
          f"（{payload['pass']['n_pass']}/{payload['pass']['n_total']}）")
    print(f"评估结果已保存 → {out_dir / 'evaluation.json'} / evaluation.md")


if __name__ == "__main__":
    main()
