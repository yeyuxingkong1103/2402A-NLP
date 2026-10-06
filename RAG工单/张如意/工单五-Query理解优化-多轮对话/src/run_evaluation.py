# -*- coding: utf-8 -*-
"""
工单05 多轮对话 RAG 评估
工单编号：人工智能NLP-RAG-Query理解优化任务

对 5 轮问答做两层评估：

1. 检索与答案口径（客观、离线可复现）
   · 检索命中：期望文档是否进 Top-k、考察点是否被片段覆盖（跨文档能力）
   · 答案准确率：关键词命中式判定（与工单「准确率 90% 以上」的口径一致）
   · 响应时间：每轮耗时与 3 秒达标率、阶段耗时拆解

2. RAGAS 风格指标（复用 rag_core.evaluate 的实现）
   · Faithfulness / Answer Relevancy / Context Precision / Context Recall /
     Answer Correctness
   · 逐指标 try/except 降级：缺少 LLM Key 或嵌入模型时标记「不可用」而不是中断评估

另外做一项「多语言支持」自检：验证提问语言识别（中/英）与答案语言一致策略。

输出：
    results/evaluation.json   逐轮明细 + 指标汇总（evaluate.save_report 落盘）
    results/evaluation.md     人读报告

运行：
    python run_evaluation.py                 # 完整评估
    python run_evaluation.py --fast          # tfidf 重排，跑得更快
    python run_evaluation.py --no-ragas      # 只跑客观口径，不调用 LLM 指标
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import config, evaluate, generator                       # noqa: E402

from wo05_common import (                                              # noqa: E402
    COLLECTION, RESULT_DIR, build_pipeline, index_ready, latency_summary,
    load_ground_truth, merged_spec, print_llm_usage, run_script,
    write_json, write_md,
)

RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision",
                 "context_recall", "answer_correctness"]


def run_and_collect(reranker: str | None, collection: str) -> tuple[list[dict], list]:
    """跑 5 轮对话，构造 EvalRecord 列表。"""
    pipeline = build_pipeline(use_query_understanding=True,
                              reranker=reranker, collection=collection)
    questions = list(config.MULTI_TURN_SCRIPT)
    records = run_script(pipeline, questions, top_k_show=3, verbose=True)

    ground_truth = load_ground_truth()
    eval_records = []
    for rec in records:
        spec = merged_spec(rec["turn"], ground_truth)
        docs = rec.get("docs") or []
        ref = spec.get("reference_answer") or (
            f"答案应包含关键信息点：{'、'.join(spec['answer_keywords'])}；"
            f"信息应来自《{spec['expected_doc']}》。"
        )
        er = evaluate.EvalRecord(
            qid=rec["turn"],
            question=rec["question"],
            answer=rec.get("answer") or "",
            ground_truth=ref,
            contexts=[d.get("text", "") for d in docs],
            reference_doc=spec["expected_doc"],
            retrieved_docs=[str(d.get("doc", "")) for d in docs],
            retrieved_pages=[int(d["page"]) for d in docs
                             if isinstance(d.get("page"), int)],
            latency=rec.get("latency", 0.0),
        )
        eval_records.append(er)
        rec["期望文档"] = spec["expected_doc"]
        rec["考察能力"] = spec["exam"]
    return records, eval_records


def safe_metrics(eval_records: list, metrics: list[str]) -> dict:
    """
    逐指标调用 rag_core.evaluate 的实现，单项失败只标记该指标不可用。
    这样即使没配 LLM Key 或没下载嵌入模型，评估报告仍然完整可读。
    """
    fn_map = {
        "faithfulness": lambda r: evaluate.faithfulness(r.answer, r.contexts),
        "answer_relevancy": lambda r: evaluate.answer_relevancy(r.question, r.answer),
        "context_precision": lambda r: evaluate.context_precision(
            r.question, r.contexts, r.ground_truth),
        "context_recall": lambda r: evaluate.context_recall(r.ground_truth, r.contexts),
        "answer_correctness": lambda r: evaluate.answer_correctness(
            r.answer, r.ground_truth),
    }
    unavailable: dict[str, str] = {}

    for name in metrics:
        fn = fn_map.get(name)
        if fn is None:
            continue
        for r in eval_records:
            try:
                r.metrics[name] = float(fn(r))
            except Exception as e:                    # 降级：该指标整列不可用
                unavailable[name] = f"{type(e).__name__}: {e}"
                for rr in eval_records:
                    rr.metrics.pop(name, None)
                break

    # 部分记录失败（个别轮次异常）时，也如实标注，避免把不完整均值当完整结果
    for name in metrics:
        if name in unavailable or name not in fn_map:
            continue
        got = sum(1 for r in eval_records if name in r.metrics)
        if got < len(eval_records):
            unavailable[name] = f"仅 {got}/{len(eval_records)} 轮次可计算（部分记录异常）"

    summary = evaluate.aggregate(eval_records, metrics)
    if unavailable:
        summary["不可用指标"] = unavailable
    return summary


def multilingual_check() -> list[dict]:
    """
    多语言支持自检：验证「提问语言识别」是否工作（答案语言跟随提问语言）。
    generator._is_english 是 rag_core.generator 内部的语言判定函数。
    """
    samples = [
        ("武汉兴图新科电子股份有限公司法定代表人是谁？", False),
        ("Who is the legal representative of Wuhan Xingtu Xinke Electronics?", True),
        ("What are the military revenue figures in the reporting period?", True),
        ("报告期内军用领域收入是多少？", False),
    ]
    out = []
    for q, expect_en in samples:
        actual = generator._is_english(q)
        out.append({"问题": q, "识别为英文": actual,
                    "期望": expect_en, "通过": actual == expect_en})
    return out


def render_md(payload: dict) -> str:
    """渲染评估报告。"""
    s = payload["ragas_summary"]
    lat = payload["latency"]
    kw = payload["keyword_accuracy"]
    ml = payload["多语言自检"]

    lines = [
        "# 工单05 多轮对话评估报告",
        "",
        "> 工单编号：人工智能NLP-RAG-Query理解优化任务  ",
        f"> 问题集：`config.MULTI_TURN_SCRIPT`（5 轮，跨《招股说明书1》《招股说明书2》）  ",
        f"> 评估器：rag_core.evaluate（自实现 RAGAS 口径）；单指标失败自动降级",
        "",
        "## 一、核心结论",
        "",
        f"- 关键词命中式答案准确率：**{kw['accuracy']:.1%}**"
        f"（{kw['correct']}/{kw['total']}，工单要求 ≥ 90%）",
        f"- 检索命中率 Hit Rate：**{s.get('hit_rate', 0):.1%}**；"
        f"MRR：{s.get('mrr', 0):.3f}",
        f"- 响应时间：平均 {lat['平均耗时(s)']}s，最大 {lat['最大耗时(s)']}s，"
        f"3 秒达标率 **{lat['3秒达标率']:.0%}**",
        "",
        "## 二、逐轮明细",
        "",
        "| 轮次 | 考察能力 | 期望文档 | 检索到的文档 | 答案命中关键点 | 判定 | 耗时(s) |",
        "|------|----------|----------|--------------|----------------|------|---------|",
    ]
    for d in kw["details"]:
        r = payload["turns"][d["id"] - 1]
        docs = "、".join(sorted(set(r.get("检索文档", [])))) or "（无）"
        lines.append(f"| {d['id']} | {r.get('考察能力', '')} | {r.get('期望文档', '')} | "
                     f"{docs} | {'、'.join(d['命中']) or '（无）'} | "
                     f"{'正确' if d['正确'] else '错误'} | {r.get('latency', 0):.3f} |")
    lines += ["", "### 关键指标汇总（RAGAS 口径）", "", "```",
              evaluate.format_summary(s), "```", ""]

    lines += ["## 三、多语言支持自检", "",
              "| 问题 | 识别为英文 | 期望 | 结果 |",
              "|------|-----------|------|------|"]
    for m in ml:
        lines.append(f"| {m['问题']} | {m['识别为英文']} | {m['期望']} | "
                     f"{'通过' if m['通过'] else '失败'} |")
    lines += [
        "",
        "说明：提问语言识别通过 `rag_core.generator._is_english` 完成，"
        "英文提问会切换英文 System Prompt，答案语言与提问语言一致。",
        "注意：检索语料为中文 PDF，英文提问需包含中文实体名（如公司全称）才能稳定召回。",
        "",
        "## 四、口径说明与改进方向",
        "",
        f"- 答案判定口径：{payload['meta']['判定口径']}",
        "- 若需更严格的参考答案口径，请创建 `results/ground_truth.json`，"
        "为每轮提供 `reference_answer`（全文），"
        "Context Recall / Answer Correctness 将自动改用该参考答案计算。",
        "- 不可用指标（缺 LLM Key / 嵌入模型）会在 summary 的「不可用指标」字段中标出，"
        "不影响其它指标。",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="工单05 多轮对话 RAG 评估")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--fast", action="store_true", help="等价于 --reranker tfidf")
    ap.add_argument("--no-ragas", action="store_true",
                    help="只跑客观口径（关键词/检索/耗时），不调用 LLM 指标")
    args = ap.parse_args()

    if not index_ready(args.collection):
        print("[提示] 未检测到索引，请先运行：python build_index.py")
        return 2

    t0 = time.perf_counter()
    records, eval_records = run_and_collect(
        "tfidf" if args.fast else args.reranker, args.collection)

    # ---- 1. 客观口径 ----
    ground_truth = load_ground_truth()
    answers_expected = {
        str(spec["turn"]): spec["answer_keywords"]
        for spec in (merged_spec(i, ground_truth) for i in range(1, 6))
    }
    kw = evaluate.keyword_accuracy(eval_records, answers_expected)

    # ---- 2. RAGAS 风格指标（可降级） ----
    metrics = [] if args.no_ragas else list(RAGAS_METRICS)
    summary = safe_metrics(eval_records, metrics) if metrics else evaluate.aggregate(
        eval_records, [])

    # ---- 3. 耗时与多语言 ----
    lat = latency_summary(records)
    ml = multilingual_check()

    # 逐轮补充用于 Markdown 的字段
    for rec, er in zip(records, eval_records):
        rec["检索文档"] = er.retrieved_docs

    print("\n" + "=" * 78)
    print("评估结论")
    print("=" * 78)
    print(f"关键词命中式准确率：{kw['accuracy']:.1%}（{kw['correct']}/{kw['total']}）")
    print(f"检索命中率：{summary.get('hit_rate', 0):.1%}    MRR：{summary.get('mrr', 0):.3f}")
    print(f"响应时间：平均 {lat['平均耗时(s)']}s / 最大 {lat['最大耗时(s)']}s / "
          f"3 秒达标率 {lat['3秒达标率']:.0%}")
    if metrics:
        print(evaluate.format_summary(summary))
    if summary.get("不可用指标"):
        print(f"[提示] 以下指标不可用（已降级）：{summary['不可用指标']}")
    bad_ml = [m for m in ml if not m["通过"]]
    print(f"多语言自检：{'全部通过' if not bad_ml else f'{len(bad_ml)} 项失败'}")
    usage = print_llm_usage()

    # ---- 落盘 ----
    report = {
        "meta": {
            "工单编号": "人工智能NLP-RAG-Query理解优化任务",
            "问题集": list(config.MULTI_TURN_SCRIPT),
            "判定口径": "检索命中=期望文档进 Top-k 且片段覆盖考察点；"
                        "答案正确=非拒答+关键信息点齐全+年份数量达标",
            "总耗时(s)": round(time.perf_counter() - t0, 3),
            "LLM用量": usage,
        },
        "keyword_accuracy": kw,
        "ragas_summary": summary,
        "latency": lat,
        "多语言自检": ml,
        "turns": records,
    }
    write_json(RESULT_DIR / "evaluation.json", report)
    evaluate.save_report(eval_records, summary, RESULT_DIR / "evaluation_records.json")
    write_md(RESULT_DIR / "evaluation.md", render_md(report))

    print(f"结果已写入：{RESULT_DIR / 'evaluation.json'}")
    print(f"          {RESULT_DIR / 'evaluation_records.json'}（evaluate.save_report）")
    print(f"          {RESULT_DIR / 'evaluation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
