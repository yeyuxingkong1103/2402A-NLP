"""
优化前后端到端对比（工单2 核心产出）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

工单2 产出物原文：「针对如下问题进行检索，将检索到的答案，对比优化前的检索答案」。
本脚本对同一套 10 道验收题，跑三条链路并逐题并排对比：

    优化前  朴素 RAG（定长切分 + 纯向量检索）
    优化后  本系统（标题感知分块 + 混合检索 + DF 过滤 + 阈值闸门）
    参照组  纯 LLM（无检索）

指标分两层：
  * 「检索层」用**客观规则**指标（关键事实命中率 / 正确答案所在页命中率），
    不依赖评审模型，任何人拿同一份索引都能复现；
  * 「答案层」用 LLM-as-judge（RAGAS 四项 + 答案正确性），评估回答本身的质量。

用法：
    python scripts/compare_optimization.py               # 全量（约 6~10 分钟）
    python scripts/compare_optimization.py --limit 3     # 先跑前 3 题
    python scripts/compare_optimization.py --no-metrics  # 只出答案对比，不打分
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, rag  # noqa: E402
from src import evaluator as ev  # noqa: E402


def _avg(vals):
    nums = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else None


def fact_hit_rate(contexts: list[str], must_have: list[str]) -> tuple[float | None, list[str]]:
    """关键事实命中率 + 漏掉的事实（纯规则，零成本、可复现）。"""
    if not must_have:
        return None, []
    joined = "\n".join(contexts)
    missed = [k for k in must_have if k not in joined]
    return round((len(must_have) - len(missed)) / len(must_have), 4), missed


def run(questions: list[dict], with_metrics: bool, top_k: int, progress=print) -> dict:
    from src.baseline import answer_naive

    cases = []
    t_all = time.perf_counter()

    for i, item in enumerate(questions, 1):
        qid = int(item.get("id", i))
        q = item["question"]
        ref = (item.get("reference") or "").strip()
        must = item.get("must_have") or []
        progress(f"[{i}/{len(questions)}] id={qid} {q[:36]}…")

        c: dict = {"id": qid, "question": q, "reference": ref, "must_have": must}

        # ---- 优化后
        try:
            a = rag.answer(q, top_k=top_k)
            c["opt_answer"] = a.answer
            c["opt_contexts"] = [x["text"] for x in a.citations]
            # 评审要用**真正喂给模型的那份上下文**（含页码表头）。只喂正文会让
            # 评审模型看不见页码，把正确引用判成「无依据」（实测 faithfulness 虚低）。
            c["opt_judge_ctx"] = [a.context_text] if a.context_text else c["opt_contexts"]
            c["opt_timing"] = a.timing
            c["opt_citations"] = a.citations
            c["opt_understanding"] = a.understanding.to_dict() if a.understanding else None
        except Exception as exc:  # noqa: BLE001
            c["opt_answer"] = f"（优化后链路失败：{exc}）"
            c["opt_contexts"] = []
            c["opt_judge_ctx"] = []
            c["opt_timing"] = {}

        # ---- 优化前（朴素 RAG）
        try:
            n = answer_naive(q, top_k=top_k)
            c["naive_answer"] = n["answer"]
            c["naive_contexts"] = n["contexts"]
            c["naive_judge_ctx"] = [n["context_text"]] if n.get("context_text") else n["contexts"]
            c["naive_timing"] = n["timing"]
            c["naive_citations"] = n["citations"]
        except Exception as exc:  # noqa: BLE001
            c["naive_answer"] = f"（优化前链路失败：{exc}）"
            c["naive_contexts"] = []
            c["naive_judge_ctx"] = []
            c["naive_timing"] = {}

        # ---- 纯 LLM 参照组
        try:
            l = rag.answer_llm_only(q)
            c["llm_answer"] = l.answer
            c["llm_timing"] = l.timing
        except Exception as exc:  # noqa: BLE001
            c["llm_answer"] = f"（LLM 失败：{exc}）"
            c["llm_timing"] = {}

        # ---- 检索层客观指标
        c["opt_fact_rate"], c["opt_fact_missed"] = fact_hit_rate(c["opt_contexts"], must)
        c["naive_fact_rate"], c["naive_fact_missed"] = fact_hit_rate(c["naive_contexts"], must)

        # ---- 答案层 LLM 评审
        if with_metrics:
            m: dict = {}
            for tag, ans, ctx in (("opt", c["opt_answer"], c["opt_judge_ctx"]),
                                  ("naive", c["naive_answer"], c["naive_judge_ctx"])):
                m[f"{tag}_faithfulness"] = ev.score_faithfulness(ans, ctx)
                m[f"{tag}_relevancy"] = ev.score_answer_relevancy(q, ans)
                m[f"{tag}_ctx_precision"] = ev.score_context_precision(q, ctx)
                if ref:
                    m[f"{tag}_ctx_recall"] = ev.score_context_recall(q, ctx, ref)
                    m[f"{tag}_correctness"] = ev.score_answer_correctness(q, ans, ref)
            if ref:
                m["llm_correctness"] = ev.score_answer_correctness(q, c["llm_answer"], ref)
                m["llm_relevancy"] = ev.score_answer_relevancy(q, c["llm_answer"])
            c["metrics"] = m
            progress("      优化后 " + _fmt(m, "opt") + " | 优化前 " + _fmt(m, "naive"))

        cases.append(c)

    return {
        "work_order_no": config.WORK_ORDER_NO_OPT,
        "work_order_short": config.WORK_ORDER_SHORT,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_cases": len(cases),
        "top_k": top_k,
        "elapsed_seconds": round(time.perf_counter() - t_all, 1),
        "summary": summarize(cases),
        "cases": cases,
    }


def _fmt(m: dict, tag: str) -> str:
    parts = []
    for k in ("faithfulness", "relevancy", "ctx_precision", "ctx_recall", "correctness"):
        v = (m.get(f"{tag}_{k}") or {}).get("score")
        parts.append(f"{k}={v if v is not None else '-'}")
    return " ".join(parts)


def summarize(cases: list[dict]) -> dict:
    def avg(key, tag=None):
        if tag:
            return _avg([(c.get("metrics") or {}).get(f"{tag}_{key}", {}).get("score") for c in cases])
        return _avg([(c.get("metrics") or {}).get(key, {}).get("score") for c in cases])

    out = {
        # 检索层（客观）
        "opt_fact_rate": _avg([c["opt_fact_rate"] for c in cases]),
        "naive_fact_rate": _avg([c["naive_fact_rate"] for c in cases]),
        "opt_all_facts": round(sum(1 for c in cases if c["opt_fact_rate"] == 1.0) / len(cases), 4),
        "naive_all_facts": round(sum(1 for c in cases if c["naive_fact_rate"] == 1.0) / len(cases), 4),
        # 答案层（LLM 评审）
        "opt_faithfulness": avg("faithfulness", "opt"),
        "naive_faithfulness": avg("faithfulness", "naive"),
        "opt_relevancy": avg("relevancy", "opt"),
        "naive_relevancy": avg("relevancy", "naive"),
        "opt_ctx_precision": avg("ctx_precision", "opt"),
        "naive_ctx_precision": avg("ctx_precision", "naive"),
        "opt_ctx_recall": avg("ctx_recall", "opt"),
        "naive_ctx_recall": avg("ctx_recall", "naive"),
        "opt_correctness": avg("correctness", "opt"),
        "naive_correctness": avg("correctness", "naive"),
        "llm_correctness": avg("correctness", "llm"),
        "llm_relevancy": avg("relevancy", "llm"),
    }
    for tag, key in (("opt", "opt_timing"), ("naive", "naive_timing"), ("llm", "llm_timing")):
        tot = [c.get(key, {}).get("total_ms") for c in cases if c.get(key, {}).get("total_ms")]
        out[f"{tag}_avg_ms"] = _avg(tot)
        out[f"{tag}_max_ms"] = max(tot) if tot else None
        out[f"{tag}_within_3s"] = round(sum(1 for t in tot if t <= 3000) / len(tot), 4) if tot else None
    out["opt_bypassed"] = sum(1 for c in cases if (c.get("opt_understanding") or {}).get("bypassed"))
    return out


def to_markdown(r: dict) -> str:
    s = r["summary"]
    c = r["cases"]
    n = len(c)
    L = [
        "# 优化前后对比报告（工单2）",
        "",
        f"- 工单编号：{r['work_order_no']}",
        f"- 生成时间：{r['generated_at']}",
        f"- 用例数：{n}，top_k={r['top_k']}，总耗时：{r['elapsed_seconds']}s",
        "",
        "## 一、汇总对比",
        "",
        "| 指标 | 优化前（朴素 RAG） | 优化后（本系统） | 纯 LLM（参照） |",
        "|---|---|---|---|",
        f"| 关键事实命中率（检索层，客观） | {s['naive_fact_rate']} | **{s['opt_fact_rate']}** | — |",
        f"| 关键事实全命中的题目比例 | {s['naive_all_facts']} | **{s['opt_all_facts']}** | — |",
        f"| 忠实度 faithfulness | {s['naive_faithfulness']} | **{s['opt_faithfulness']}** | — |",
        f"| 答案相关性 relevancy | {s['naive_relevancy']} | **{s['opt_relevancy']}** | {s['llm_relevancy']} |",
        f"| 上下文精确率 ctx_precision | {s['naive_ctx_precision']} | **{s['opt_ctx_precision']}** | — |",
        f"| 上下文召回率 ctx_recall | {s['naive_ctx_recall']} | **{s['opt_ctx_recall']}** | — |",
        f"| 答案正确性 correctness | {s['naive_correctness']} | **{s['opt_correctness']}** | {s['llm_correctness']} |",
        f"| 平均响应时间 | {s['naive_avg_ms']} ms | **{s['opt_avg_ms']} ms** | {s['llm_avg_ms']} ms |",
        f"| 最大响应时间 | {s['naive_max_ms']} ms | {s['opt_max_ms']} ms | {s['llm_max_ms']} ms |",
        f"| ≤3 秒达标率 | {s['naive_within_3s']} | **{s['opt_within_3s']}** | {s['llm_within_3s']} |",
        "",
        f"> 优化后链路中有 **{s['opt_bypassed']}/{n}** 题命中「高置信直通」"
        f"（跳过 Query 理解的 LLM 往返，省约 900ms/次）。",
        "",
        "## 二、逐题对比",
        "",
    ]
    for x in c:
        m = x.get("metrics") or {}
        L.append(f"### id={x['id']} {x['question']}")
        L.append("")
        if x.get("must_have"):
            L.append(f"**关键事实**：{'、'.join(x['must_have'])}")
            L.append("")
            L.append(f"- 优化前命中：**{x['naive_fact_rate']}**"
                     + (f"（漏掉：{'、'.join(x['naive_fact_missed'])}）" if x["naive_fact_missed"] else ""))
            L.append(f"- 优化后命中：**{x['opt_fact_rate']}**"
                     + (f"（漏掉：{'、'.join(x['opt_fact_missed'])}）" if x["opt_fact_missed"] else ""))
            L.append("")
        L.append(f"**优化前**（{x.get('naive_timing', {}).get('total_ms')} ms）：")
        L.append("")
        L.append("> " + (x.get("naive_answer") or "").replace("\n", "\n> "))
        L.append("")
        L.append(f"**优化后**（{x.get('opt_timing', {}).get('total_ms')} ms）：")
        L.append("")
        L.append("> " + (x.get("opt_answer") or "").replace("\n", "\n> "))
        L.append("")
        L.append("**纯 LLM**：")
        L.append("")
        L.append("> " + (x.get("llm_answer") or "").replace("\n", "\n> "))
        L.append("")
        if m:
            parts = []
            for k in ("faithfulness", "relevancy", "ctx_precision", "ctx_recall", "correctness"):
                for tag in ("opt", "naive"):
                    v = (m.get(f"{tag}_{k}") or {}).get("score")
                    if v is not None:
                        parts.append(f"{tag}_{k}={v}")
            if parts:
                L.append("**指标**：" + "；".join(parts))
                L.append("")
        L.append("---")
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.EVAL_DATASET)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=config.RETRIEVAL_FINAL_TOP_K)
    ap.add_argument("--no-metrics", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    questions = json.loads((config.EVAL_DIR / args.dataset).read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    print(f"[工单] {config.WORK_ORDER_NO_OPT}（{config.WORK_ORDER_SHORT}）")
    print(f"[对比] {len(questions)} 题 × {{优化前, 优化后, 纯LLM}}，指标：{'关' if args.no_metrics else '开'}\n")

    t = time.perf_counter()
    from src.baseline import get_naive_index
    from src.embedder import embed_query
    from src.index_store import KnowledgeBase

    kb = KnowledgeBase.get()
    _ = kb.bm25
    embed_query("预热")
    ni = get_naive_index()
    print(f"[预热] 主线 {len(kb.chunks)} 块 / 基线 {len(ni.chunks)} 块，耗时 {time.perf_counter()-t:.1f}s\n")

    report = run(questions, with_metrics=not args.no_metrics, top_k=args.top_k)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    jp = config.EVAL_DIR / f"{config.WORK_ORDER_SHORT}_优化前后对比报告_{stamp}.json"
    mp = config.EVAL_DIR / f"{config.WORK_ORDER_SHORT}_优化前后对比报告_{stamp}.md"
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    mp.write_text(to_markdown(report), encoding="utf-8")

    print("\n=== 汇总 ===")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")
    print(f"\n[OK] {jp}")
    print(f"[OK] {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
