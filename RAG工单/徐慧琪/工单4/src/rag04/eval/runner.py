# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""16 题评估执行器：产出「答案 + 精确度 + 引用来源」。"""
from __future__ import annotations

import logging
import statistics
import time
from pathlib import Path

from rag04.config import Settings
from rag04.eval.metrics import (
    hit_rate,
    is_relevant,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    verdict,
)
from rag04.generate.llm import detect_answer_refusal
from rag04.eval.questions import QUESTIONS, coverage
from rag04.schema import Answer, Hit

logger = logging.getLogger("rag04.eval.runner")

_DOC_FILTER_NOTE = (
    "检索指标只统计与题目同文档（doc_id）的召回块：两本招股书页码大量重叠"
    "（如 p39、p72 两本皆有），不按文档过滤会把异库同页块误判为相关而虚高。"
)
_EN_ACCURACY_NOTE = (
    "英文模式下问题为英文、标准要点（answer_key）为中文，逐字覆盖率无意义，"
    "故不判定答案正确性（answer_accuracy 为 null）；检索指标与耗时不受语言影响，"
    "照常统计。"
)


def _hits_in_doc(hits: list[Hit], doc_id: str) -> list[Hit]:
    """只保留与题目同文档的召回块。

    `is_relevant(hit, gold_pages)` 的既定口径只比页码，而 gold_pages 只对
    q.doc_id 才有意义；两份语料页码重叠，必须先按文档过滤再计分。
    """
    return [h for h in hits if h.doc_id == doc_id]


def run_eval(settings: Settings, pipeline=None, use_english: bool = False) -> dict:
    """跑完 16 题，返回结构化结果。"""
    if pipeline is None:
        from rag04.pipeline import RAGPipeline
        pipeline = RAGPipeline(settings)
    preload = getattr(pipeline, "_ensure_loaded", None)   # 桩管道可无此钩子
    if callable(preload):
        preload()

    rows: list[dict] = []
    hits_list: list[list[Hit]] = []
    gold_list: list[list[int]] = []
    latencies: list[float] = []

    for q in QUESTIONS:
        question = q.question_en if use_english else q.question
        t0 = time.perf_counter()
        try:
            ans = pipeline.ask(question)
            err = ""
        except Exception as e:
            logger.error("id %s 评估失败：%s: %s", q.qid, type(e).__name__, e)
            ans = Answer(question=question, answer="",
                         lang="en" if use_english else "zh")
            err = f"{type(e).__name__}: {e}"

        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        scored = _hits_in_doc(ans.hits, q.doc_id)
        hits_list.append(scored)
        gold_list.append(q.gold_pages)

        if use_english:
            # 约束 11 只要求按提问语言作答；answer_key 是中文，英文模式下
            # 覆盖率不可用，宁可显式置空，也不上报 ~0 的假准确率。
            cov: float | None = None
            correct: bool | None = None
        else:
            cov = coverage(ans.answer, q)
            correct = verdict(cov, threshold=settings.answer_acc_threshold,
                              strict=q.strict,          # 约束 10：strict 必须全中
                              refused=ans.refused)      # RC6：拒答不得判对

        rows.append({
            "qid": q.qid,
            "question": question,
            # Fix 2：逐行记录语言，英文报告可自证（不再靠「整份报告是英文」推断）
            "lang": ans.lang,
            "answer": ans.answer,
            "answer_key": q.answer_key,
            "coverage": None if cov is None else round(cov, 4),
            "correct": correct,
            "strict": q.strict,
            "gold_pages": q.gold_pages,
            "block_type": q.block_type,
            "citations": ans.citations,
            "retrieved_pages": [h.page for h in ans.hits],
            "scored_pages": [h.page for h in scored],
            "hits_other_doc": len(ans.hits) - len(scored),
            "latency_ms": round(latency, 1),
            "llm_backend": ans.llm_backend,
            "refused": ans.refused,
            # RC7：结构化「证据是否充分」字段为主判据，正则为兜底。
            # regex_refused 记录同一答案若只走正则会得到的判定，供一致/分歧清单。
            "answerable": getattr(ans, "answerable", None),
            "refusal_source": getattr(ans, "refusal_source", ""),
            "regex_refused": detect_answer_refusal(ans.answer),
            "error": err,
        })
        logger.info("id %-4s 覆盖率 %s 用时 %.0fms", q.qid,
                    "不适用" if cov is None else f"{cov:.2f}", latency)

    # 检索指标（主口径 k=5，另出 k=3 与 k=10）
    metrics: dict = {}
    for k in settings.k_report:
        metrics[f"precision@{k}"] = round(
            statistics.mean(precision_at_k(h, g, k)
                            for h, g in zip(hits_list, gold_list)), 4)
        metrics[f"recall@{k}"] = round(
            statistics.mean(recall_at_k(h, g, k)
                            for h, g in zip(hits_list, gold_list)), 4)
        metrics[f"ndcg@{k}"] = round(
            statistics.mean(ndcg_at_k(h, g, k)
                            for h, g in zip(hits_list, gold_list)), 4)

    metrics["hit_rate"] = round(
        hit_rate([any(is_relevant(h, g) for h in hs)
                  for hs, g in zip(hits_list, gold_list)]), 4)
    metrics["mrr"] = round(mrr(hits_list, gold_list), 4)
    if use_english:
        metrics["answer_accuracy"] = None
        metrics["accuracy_note"] = _EN_ACCURACY_NOTE
    else:
        metrics["answer_accuracy"] = round(
            sum(1 for r in rows if r["correct"]) / len(rows), 4)

    if latencies:
        s = sorted(latencies)
        metrics["latency_p50_ms"] = round(statistics.median(s), 1)
        metrics["latency_p95_ms"] = round(s[min(len(s) - 1, int(len(s) * 0.95))], 1)
        metrics["latency_p99_ms"] = round(s[min(len(s) - 1, int(len(s) * 0.99))], 1)
        metrics["latency_mean_ms"] = round(statistics.mean(s), 1)

    notes = [_DOC_FILTER_NOTE]
    if use_english:
        notes.append(_EN_ACCURACY_NOTE)

    return {
        "mode": settings.pipeline_mode,
        "lang": "en" if use_english else "zh",
        "n_questions": len(rows),
        "metrics": metrics,
        "notes": notes,
        "rows": rows,
    }


def write_report(result: dict, path: Path) -> Path:
    """把评估结果写成 Markdown 报告（答案 + 精确度 + 引用来源）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = result["metrics"]
    ks = sorted({int(key.split("@")[1]) for key in m if "@" in key})

    def fmt(value) -> str:
        return "不适用" if value is None else str(value)

    lines = [
        f"# 检索精确度报告（{result['mode']}｜{'英文' if result['lang'] == 'en' else '中文'}）",
        "",
        f"- 题目数：{result['n_questions']}",
        f"- 运行模式：`{result['mode']}`",
        "",
        "## 一、总体指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
    ]
    table_keys = ["answer_accuracy", "hit_rate", "mrr"]
    for name in ("precision", "recall", "ndcg"):
        table_keys += [f"{name}@{k}" for k in ks]
    table_keys += ["latency_mean_ms", "latency_p50_ms",
                   "latency_p95_ms", "latency_p99_ms"]
    for key in table_keys:
        if key in m:
            lines.append(f"| {key} | {fmt(m[key])} |")
    if m.get("accuracy_note"):
        lines += ["", f"> 说明：{m['accuracy_note']}"]
    for note in result.get("notes", []):
        lines.append(f"> 口径：{note}")

    lines += ["", "## 二、逐题结果", ""]
    for r in result["rows"]:
        strict = "（严格口径）" if r["strict"] else ""
        # Fix 2：先看 refused 再看 correct。原顺序（correct is None 优先）会让
        # 英文模式的拒答一律渲染成「判定：不适用」，把拒答藏了起来。
        if r.get("refused"):
            verdict_text = "🚫 拒答（不计正确）"
        elif r["correct"] is None:
            verdict_text = "不适用（本语言模式不判答案正确性）"
        else:
            verdict_text = "✅ 正确" if r["correct"] else "❌ 不正确"
        lines += [
            f"### id {r['qid']}{strict}",
            "",
            f"**问题**：{r['question']}",
            "",
            f"**答案**：{r['answer'] or '（无）'}",
            "",
            (f"**要点覆盖率**：{fmt(r['coverage'])}　**判定**：{verdict_text}"
             f"　**用时**：{r['latency_ms']} ms　**后端**：{r['llm_backend']}"
             f"　**语言**：{r.get('lang') or '—'}"),
            "",
            (f"**answerable（结构化）**：{fmt(r.get('answerable'))}"
             f"　**拒答判据来源**：{r.get('refusal_source') or '—'}"
             f"　**纯正则判定**：{'拒答' if r.get('regex_refused') else '非拒答'}"),
            "",
            f"**标准要点**：{'、'.join(r['answer_key'])}",
            "",
            (f"**出处页码**：{r['gold_pages']}　**召回页码**：{r['retrieved_pages']}"
             f"　**计分页码（同文档）**：{r['scored_pages']}"),
            "",
            "**引用来源**：",
            "",
            "| 文档 | 页码 | 类型 | 来源ID | 相似度 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for c in r["citations"][:5]:
            lines.append(
                f"| {c['doc_id']} | {c['page']} | {c['block_type']} "
                f"| {c['source_id']} | {c['score']} |"
            )
        if r["error"]:
            lines += ["", f"> ⚠️ 错误：{r['error']}"]
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path
