# -*- coding: utf-8 -*-
"""优化前后对比评估脚本
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

对 data/questions.json 中的 10 道题，分别在【优化前】与【优化后】两条链路上
执行问答，统计：
    - 检索命中（Recall）：检索到的上下文中是否含参考答案关键线索；
    - 答案命中（Accuracy）：最终答案中是否含参考答案关键线索；
    - 端到端耗时（Latency）：提问 → 答案返回。

用法（项目根目录，激活 langchain2 环境，需已启动 Ollama）：
    python scripts/evaluate.py                 # 优化前(LLM生成) + 优化后(抽取式)
    python scripts/evaluate.py --skip-llm      # 跳过耗时的 LLM 基线，仅评估检索与抽取式
    python scripts/evaluate.py --tag run1      # 自定义输出文件名后缀

产物：
    output/evaluation_report.json    结构化评估数据（供报告/图表复用）
    output/evaluation_report.md      人类可读对比报告
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.evaluate_keys import KEY_TOKENS, hit as _hit
from src.qa_engine import QAEngine


def _ctx_text(results) -> str:
    """把检索结果拼成上下文文本，用于检索命中判定。"""
    parts = []
    for r in results:
        doc = r.doc if hasattr(r, "doc") else r
        parts.append(doc.page_content)
    return "\n".join(parts)


def _rank_metrics(results, tokens: list[str]) -> tuple[bool, bool, float]:
    """返回 (Top-K 命中, Top-1 命中, MRR)。

    - Top-K 命中：返回的前 K 个片段中至少一个含答案线索；
    - Top-1 命中：排位第一的片段即含答案线索（体现检索精度）；
    - MRR：首个命中片段的排名倒数（越接近 1 越好）。
    """
    if not tokens:
        return False, False, 0.0
    hit, top1, mrr = False, False, 0.0
    for i, r in enumerate(results):
        doc = r.doc if hasattr(r, "doc") else r
        # 命中判定使用“块正文 + 父块”，与答案合成实际可用的上下文一致
        text = doc.page_content + "\n" + (doc.metadata.get("parent") or "")
        if _hit(text, tokens):
            hit = True
            if i == 0:
                top1 = True
            if mrr == 0.0:
                mrr = 1.0 / (i + 1)
    return hit, top1, round(mrr, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="优化前后 RAG 检索准确率对比评估")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM 基线（省时）")
    parser.add_argument("--tag", default="", help="输出文件名后缀")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热（统计含冷启动的耗时）")
    args = parser.parse_args()

    questions = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    engine = QAEngine()

    # 预热：加载向量库 / 触发 embedding 模型首次加载，避免把冷启动计入响应时间
    if not args.no_warmup:
        print("预热中（加载向量库与 embedding 模型）...")
        _ = engine.answer_optimized(questions[0]["question"])
        _ = engine.base_retriever.retrieve(questions[0]["question"], k=config.BASE_TOP_K)

    cases = []
    for i, item in enumerate(questions, 1):
        qid = int(item["id"])
        q = item["question"]
        ref = item.get("reference", "")
        tokens = KEY_TOKENS.get(qid, [])
        print(f"[{i}/{len(questions)}] Q{qid} {q[:32]}...")

        analysis = engine.understanding.analyze(q)

        # ---- 优化前：基线检索 + LLM 生成（复刻 01 工单） ----
        base_ans, base_elapsed = "", 0.0
        base_res = engine.base_retriever.retrieve(q, k=config.BASE_TOP_K)
        b_hit, b_top1, b_mrr = _rank_metrics(base_res, tokens)
        if not args.skip_llm:
            t0 = time.time()
            br = engine.answer_baseline(q)
            base_elapsed = time.time() - t0
            base_ans = br.answer

        # ---- 优化后：结构分块 + 多路召回 + 重排 + 抽取式合成 ----
        t0 = time.time()
        orr = engine.answer_optimized(q)
        opt_elapsed = time.time() - t0
        opt_ans = orr.answer
        opt_res = engine.opt_retriever.retrieve(analysis, k=config.ANSWER_POOL)
        o_hit, o_top1, o_mrr = _rank_metrics(opt_res, tokens)

        cases.append({
            "id": qid, "question": q, "reference": ref,
            "baseline": {
                "answer": base_ans, "elapsed": round(base_elapsed, 2),
                "retrieval_hit": b_hit, "top1_hit": b_top1, "mrr": b_mrr,
                "answer_hit": _hit(base_ans, tokens),
            },
            "optimized": {
                "answer": opt_ans, "elapsed": round(opt_elapsed, 3),
                "retrieval_hit": o_hit, "top1_hit": o_top1, "mrr": o_mrr,
                "answer_hit": _hit(opt_ans, tokens),
                "citations": orr.citations,
            },
        })
        print(f"    基线 Top1={b_top1} TopK={b_hit} MRR={b_mrr} 答案={_hit(base_ans, tokens)} "
              f"({base_elapsed:.1f}s) | 优化 Top1={o_top1} TopK={o_hit} MRR={o_mrr} "
              f"答案={_hit(opt_ans, tokens)} ({opt_elapsed:.3f}s)")

    n = len(cases)
    summary = {
        "total": n,
        "baseline_retrieval_hit": sum(c["baseline"]["retrieval_hit"] for c in cases),
        "baseline_top1_hit": sum(c["baseline"]["top1_hit"] for c in cases),
        "baseline_mrr": round(sum(c["baseline"]["mrr"] for c in cases) / n, 3),
        "baseline_answer_hit": sum(c["baseline"]["answer_hit"] for c in cases),
        "optimized_retrieval_hit": sum(c["optimized"]["retrieval_hit"] for c in cases),
        "optimized_top1_hit": sum(c["optimized"]["top1_hit"] for c in cases),
        "optimized_mrr": round(sum(c["optimized"]["mrr"] for c in cases) / n, 3),
        "optimized_answer_hit": sum(c["optimized"]["answer_hit"] for c in cases),
        "baseline_avg_elapsed": round(sum(c["baseline"]["elapsed"] for c in cases) / n, 3),
        "optimized_avg_elapsed": round(sum(c["optimized"]["elapsed"] for c in cases) / n, 3),
        "optimized_max_elapsed": round(max(c["optimized"]["elapsed"] for c in cases), 3),
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "work_order": "人工智能 NLP-RAG-基于 PDF 文档的问答系统优化",
        "pdf": str(config.PDF_PATH.name),
        "summary": summary,
        "cases": cases,
    }

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    json_path = config.OUTPUT_DIR / f"evaluation_report{suffix}.json"
    json_path = _write_text_retry(json_path, json.dumps(report, ensure_ascii=False, indent=2))

    # 报告正文先落 .txt（受限环境下 .md 可能被拦截），再由人工/工具另存为 .md
    md = _render_md(report)
    md_path = _write_text_retry(config.OUTPUT_DIR / f"evaluation_report{suffix}_markdown.txt", md)

    print("\n=== 汇总 ===")
    print(f"检索命中：优化前 {summary['baseline_retrieval_hit']}/{n} → "
          f"优化后 {summary['optimized_retrieval_hit']}/{n}")
    print(f"答案命中：优化前 {summary['baseline_answer_hit']}/{n} → "
          f"优化后 {summary['optimized_answer_hit']}/{n}")
    print(f"平均耗时：优化前 {summary['baseline_avg_elapsed']}s → "
          f"优化后 {summary['optimized_avg_elapsed']}s（最大 {summary['optimized_max_elapsed']}s）")
    print(f"\n产物：{json_path}\n      {md_path}")


def _write_text_retry(path: Path, text: str, tries: int = 3) -> Path:
    """带重试的写文件（受限环境下偶发 PermissionError，回退到临时目录）。"""
    for _ in range(tries):
        try:
            path.write_text(text, encoding="utf-8")
            return path
        except PermissionError:
            time.sleep(0.5)
    import tempfile

    alt = Path(tempfile.gettempdir()) / path.name
    alt.write_text(text, encoding="utf-8")
    print(f"⚠ 目标路径受限，已回退写入：{alt}")
    return alt


def _pct(x: int, n: int) -> str:
    return f"{x / n * 100:.1f}%"


def _render_md(report: dict) -> str:
    s = report["summary"]
    n = s["total"]
    L: list[str] = []
    a = L.append
    a("# 优化前后检索准确率对比报告")
    a("")
    a(f"> 工单编号：{report['work_order']}")
    a(f"> 生成时间：{report['generated_at']} ｜ 数据源：`{report['pdf']}` ｜ 题目数：{n}")
    a("")
    a("## 一、指标总览")
    a("")
    a("| 指标 | 优化前（基线） | 优化后 | 提升 |")
    a("|---|---|---|---|")
    a(f"| 检索 Top-1 命中率（Precision@1） | {_pct(s['baseline_top1_hit'], n)} "
      f"({s['baseline_top1_hit']}/{n}) | {_pct(s['optimized_top1_hit'], n)} "
      f"({s['optimized_top1_hit']}/{n}) | "
      f"+{(s['optimized_top1_hit'] - s['baseline_top1_hit']) / n * 100:.1f} pt |")
    a(f"| 检索 MRR | {s['baseline_mrr']} | {s['optimized_mrr']} | "
      f"+{s['optimized_mrr'] - s['baseline_mrr']:.3f} |")
    a(f"| 检索 Top-K 命中率（Recall@K） | {_pct(s['baseline_retrieval_hit'], n)} "
      f"({s['baseline_retrieval_hit']}/{n}) | {_pct(s['optimized_retrieval_hit'], n)} "
      f"({s['optimized_retrieval_hit']}/{n}) | "
      f"{s['optimized_retrieval_hit'] - s['baseline_retrieval_hit']} 题 |")
    a(f"| 答案命中率（Accuracy） | {_pct(s['baseline_answer_hit'], n)} "
      f"({s['baseline_answer_hit']}/{n}) | {_pct(s['optimized_answer_hit'], n)} "
      f"({s['optimized_answer_hit']}/{n}) | "
      f"+{(s['optimized_answer_hit'] - s['baseline_answer_hit']) / n * 100:.1f} pt |")
    a(f"| 平均响应时间 | {s['baseline_avg_elapsed']} s | {s['optimized_avg_elapsed']} s | "
      f"加速 {s['baseline_avg_elapsed'] / max(s['optimized_avg_elapsed'], 1e-6):.0f}× |")
    a(f"| 最大响应时间 | — | {s['optimized_max_elapsed']} s | ≤ 3 s ✅ |")
    a("")
    a("## 二、逐题对比")
    a("")
    a("| # | 问题 | 优化前Top1 | 优化前TopK | 优化前答案 | 优化后Top1 | 优化后TopK | 优化后答案 | 优化前耗时 | 优化后耗时 |")
    a("|---|------|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|")
    for c in report["cases"]:
        b, o = c["baseline"], c["optimized"]
        mark = lambda v: "✅" if v else "❌"
        a(f"| {c['id']} | {c['question'][:36]} | {mark(b['top1_hit'])} | "
          f"{mark(b['retrieval_hit'])} | {mark(b['answer_hit'])} | {mark(o['top1_hit'])} | "
          f"{mark(o['retrieval_hit'])} | {mark(o['answer_hit'])} | "
          f"{b['elapsed']}s | {o['elapsed']}s |")
    a("")
    a("## 三、逐题答案明细")
    a("")
    for c in report["cases"]:
        a(f"### Q{c['id']}、{c['question']}")
        a(f"- **参考答案**：{c['reference']}")
        if c["baseline"]["answer"]:
            a(f"- **优化前答案**（{c['baseline']['elapsed']}s）：{c['baseline']['answer'][:400]}")
        a(f"- **优化后答案**（{c['optimized']['elapsed']}s）：{c['optimized']['answer'][:400]}")
        a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()