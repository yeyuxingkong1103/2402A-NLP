# -*- coding: utf-8 -*-
"""T6 端到端问答评估：14 题带引用答案 + 首字延迟 + 引用可回溯校验。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

实测内容（全部落盘，禁止口头结论）：
    1) 每题答案文本 + 引用（`[文件名: 页码]`）+ 引用校验报告；
    2) **首字延迟**：预热后逐题首字 ms（max 必须 ≤ 3000），另记第 1 题冷启动值；
    3) **引用可回溯**：引用页码 ∈ [1, 页数]，且该页正文里能找到答案关键片段（数值/项目名）；
    4) 文件归属：题 1~4 引用 招股说明书2.pdf，其余引用 招股说明书1.pdf。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/离线/eval_answers.py
    pwsh -NoProfile -File run_py.ps1 测试/离线/eval_answers.py --limit 3      # 冒烟
    pwsh -NoProfile -File run_py.ps1 测试/离线/eval_answers.py --file-scoped   # 每题限定到目标 PDF
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import citation as citation_mod, text_utils  # noqa: E402
from app.core.config import get_config  # noqa: E402
from app.core.generator import build_generator  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.retriever import build_retriever  # noqa: E402

EVAL_SET = REPO_ROOT / "测试" / "测试数据" / "eval_retrieval_14.jsonl"
OUT_DIR = REPO_ROOT / "优化" / "评估结果"
# 题 1~4 属 PDF2；其余属 PDF1（用于「文件归属」校验）
PDF2_IDS = {1, 2, 3, 4}


def page_text_from_db(cfg, file_name: str, page: int) -> str:
    """从 SQLite pages 表取页正文（引用可回溯校验用）。"""
    import sqlite3

    conn = sqlite3.connect(str(cfg.paths.index_dir / "rag.sqlite3"))
    try:
        row = conn.execute("SELECT text FROM pages WHERE file_name=? AND page=?", (file_name, int(page))).fetchone()
    finally:
        conn.close()
    return str(row[0]) if row else ""


def support_check(answer_text: str, evidence_text: str) -> tuple[bool, str]:
    """答案能否在引用处核实（与生产链路同一条判据：citation.answer_support_check）。"""
    return citation_mod.answer_support_check(answer_text, evidence_text)


def correctness_check(answer_text: str, golden_answer: str) -> tuple[bool, str]:
    """答案与 golden 的**语义一致性**判定（数值优先，其次关键实体覆盖率）。

    数值题：golden 的数值必须全部出现在答案里（保真）；
    文本题：golden 的 2 字以上实词覆盖率 ≥ 0.6 视为答对。
    """
    if not str(answer_text or "").strip() or answer_text.strip() == citation_mod.UNKNOWN_TEXT:
        return False, "答案为空或「不清楚」"
    golden_numbers = [citation_mod.normalize_number(n) for n in text_utils.extract_numbers(golden_answer)]
    if golden_numbers:
        answer_numbers = {citation_mod.normalize_number(n) for n in text_utils.extract_numbers(answer_text)}
        hit = [n for n in golden_numbers if n in answer_numbers]
        rate = len(hit) / len(golden_numbers)
        return (rate >= 0.99), f"golden 数值命中 {len(hit)}/{len(golden_numbers)}={rate:.2f}"
    tokens = [t for t in text_utils.tokenize(golden_answer)
              if len(t) >= 2 and t not in text_utils.STOPWORDS]
    if not tokens:
        return False, "golden 无可比 token"
    answer_tokens = set(text_utils.tokenize(answer_text))
    coverage = len([t for t in tokens if t in answer_tokens]) / len(tokens)
    return (coverage >= 0.60), f"golden 实词覆盖 {coverage:.2f}（{len(tokens)} 个）"


def main(argv: list[str] | None = None) -> int:
    """跑 14 题端到端问答并输出实测报告。"""
    parser = argparse.ArgumentParser(description="T6 端到端问答评估（14 题带引用答案 + 首字）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟）")
    parser.add_argument("--file-scoped", action="store_true", help="每题限定到其目标 PDF（file_names 硬过滤）")
    parser.add_argument("--json", default=str(OUT_DIR / "answer_eval_t6.json"))
    parser.add_argument("--md", default=str(OUT_DIR / "answer_eval_t6.md"))
    args = parser.parse_args(argv)

    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("eval_answers")
    questions = [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[: args.limit]
    print(f"题集：{EVAL_SET}（{len(questions)} 题，file_scoped={args.file_scoped}）")

    retriever = build_retriever(cfg=cfg)                     # load() 内预热 jieba + 嵌入
    generator = build_generator(cfg=cfg, logger=log)         # 解析 LLM 后端（ollama→openai→extractive）
    backend = generator.llm.backend if generator.llm is not None else None
    print(f"检索器：{retriever.health()['chunks']} 块；生成后端：{backend.name if backend else 'extractive'}"
          f"（{backend.model if backend else 'rule-based'}）")

    # —— 显式预热大模型：让「首字 ≤3 s」按在线热路径口径测量（冷启动值另行记录）——
    warm_cold_ms = None
    if generator.llm is not None:
        t0 = time.perf_counter()
        try:
            generator.llm.generate_full("预热：请只回答「好」。", max_tokens=4, temperature=0.0, logger=log)
            warm_cold_ms = round((time.perf_counter() - t0) * 1000, 2)
            print(f"LLM 预热完成：冷启动 {warm_cold_ms} ms（不计入首字预算）")
        except Exception as exc:  # noqa: BLE001 —— 预热失败不应静默：记录并继续（后续会如实报错）
            log.log_event("eval.warmup_failed", level="ERROR", error_type=type(exc).__name__, message=str(exc))
            print(f"⚠️ LLM 预热失败：{type(exc).__name__}: {exc}")

    results: list[dict] = []
    for index, item in enumerate(questions, start=1):
        target = [item["file_name"]] if args.file_scoped else None
        started = time.perf_counter()
        retrieval = retriever.retrieve(item["question"], top_k=cfg.retrieval.top_k, file_names=target, logger=log)
        answer = generator.answer(item["question"], retrieval, trace_id=f"ev{index:02d}", logger=log)
        wall_ms = round((time.perf_counter() - started) * 1000, 2)
        # 引用可回溯校验：① 答案能在引用处（引用块内容 ∪ 引用页正文）核实；② 单看页正文层也能核实的比例
        checks = []
        golden_pages = {int(p) for p in (item.get("evidence_pages") or [])}
        for cite in answer.citations:
            evidence = citation_mod.evidence_text_for(cite, chunk_lookup=generator.chunk_lookup,
                                                      page_text_lookup=generator.page_text_lookup)
            ok, why = support_check(answer.text, evidence)
            page_text = page_text_from_db(cfg, cite.file_name, cite.page)
            page_ok, page_why = support_check(answer.text, page_text)
            checks.append({"citation": cite.render(language=answer.language), "page": cite.page,
                           "file_name": cite.file_name, "chunk_id": cite.chunk_id,
                           "supported": ok, "why": why,
                           "supported_in_page_text": page_ok, "page_text_why": page_why,
                           "is_golden_page": int(cite.page) in golden_pages})
        wrong_file = [c["file_name"] for c in checks
                      if (item["id"] in PDF2_IDS) != c["file_name"].endswith("2.pdf")]
        correct, correct_why = correctness_check(answer.text, str(item.get("answer") or ""))
        results.append({
            "id": item["id"], "question": item["question"], "subject": item.get("subject"),
            "expected_file": item["file_name"], "answer": answer.text,
            "citations": [c.to_dict() for c in answer.citations],
            "citation_checks": checks, "wrong_file_citations": wrong_file,
            "is_unknown": answer.is_unknown, "unknown_reason": answer.unknown_reason,
            "correct": correct, "correct_why": correct_why,
            "first_token_ms": answer.first_token_ms, "total_ms": answer.total_ms, "wall_ms": wall_ms,
            "citation_accuracy": (answer.citation_report.accuracy if answer.citation_report else None),
            "citation_reasons": (answer.citation_report.reasons if answer.citation_report else []),
            "golden_page_hit": bool(checks) and all(c["is_golden_page"] for c in checks),
            "top_chunks": [c.chunk_id for c in retrieval.chunks],
            "golden_answer": item.get("answer"),
            "evidence_pages": item.get("evidence_pages"),
        })
        flag = "✅" if (correct and checks and all(c["supported"] for c in checks) and not wrong_file) else "⚠️"
        print(f"  {flag} id={item['id']:<4} 首字={answer.first_token_ms:7.1f} ms 总={answer.total_ms:8.1f} ms "
              f"引用={[c.render(language=answer.language) for c in answer.citations]} "
              f"{'【不清楚】' if answer.is_unknown else ''} 正确={correct}（{correct_why}）")

    firsts = [r["first_token_ms"] for r in results if not r["is_unknown"]]
    summary = {
        "total": len(results),
        "answered": sum(1 for r in results if not r["is_unknown"]),
        "unknown_ids": [r["id"] for r in results if r["is_unknown"]],
        "first_token_warm": {
            "count": len(firsts), "max": max(firsts) if firsts else None,
            "avg": (round(sum(firsts) / len(firsts), 2) if firsts else None),
            "budget_ms": 3000.0,
            "within_budget": bool(firsts) and max(firsts) <= 3000.0,
        },
        "first_question_cold_ms": results[0]["first_token_ms"] if results else None,
        "llm_warmup_cold_ms": warm_cold_ms,
        "citation_supported": sum(1 for r in results for c in r["citation_checks"] if c["supported"]),
        "citation_total": sum(len(r["citation_checks"]) for r in results),
        "citation_accuracy": (
            round(sum(1 for r in results for c in r["citation_checks"] if c["supported"])
                  / max(1, sum(len(r["citation_checks"]) for r in results)), 4)),
        # 只看页正文文本层也能核实的引用数（表格块的证据在 Markdown 里，正文层常核不出，需如实披露）
        "citation_page_text_supported": sum(1 for r in results for c in r["citation_checks"]
                                            if c["supported_in_page_text"]),
        "golden_page_hit_questions": [r["id"] for r in results if r["golden_page_hit"]],
        "correct_count": sum(1 for r in results if r["correct"]),
        "accuracy": round(sum(1 for r in results if r["correct"]) / max(1, len(results)), 4),
        "incorrect_ids": [r["id"] for r in results if not r["correct"]],
        "incorrect_detail": {str(r["id"]): r["correct_why"] for r in results if not r["correct"]},
        "wrong_file_questions": [r["id"] for r in results if r["wrong_file_citations"]],
        "file_scoped": bool(args.file_scoped),
        "backend": ({"name": backend.name, "model": backend.model, "base_url": backend.base_url}
                    if backend else {"name": "extractive", "model": "rule-based", "base_url": "local"}),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    lines = [f"# T6 端到端问答（{len(results)} 题，后端 {summary['backend']['name']}）", "",
             f"> 生成时间：{summary['generated_at']}　模式：{'按题限定 PDF' if args.file_scoped else '全库检索'}", "",
             "| 指标 | 值 |", "| --- | --- |",
             f"| 作答题数（非「不清楚」） | {summary['answered']}/{summary['total']} |",
             f"| **准确率（与 golden 语义一致）** | **{summary['correct_count']}/{summary['total']} = "
             f"{summary['accuracy'] * 100:.1f}%**（错题：{summary['incorrect_ids'] or '无'}） |",
             f"| 首字（预热后）max | **{summary['first_token_warm']['max']} ms**（预算 ≤3000ms，达标="
             f"{summary['first_token_warm']['within_budget']}） |",
             f"| 首字（预热后）平均 | {summary['first_token_warm']['avg']} ms |",
             f"| 引用可回溯 | {summary['citation_supported']}/{summary['citation_total']} = "
             f"{summary['citation_accuracy'] * 100:.1f}% |",
             f"| 引用可回溯（仅看页正文文本层） | {summary['citation_page_text_supported']}/"
             f"{summary['citation_total']}（表格证据在 Markdown 表里，正文层核不出属预期） |",
             f"| 引用页命中 golden 证据页的题 | {len(summary['golden_page_hit_questions'])}/{summary['total']} |",
             f"| 文件归属错误题 | {summary['wrong_file_questions'] or '无'} |", "",
             "## 逐题答案", ""]
    for r in results:
        lines.append(f"### 题 {r['id']}：{r['question']}")
        lines.append(f"- 答案：{r['answer']}")
        lines.append(f"- 引用：{['%s（可回溯=%s）' % (c['citation'], c['supported']) for c in r['citation_checks']]}")
        lines.append(f"- 首字 {r['first_token_ms']} ms / 总 {r['total_ms']} ms / 引用精度 {r['citation_accuracy']}")
        lines.append(f"- golden：{r['golden_answer']}")
        lines.append("")
    Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"作答 {summary['answered']}/{summary['total']}；准确率 {summary['correct_count']}/{summary['total']}"
          f"（错题 {summary['incorrect_ids'] or '无'}）；首字 max={summary['first_token_warm']['max']} ms "
          f"（预算 3000，达标={summary['first_token_warm']['within_budget']}）；"
          f"引用可回溯 {summary['citation_supported']}/{summary['citation_total']}；"
          f"文件归属错误={summary['wrong_file_questions'] or '无'}")
    print(f"JSON：{out_json}\nMD  ：{args.md}")
    shutdown_logging()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 顶层兜底
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
