# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估

RAG 测试执行脚本 —— 对 10 个测试用例跑完整 RAG 流程（检索 + 生成）并留痕。

执行的 RAG 系统为「01-06 工单」的最终形态（wo06_hybrid 预设）：
    Query 理解（工单05）→ 混合检索：向量 + BM25 + RRF 融合（工单06）
    → 级联重排（工单02/06）→ 基于上下文的 LLM 生成（工单01）
    表格块参与检索（工单03）；图像能力默认关闭（工单04，年报图表多、耗时高，
    可用 --with-images 在 prepare_corpus.py 中打开）。

每个问题记录：
    · 检索结果：命中的片段（文档名 / 页码 / 类型 / 章节 / 融合分数 / 片段摘要）
    · 检索命中判定：是否命中主责文档、主责文档首次出现的排名、期望页码是否命中
    · 生成的答案、引用来源、是否触发拒答
    · 各阶段耗时（Query理解 / 向量召回 / BM25召回 / 融合 / 重排 / 生成 / 总耗时）

用法：
    python run_rag_test.py                     # 跑全部 10 题
    python run_rag_test.py --ids Q05 Q09       # 只跑指定题
    python run_rag_test.py --top-k 8           # 覆盖题目自带的 top_k
    python run_rag_test.py --dry-run           # 只打印题目，不调用模型
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import config, llm  # noqa: E402
from prepare_corpus import (COLLECTION, PRESET, WO_NO, ensure_results_dir,  # noqa: E402
                            load_pipeline)


# ---------------------------------------------------------------------------
# 单题执行
# ---------------------------------------------------------------------------
def run_case(pipeline, case: dict, top_k: int | None = None) -> dict:
    """跑单个测试用例，返回结构化测试结果。"""
    k = top_k or case.get("top_k") or config.TOP_K_RERANK
    t0 = time.perf_counter()
    trace = pipeline.ask(case["question"], top_k=k, return_trace=True)
    wall = time.perf_counter() - t0

    docs = trace.get("docs", []) or []
    retrieved = []
    for i, d in enumerate(docs, 1):
        retrieved.append({
            "rank": i,
            "doc": d.get("doc", ""),
            "page": d.get("page", 0),
            "type": d.get("type", "text"),
            "section": d.get("section", ""),
            "score": round(float(d.get("final_score", d.get("score", 0.0))), 4),
            "source": d.get("source", ""),
            "chunk_id": d.get("chunk_id", ""),
            "text_len": len(d.get("text", "") or ""),
            # 完整片段留给评估脚本做忠实度/召回判定；snippet 只用于报告展示
            "text": d.get("text", "") or "",
            "snippet": (d.get("text", "") or "")[:220].replace("\n", " "),
        })

    retrieved_docs = [r["doc"] for r in retrieved]
    retrieved_pages = sorted({r["page"] for r in retrieved})

    # ---- 检索命中判定 ----
    ref = case.get("reference_doc", "")
    rank_of_ref = next((i for i, d in enumerate(retrieved_docs, 1) if ref and ref in d), 0)
    hit_ref = rank_of_ref > 0
    ref_docs_all = case.get("reference_docs", [ref]) or [ref]
    covered = [d for d in ref_docs_all if any(d in rd for rd in retrieved_docs)]
    expected_pages = case.get("expected_pages", []) or []
    page_hits = sorted(set(expected_pages) & set(retrieved_pages))

    timings = {k2: round(v, 3) for k2, v in (trace.get("timings") or {}).items()}

    return {
        "id": case["id"],
        "question": case["question"],
        "question_type": case["question_type"],
        "origin": case["origin"],
        "difficulty": case["difficulty"],
        "need_table": case.get("need_table", False),
        "top_k": k,
        "reference_doc": ref,
        "reference_docs": ref_docs_all,
        "expected_pages": expected_pages,
        "answer": trace.get("answer", ""),
        "refused": bool(trace.get("refused", False)),
        "citations": trace.get("citations", []),
        "understanding": trace.get("understanding", {}),
        "retrieved": retrieved,
        "retrieved_docs": retrieved_docs,
        "retrieved_pages": retrieved_pages,
        "n_table_chunks": sum(1 for r in retrieved if r["type"] == "table"),
        "n_docs_covered": len(covered),
        "n_docs_expected": len(ref_docs_all),
        "doc_coverage": round(len(covered) / max(len(ref_docs_all), 1), 4),
        "hit_reference_doc": hit_ref,
        "rank_of_reference_doc": rank_of_ref,
        "page_hits": page_hits,
        "page_hit_rate": round(len(page_hits) / max(len(expected_pages), 1), 4)
        if expected_pages else None,
        "timings": timings,
        "latency": round(wall, 3),
    }


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    hits = sum(1 for r in results if r["hit_reference_doc"])
    lat = [r["latency"] for r in results]
    page_rates = [r["page_hit_rate"] for r in results if r["page_hit_rate"] is not None]
    cov = [r["doc_coverage"] for r in results]
    stage_keys = ["query_understanding", "vector_recall", "fulltext_recall",
                  "fusion", "rerank", "retrieve", "generate"]
    stage_avg = {}
    for k in stage_keys:
        vals = [r["timings"].get(k) for r in results if r["timings"].get(k) is not None]
        if vals:
            stage_avg[k] = round(sum(vals) / len(vals), 3)
    return {
        "n": n,
        "retrieval_hit_rate": round(hits / n, 4),
        "avg_doc_coverage": round(sum(cov) / n, 4),
        "avg_page_hit_rate": round(sum(page_rates) / len(page_rates), 4) if page_rates else None,
        "refused_count": sum(1 for r in results if r["refused"]),
        "avg_latency": round(sum(lat) / n, 3),
        "max_latency": round(max(lat), 3),
        "avg_stage_seconds": stage_avg,
        "llm_usage": llm.get_usage(),
        "preset": PRESET,
        "collection": COLLECTION,
    }


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def save(results: list[dict], summary: dict, out_dir: Path) -> None:
    payload = {
        "wo_no": WO_NO,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rag_system": {
            "preset": PRESET,
            "collection": COLLECTION,
            "chunk_strategy": "structure",
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "strategy": "hybrid(向量+BM25, RRF融合)",
            "reranker": "cascade(级联重排)",
            "recall_k": config.TOP_K_RECALL,
            "embed_model": config.EMBED_MODEL_NAME,
            "llm_model": config.LLM_MODEL,
        },
        "summary": summary,
        "results": results,
    }
    (out_dir / "rag_test_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "rag_test_results.md").write_text(_render_md(payload), encoding="utf-8")


def _render_md(payload: dict) -> str:
    s, results = payload["summary"], payload["results"]
    lines = [
        f"# RAG 测试结果（{s['n']} 个问题）",
        "",
        f"> 工单编号：{WO_NO}　生成时间：{payload['generated_at']}",
        "> RAG 系统：01-06 工单成果 —— 结构分块 + 混合检索（向量+BM25+RRF）+ 级联重排 + LLM 生成",
        "",
        "## 一、总体情况",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 测试问题数 | {s['n']} |",
        f"| 检索命中率（主责文档进入 Top-k） | {s['retrieval_hit_rate']:.2%} |",
        f"| 平均文档覆盖率（多文档题） | {s['avg_doc_coverage']:.2%} |",
        f"| 平均期望页码命中率 | "
        f"{(s['avg_page_hit_rate'] or 0):.2%} |",
        f"| 拒答次数 | {s['refused_count']} |",
        f"| 平均耗时 | {s['avg_latency']:.2f} s |",
        f"| 最大耗时 | {s['max_latency']:.2f} s |",
        "",
        "**平均阶段耗时（秒）**：" + "、".join(
            f"{k} {v}" for k, v in s["avg_stage_seconds"].items()),
        "",
        "## 二、逐题检索结果",
        "",
        "| 编号 | 题型 | 主责文档 | 命中 | 首次排名 | 页码命中 | 表格块 | 耗时(s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        pages = "、".join(str(p) for p in r["page_hits"]) or "—"
        lines.append(
            f"| {r['id']} | {r['question_type']} | {r['reference_doc']} | "
            f"{'✓' if r['hit_reference_doc'] else '✗'} | "
            f"{r['rank_of_reference_doc'] or '—'} | {pages} | "
            f"{r['n_table_chunks']} | {r['latency']:.2f} |")

    lines += ["", "## 三、逐题明细", ""]
    for r in results:
        lines += [
            f"### {r['id']}　{r['question']}",
            "",
            f"- **题型**：{r['question_type']}　**难度**：{r['difficulty']}　"
            f"**top_k**：{r['top_k']}　**耗时**：{r['latency']:.2f}s",
            f"- **检索命中主责文档**：{'是' if r['hit_reference_doc'] else '否'}"
            f"（首次出现排名 {r['rank_of_reference_doc'] or '未出现'}）",
            f"- **文档覆盖**：{r['n_docs_covered']}/{r['n_docs_expected']}"
            f"（{r['doc_coverage']:.0%}）",
            f"- **期望页码命中**：{'、'.join(str(p) for p in r['page_hits']) or '无'}"
            f"（期望 {'、'.join(str(p) for p in r['expected_pages']) or '不限'}）",
            "",
            "**检索片段（Top-k）**：",
            "",
            "| # | 文档 | 页 | 类型 | 章节 | 分数 | 片段摘要 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for d in r["retrieved"]:
            sec = (d["section"] or "—")[:18]
            snip = d["snippet"][:60].replace("|", "｜")
            lines.append(f"| {d['rank']} | {d['doc']} | {d['page']} | {d['type']} | "
                         f"{sec} | {d['score']:.4f} | {snip}… |")
        lines += [
            "",
            "**RAG 生成的答案**：",
            "",
            "> " + (r["answer"] or "（空）").replace("\n", "\n> "),
            "",
            "**引用来源**：" + ("、".join(
                f"《{c.get('doc')}》第{c.get('page')}页" for c in r["citations"]) or "无"),
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=f"RAG 测试执行（{WO_NO}）")
    ap.add_argument("--cases", default="results/test_cases.json")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--top-k", type=int, default=0, help="覆盖题目自带 top_k（0=不覆盖）")
    ap.add_argument("--ids", nargs="*", default=None, help="只跑指定题目编号")
    ap.add_argument("--dry-run", action="store_true", help="只打印题目，不调用模型")
    args = ap.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = Path(__file__).resolve().parents[1] / args.cases
    if not cases_path.exists():
        raise SystemExit(f"未找到测试用例：{cases_path}\n请先运行：python build_questions.py")

    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if args.ids:
        cases = [c for c in cases if c["id"] in set(args.ids)]
    if not cases:
        raise SystemExit("没有匹配的测试用例。")

    print(f"===== {WO_NO} · RAG 测试执行 =====")
    print(f"用例数 {len(cases)} | 集合 {args.collection} | 预设 {PRESET}")

    if args.dry_run:
        for c in cases:
            print(f"  {c['id']} [{c['question_type']}] top_k={c.get('top_k')} "
                  f"{c['question'][:50]}…")
        print("\n（--dry-run：未调用检索与生成）")
        return

    pipeline = load_pipeline(args.collection)

    results: list[dict] = []
    for i, c in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {c['id']} {c['question'][:40]}…")
        try:
            r = run_case(pipeline, c, top_k=args.top_k or None)
        except Exception as e:                       # 单题失败不中断整轮测试
            print(f"  [error] {c['id']} 执行失败：{e}")
            r = {"id": c["id"], "question": c["question"],
                 "question_type": c["question_type"], "origin": c["origin"],
                 "difficulty": c["difficulty"], "need_table": c.get("need_table", False),
                 "top_k": c.get("top_k"), "reference_doc": c.get("reference_doc", ""),
                 "reference_docs": c.get("reference_docs", []),
                 "expected_pages": c.get("expected_pages", []),
                 "answer": f"[执行失败] {e}", "refused": True, "citations": [],
                 "understanding": {}, "retrieved": [], "retrieved_docs": [],
                 "retrieved_pages": [], "n_table_chunks": 0, "n_docs_covered": 0,
                 "n_docs_expected": len(c.get("reference_docs", [])),
                 "doc_coverage": 0.0, "hit_reference_doc": False,
                 "rank_of_reference_doc": 0, "page_hits": [], "page_hit_rate": None,
                 "timings": {}, "latency": 0.0, "error": str(e)}
        results.append(r)
        print(f"  命中主责文档：{'是' if r['hit_reference_doc'] else '否'}"
              f"（排名 {r['rank_of_reference_doc'] or '—'}）| "
              f"检索 {len(r['retrieved'])} 段 | 耗时 {r['latency']:.2f}s")

    summary = summarize(results)
    out_dir = ensure_results_dir()
    save(results, summary, out_dir)

    print(f"\n===== 测试完成 =====")
    print(f"检索命中率 {summary['retrieval_hit_rate']:.2%} | "
          f"拒答 {summary['refused_count']} 题 | 平均耗时 {summary['avg_latency']:.2f}s")
    llm.print_usage("LLM 用量：")
    print(f"结果已保存 → {out_dir / 'rag_test_results.json'} / rag_test_results.md")


if __name__ == "__main__":
    main()
