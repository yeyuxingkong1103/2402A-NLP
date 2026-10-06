# -*- coding: utf-8 -*-
"""T5 检索评估：14 题「证据原文是否落进 top-5」实测（判据 = 证据原文命中返回块）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

判据（硬性）：用 ``retrieval_utils.is_evidence_hit`` → ``text_utils.evidence_contains``，
**不得**用「引用页 == evidence_pages」判定命中。同时记录「证据页是否出现在召回块页码中」作为**辅助诊断**，
两者必须分开呈现，避免把页码巧合当命中。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/离线/eval_retrieval.py            # 全库检索
    pwsh -NoProfile -File run_py.ps1 测试/离线/eval_retrieval.py --file-filter  # 额外跑按文件过滤
退出码：0（评估总是产出报告）；出现基础设施错误时为 1。
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

from app.core import retrieval_utils, text_utils  # noqa: E402
from app.core.config import get_config  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.retriever import build_retriever  # noqa: E402

EVAL_SET = REPO_ROOT / "测试" / "测试数据" / "eval_retrieval_14.jsonl"
OUT_DIR = REPO_ROOT / "优化" / "评估结果"


def main(argv: list[str] | None = None) -> int:
    """跑 14 题检索并输出命中率与逐题明细。"""
    parser = argparse.ArgumentParser(description="T5 检索评估（14 题 top-5 证据命中）")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", default=str(OUT_DIR / "retrieval_eval_t5.json"))
    parser.add_argument("--md", default=str(OUT_DIR / "retrieval_eval_t5.md"))
    parser.add_argument("--file-filter", action="store_true", help="额外验证按 file_name 过滤")
    args = parser.parse_args(argv)

    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("eval_retrieval")
    questions = [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"题集：{EVAL_SET}（{len(questions)} 题）")
    retriever = build_retriever(cfg=cfg)            # load() 内已预热分词器与嵌入后端
    health = retriever.health()
    print(f"检索器：{health['chunks']} 块 / 向量 {health['vectors']}×{health['dim']} / "
          f"BM25 词表 {health['bm25_vocab']} / 文件 {health['files']}")
    if health.get("warmup"):
        tk = health["warmup"].get("tokenizer", {})
        print(f"预热：jieba 冷 {tk.get('cold_ms')} ms → 热 {tk.get('warm_ms')} ms")

    results: list[dict] = []
    for item in questions:
        target = [item["file_name"]] if args.file_filter else None
        started = time.perf_counter()
        result = retriever.retrieve(item["question"], top_k=args.top_k, file_names=target, logger=log)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        # **硬校验**：按文件过滤时，返回块必须全部属于所选文件（不得夹带另一份 PDF 的块）
        filter_violations: list[str] = []
        if target:
            filter_violations = [c.chunk_id for c in result.chunks if c.file_name not in set(target)]
        report = retriever.retrieve_evidence(result.chunks, item["evidence"], question_id=item["id"],
                                             top_k=args.top_k)
        # 设计允许「同一事实的多条原文依据」：主证据未命中时，再看备选原文（如题 207 的 479 表行 / 490 正文）
        alternatives = list(item.get("evidence_alternatives") or [])
        alt_reports = [retriever.retrieve_evidence(result.chunks, ev, question_id=item["id"],
                                                   top_k=args.top_k) for ev in alternatives]
        hit_any = report.hit or any(r.hit for r in alt_reports)
        matched_any = list(report.matched_chunk_ids)
        for alt in alt_reports:
            matched_any.extend(alt.matched_chunk_ids)
        retrieved_pages = sorted({c.page for c in result.chunks})
        page_hit = any(p in retrieved_pages for p in item.get("evidence_pages") or [])
        results.append({
            "id": item["id"], "question": item["question"], "subject": item.get("subject"),
            "file_name": item["file_name"], "evidence_pages": item.get("evidence_pages"),
            "baseline_ok": item.get("baseline_ok"),
            "hit": report.hit, "hit_any_evidence": hit_any,
            "hit_evidence": item["evidence"][:60] if report.hit else None,
            "matched_chunk_ids": report.matched_chunk_ids,
            "matched_chunk_ids_any": sorted(set(matched_any)),
            "evidence_alternatives": alternatives,
            "retrieved_pages": retrieved_pages, "page_hit_diagnostic": page_hit,
            "filter_violations": filter_violations,
            "elapsed_ms": elapsed_ms, "stages": result.stages,
            "table_fallback_used": result.table_fallback_used,
            "top_chunks": [
                {"rank": c.rank, "chunk_id": c.chunk_id, "file_name": c.file_name, "page": c.page,
                 "type": c.type, "score": round(c.score, 6), "vector_score": c.vector_score,
                 "bm25_score": c.bm25_score, "boost": c.boosts.get("table"),
                 "content_head": c.content[:90].replace("\n", " ")}
                for c in result.chunks
            ],
        })
        flag = "✅" if hit_any else "❌"
        strict_flag = "" if hit_any == report.hit else ("  （严格单证据未命中，主锚点备选依据命中）" if hit_any else "  （含备选依据仍未命中）")
        print(f"  {flag} id={item['id']:<4} {elapsed_ms:7.1f} ms  "
              f"命中块={sorted(set(matched_any)) or '-'}  召回页码={retrieved_pages}{strict_flag}")

    hits = [r for r in results if r["hit_any_evidence"]]
    hits_strict = [r for r in results if r["hit"]]
    pdf1 = [r for r in results if r["file_name"].endswith("1.pdf")]
    pdf2 = [r for r in results if r["file_name"].endswith("2.pdf")]
    baseline_fail = [r for r in results if r.get("baseline_ok") is False]
    summary = {
        "total": len(results), "hit": len(hits),
        "hit_rate": round(len(hits) / len(results), 4) if results else 0.0,
        "hit_strict_single_evidence": len(hits_strict),
        "hit_rate_strict_single_evidence": round(len(hits_strict) / len(results), 4) if results else 0.0,
        "strict_miss_but_alt_hit": [r["id"] for r in results if r["hit_any_evidence"] and not r["hit"]],
        "pdf1": {"total": len(pdf1), "hit": sum(1 for r in pdf1 if r["hit_any_evidence"])},
        "pdf2": {"total": len(pdf2), "hit": sum(1 for r in pdf2 if r["hit_any_evidence"])},
        "miss_ids": [r["id"] for r in results if not r["hit_any_evidence"]],
        "baseline_failures": {
            str(r["id"]): {"hit": r["hit_any_evidence"], "evidence_pages": r["evidence_pages"],
                           "retrieved_pages": r["retrieved_pages"],
                           "matched_chunk_ids": r["matched_chunk_ids_any"]}
            for r in baseline_fail
        },
        "page_hit_only": [r["id"] for r in results if r["page_hit_diagnostic"] and not r["hit_any_evidence"]],
        "filter_violations": {str(r["id"]): r["filter_violations"] for r in results if r.get("filter_violations")},
        "file_filter_mode": bool(args.file_filter),
        "top_k": args.top_k,
        "retriever_health": health,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    payload = {"summary": summary, "results": results}
    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# T5 检索评估（{len(results)} 题，top-{args.top_k} 证据原文命中）",
        "",
        f"> 判据：`text_utils.evidence_contains`（证据原文是否落在返回 chunk 中）；"
        f"**不是**「引用页 == evidence_pages」。",
        f"> 生成时间：{summary['generated_at']}　检索模式：{'按文件过滤' if args.file_filter else '全库'}",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 命中率 | **{len(hits)}/{len(results)} = {summary['hit_rate'] * 100:.1f}%** |",
        f"| PDF1（10 题） | {summary['pdf1']['hit']}/{summary['pdf1']['total']} |",
        f"| PDF2（4 题） | {summary['pdf2']['hit']}/{summary['pdf2']['total']} |",
        f"| 未命中题号 | {summary['miss_ids'] or '无'} |",
        f"| 仅页码巧合（页码命中但原文未命中，**不算命中**） | {summary['page_hit_only'] or '无'} |",
        "",
        "## 工单2 基线失败题（33/95/531/793/957）逐题复核",
        "",
        "| 题号 | 本次是否命中 | 证据页 | 召回页码 | 命中的 chunk |",
        "| --- | --- | --- | --- | --- |",
    ]
    for rid, info in summary["baseline_failures"].items():
        lines.append(f"| {rid} | {'✅ 已命中' if info['hit'] else '❌ 未命中'} | {info['evidence_pages']} | "
                     f"{info['retrieved_pages']} | {info['matched_chunk_ids'] or '-'} |")
    lines += ["", "## 逐题明细", "",
              "| 题号 | 命中 | 耗时 ms | 证据页 | 召回页码 | top-1 chunk | top-1 页 | 类型 | 分数 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        top = r["top_chunks"][0] if r["top_chunks"] else {}
        lines.append(f"| {r['id']} | {'✅' if r['hit'] else '❌'} | {r['elapsed_ms']} | {r['evidence_pages']} | "
                     f"{r['retrieved_pages']} | {top.get('chunk_id', '-')} | {top.get('page', '-')} | "
                     f"{top.get('type', '-')} | {top.get('score', '-')} |")
    out_md = Path(args.md)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"命中率：{len(hits)}/{len(results)} = {summary['hit_rate'] * 100:.1f}%"
          f"（PDF1 {summary['pdf1']['hit']}/{summary['pdf1']['total']}，PDF2 {summary['pdf2']['hit']}/{summary['pdf2']['total']}）")
    print(f"严格单证据口径：{summary['hit_strict_single_evidence']}/{len(results)}"
          f"（差集 = {summary['strict_miss_but_alt_hit']}，属「主锚点备选原文命中」）")
    print(f"未命中：{summary['miss_ids'] or '无'}")
    print("基线失败题：", {k: ("已命中" if v["hit"] else "未命中") for k, v in summary["baseline_failures"].items()})
    if args.file_filter:
        bad = summary["filter_violations"]
        print(f"按文件过滤硬校验：{'✅ 无越界块（返回块全部属于所选文件）' if not bad else f'❌ 越界：{bad}'}")
    print(f"JSON：{out_json}\nMD  ：{out_md}")
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
