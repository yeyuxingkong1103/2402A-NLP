# -*- coding: utf-8 -*-
"""T6 问题诊断：对被标记的题目打印「检索块 → 提示词 → LLM 原始输出 → 引用校验」全链路证据。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

用途：定位「不清楚」「引用不可回溯」的真实根因（检索错页 / 生成错引 / 校验过严），
作为修复依据（禁止凭猜测改代码）。只读检索与生成，不改任何数据。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/离线/diag_flagged.py --ids 531,543,795,34,793,957
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import answerability, citation as citation_mod, text_utils  # noqa: E402
from app.core.config import get_config  # noqa: E402
from app.core.generator import build_generator  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.query_understanding import understand  # noqa: E402
from app.core.retriever import build_retriever  # noqa: E402

EVAL_SET = REPO_ROOT / "测试" / "测试数据" / "eval_retrieval_14.jsonl"


def main(argv: list[str] | None = None) -> int:
    """对被标记题打印全链路证据。"""
    parser = argparse.ArgumentParser(description="T6 标记题全链路诊断")
    parser.add_argument("--ids", default="531,543,795,34,793,957")
    parser.add_argument("--show-chars", type=int, default=220)
    args = parser.parse_args(argv)
    wanted = {s.strip() for s in args.ids.split(",") if s.strip()}

    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("diag_flagged")
    items = [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    conn = sqlite3.connect(str(cfg.paths.index_dir / "rag.sqlite3"))

    retriever = build_retriever(cfg=cfg)
    generator = build_generator(cfg=cfg, logger=log)

    for item in items:
        if str(item["id"]) not in wanted:
            continue
        print("=" * 100)
        print(f"### 题 {item['id']}：{item['question']}")
        retrieval = retriever.retrieve(item["question"], top_k=cfg.retrieval.top_k, logger=log)
        info = understand(item["question"], None, cfg=cfg, llm=None, logger=log)
        decision = answerability.decide(item["question"], retrieval, field_type=info["field_type"],
                                       expects_numeric=info["expects_numeric"], cfg=cfg, logger=log)
        print(f"-- 理解：field_type={info['field_type']} expects_numeric={info['expects_numeric']} "
              f"expected_subject={info.get('expected_subject')}")
        print(f"-- 可答性：answerable={decision.is_answerable} reason={decision.reason} "
              f"score={decision.top_score} coverage={decision.keyword_coverage}")
        for rank, chunk in enumerate(retrieval.chunks, start=1):
            snippet = text_utils.squash_text(getattr(chunk, "content", ""))[:args.show_chars]
            print(f"   [{rank}] {chunk.chunk_id} {chunk.file_name}:p{chunk.page} type={chunk.type} "
                  f"score={getattr(chunk, 'score', 0):.4f} | {snippet}")
        # 提示词 + LLM 原始输出（复现 answer() 的调用序列）
        lang = "zh"
        prompt = generator.build_prompt(item["question"], retrieval.chunks, history=None, language=lang)
        raw = ""
        if generator.llm is not None:
            res = generator.llm.generate_full(prompt, trace_id=f"dg{item['id']}", logger=log)
            raw = res.text
            print(f"-- LLM 裸输出（无前置引导，首字 {res.first_token_ms} ms）：{raw!r}")
        text, first_ms, total_ms = generator._generate_text(prompt, item["question"], retrieval.chunks,
                                                            lang, f"dg{item['id']}", log)
        print(f"-- 生产口径输出（带「答案正文：」前置引导，首字 {first_ms} ms / 总 {total_ms} ms）：{text!r}")
        cleaned = generator.sanitize_answer(text, trace=f"dg{item['id']}", logger=log)
        print(f"-- 清洗后：{cleaned!r}  空答案={citation_mod.is_empty_answer(cleaned)}")
        cites, report, actions = generator._resolve_citations(cleaned, retrieval.chunks,
                                                             trace=f"dg{item['id']}", log=log, info=info)
        gate = answerability.subject_gate(item["question"], cleaned, retrieval.chunks, issuer_names=None,
                                          cfg=cfg, logger=log)
        print(f"-- 引用落位：{[c.render() for c in cites]}")
        print(f"-- 引用动作：{actions}")
        print(f"-- 校验：total={report.total} valid={report.valid} reasons={report.reasons}")
        print(f"-- 主体闸门：ok={gate.ok} reason={gate.reason} leaked={list(gate.leaked)}")
        # 逐个返回块量出「答案能否核实」，判断是「核验过严」还是「确实无据」
        for rank, chunk in enumerate(retrieval.chunks, start=1):
            ok, why = citation_mod.answer_support_check(cleaned, str(getattr(chunk, "content", "") or ""))
            print(f"   [块{rank}] {chunk.chunk_id} p{chunk.page} 可核实={ok} {why}")
        # golden 证据页里到底有没有答案片段（区分「检索错页」与「校验过严」）
        for page in item.get("evidence_pages") or []:
            row = conn.execute("SELECT text FROM pages WHERE file_name=? AND page=?",
                               (item["file_name"], int(page))).fetchone()
            ptext = str(row[0]) if row else ""
            hit = text_utils.squash_text(str(item.get("evidence") or ""))[:60] in text_utils.squash_text(ptext)
            print(f"   golden p{page}：证据前 60 字在该页={hit}，页长={len(ptext)}")
        print()
    conn.close()
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
