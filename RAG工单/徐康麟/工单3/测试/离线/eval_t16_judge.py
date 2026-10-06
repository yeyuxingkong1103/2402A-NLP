# -*- coding: utf-8 -*-
"""t16 快速判分回路：14 题离线作答 → **工单1 官方判分器**（正文口径 + 原文口径）逐题判定。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

定位：诊断用快回路（不落正式产物）。**验收口径 = 官方判分器对 `answer_body`（去引用标签）的判定**，
另有更严的 `answer_raw_text`（含引用标签）口径一并报出，供「不劣化」回归比对。

为什么单独写：
    * 正式评估（研发/scripts/evaluate.py）含 12 条无关问题与报告落盘，单轮 3~4 分钟，
      迭代改法时需要分钟级回路；
    * 本脚本复用 evaluate.py 的 ``judge_pairs``（内部走 优化/脚本/ref_judge_runner.py 只读子进程，
      与 T9 正式产物**同一判分链路**），因此数值可对齐正式产物。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/离线/eval_t16_judge.py --tag t16_a
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_DIR = REPO_ROOT / "研发"
sys.path.insert(0, str(DEV_DIR))
sys.path.insert(0, str(DEV_DIR / "scripts"))

from app.core import citation as citation_mod  # noqa: E402
from app.core.config import get_config  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.qa_engine import build_engine  # noqa: E402
from evaluate import judge_pairs, load_golden, percentile_nearest_rank  # noqa: E402

DEFAULT_GOLDEN = DEV_DIR / "data" / "eval" / "golden_qa.jsonl"
PROCESS_LOG_DIR = REPO_ROOT / "优化" / "评估结果" / "过程日志"


def run(tag: str, *, top_k: int, limit: int) -> int:
    """跑 14 题并判分；返回进程退出码（0 成功）。"""
    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("eval_t16_judge")
    started = time.perf_counter()
    with log.enter("run", {"tag": tag, "top_k": top_k, "limit": limit}) as span:
        golden = load_golden(DEFAULT_GOLDEN, logger=log)
        engine = build_engine(cfg=cfg, warmup=True, logger=log)
        items = list(golden)[:limit] if limit else list(golden)
        rows: list[dict[str, object]] = []
        for index, item in enumerate(items):
            t0 = time.perf_counter()
            answer = engine.ask(str(item["question"]), top_k=top_k)
            body = citation_mod.answer_body(answer.text)
            rows.append({
                "id": int(item["id"]),
                "question": str(item["question"]),
                "answer": answer.text,
                "answer_body": body,
                "citations": [cite.render(language=answer.language) for cite in answer.citations],
                "is_unknown": bool(answer.is_unknown),
                "first_token_ms": round(float(answer.first_token_ms or 0.0), 2),
                "total_ms": round(float(answer.total_ms or 0.0), 2),
                "wall_ms": round((time.perf_counter() - t0) * 1000, 2),
                "golden_answer": str(item["answer"]),
                "warm": index > 0,
            })
        body_verdicts, source = judge_pairs(
            [{"id": row["id"], "answer": row["answer_body"], "golden": row["golden_answer"]} for row in rows],
            logger=log)
        raw_verdicts, _ = judge_pairs(
            [{"id": row["id"], "answer": row["answer"], "golden": row["golden_answer"]} for row in rows],
            logger=log)
        for row in rows:
            verdict = body_verdicts.get(row["id"], {"ok": None, "reason": "未判分"})
            row["correct"] = bool(verdict.get("ok"))
            row["judge_reason"] = verdict.get("reason")
            raw = raw_verdicts.get(row["id"], {})
            row["correct_raw"] = bool(raw.get("ok"))
            row["judge_reason_raw"] = raw.get("reason")

        total = len(rows)
        correct = sum(1 for row in rows if row["correct"])
        correct_raw = sum(1 for row in rows if row["correct_raw"])
        first_tokens = [float(row["first_token_ms"]) for row in rows]
        payload = {
            "work_order": WORK_ORDER,
            "tag": tag,
            "judge_source": source,
            "acceptance_note": "验收口径 = 官方判分器(工单1 Evaluator.check_answer) 判 answer_body（去引用标签）",
            "total": total,
            "correct_body": correct,
            "accuracy_body": round(correct / total, 4) if total else 0.0,
            "incorrect_ids_body": [row["id"] for row in rows if not row["correct"]],
            "correct_raw_text": correct_raw,
            "accuracy_raw_text": round(correct_raw / total, 4) if total else 0.0,
            "incorrect_ids_raw_text": [row["id"] for row in rows if not row["correct_raw"]],
            "unknown_ids": [row["id"] for row in rows if row["is_unknown"]],
            "citation_total": sum(len(row["citations"]) for row in rows),
            "citation_rows": sum(1 for row in rows if row["citations"]),
            "first_token_max_ms": max(first_tokens) if first_tokens else None,
            "first_token_p95_ms": percentile_nearest_rank(first_tokens, 0.95),
            "first_token_within_budget": bool(first_tokens) and max(first_tokens) <= 3000.0,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "rows": rows,
        }
        PROCESS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESS_LOG_DIR / f"_t16_judge_{tag}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print("=" * 100)
        print(f"[t16:{tag}] 官方判分器口径  body {correct}/{total}={payload['accuracy_body'] * 100:.1f}%"
              f"（错题 {payload['incorrect_ids_body'] or '无'}）"
              f" | raw_text {correct_raw}/{total}={payload['accuracy_raw_text'] * 100:.1f}%"
              f"（错题 {payload['incorrect_ids_raw_text'] or '无'}）")
        print(f"作答 {total - len(payload['unknown_ids'])}/{total}；误拒答 {payload['unknown_ids'] or '无'}；"
              f"引用行 {payload['citation_rows']}/{total}；首字 max {payload['first_token_max_ms']} ms")
        print(f"判分链路 {source}")
        print("-" * 100)
        for row in rows:
            mark = "✅" if row["correct"] else "❌"
            mark_raw = "✅" if row["correct_raw"] else "❌"
            print(f"{mark}/raw{mark_raw} [{row['id']}] {row['question']}")
            print(f"    答案: {str(row['answer_body'])[:300]}")
            print(f"    引用: {row['citations']}  首字 {row['first_token_ms']} ms")
            if not row["correct"]:
                print(f"    ❌判分说明: {row['judge_reason']}")
            if row["correct"] != row["correct_raw"]:
                print(f"    raw口径说明: {row['judge_reason_raw']}")
        print(f"产物 {out_path}（{out_path.stat().st_size} B）")
        span.set_output({"correct_body": correct, "correct_raw_text": correct_raw})
    shutdown_logging()
    return 0


def main(argv: list[str] | None = None) -> int:
    """入口：解析参数 → 跑回路。"""
    parser = argparse.ArgumentParser(description="t16 官方判分器快速回路")
    parser.add_argument("--tag", default="t16")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        return run(args.tag, top_k=args.top_k, limit=args.limit)
    except Exception as exc:  # noqa: BLE001 —— 顶层兜底：非零退出，绝不静默
        import traceback

        print(json.dumps({"event": "eval_t16_judge.failed", "level": "ERROR",
                          "error_type": type(exc).__name__, "message": str(exc),
                          "stack": traceback.format_exc()}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
