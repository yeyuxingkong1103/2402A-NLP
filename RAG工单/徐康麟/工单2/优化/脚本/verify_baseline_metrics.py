# -*- coding: utf-8 -*-
"""基线判分复核：用**工单1 原始 `evaluator.py`** 重跑评估运行记录里的每条回答，独立复算基线准确率。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 基线采集（复用判分口径，固化优化前指标）

用途（T9 会重点核查的三点之一）：
    1. 证明「优化前 50%」是被**同一把尺子**量出来的——不誊抄、不放宽；
    2. 逐题判定 ✅34/207/260/543/795、❌33/95/531/793/957 可被独立复算；
    3. 输出 `baseline_metrics.json` 作为优化前后对比的基线侧输入。

隔离说明：判分逻辑来自 `优化/基线/repro/app`（工单1 基线代码的**只读副本**），
         `RAG_PATHS__*` 全部指向 repro，绝不写工单1。

复现命令（工作目录 = E:\\gao6gongdan\\工单2）：

    pwsh -NoProfile -File run_py.ps1 -B 优化/脚本/verify_baseline_metrics.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True

W1 = Path(r"E:\gao6gongdan\工单1")
W2 = Path(r"E:\gao6gongdan\工单2")
REPRO = W2 / "优化" / "基线" / "repro"
OUT = W2 / "优化" / "基线" / "baseline_metrics.json"

EVAL_RECORDS = W1 / "优化" / "评估结果" / "eval_results" / "eval_records.json"

os.environ.update(
    {
        "PYTHONDONTWRITEBYTECODE": "1",
        "RAG_PATHS__DATA_RAW": str(REPRO / "data" / "raw"),
        "RAG_PATHS__DATA_PROCESSED": str(REPRO / "data" / "processed"),
        "RAG_PATHS__DATA_INDEX": str(REPRO / "data" / "index"),
        "RAG_PATHS__DATA_EVAL": str(REPRO / "data" / "eval"),
        "RAG_PATHS__EVAL_RESULTS": str(REPRO / "eval_results"),
        "RAG_PATHS__LOGS": str(REPRO / "logs"),
        "RAG_PATHS__SQLITE_PATH": str(REPRO / "data" / "index" / "rag.sqlite3"),
        "RAG_EMBEDDING__LOCAL_MODEL_DIR": str(W1 / "models" / "bge-small-zh-v1.5"),
        "RAG_EMBEDDING__MODEL_NAME": str(W1 / "models" / "bge-small-zh-v1.5"),
    }
)
sys.path.insert(0, str(REPRO))


def main() -> int:
    from app.core.config import get_settings
    from app.core.evaluator import FUZZY_THRESHOLD, Evaluator

    settings = get_settings()
    assert str(settings.paths.logs).startswith(str(REPRO)), "日志路径未隔离到 repro，拒绝继续"

    payload = json.loads(EVAL_RECORDS.read_text(encoding="utf-8"))
    summaries, records = payload["summaries"], payload["records"]

    evaluator = Evaluator()

    rows = []
    for rec in records:
        if rec.get("mode") != "rag":
            continue
        recomputed, reason = evaluator.check_answer(rec["answer"], rec["golden"])
        rows.append(
            {
                "question_id": int(rec["question_id"]),
                "answer": rec["answer"],
                "golden": rec["golden"],
                "is_correct_recorded": bool(rec["is_correct"]),
                "is_correct_recomputed": bool(recomputed),
                "consistent": bool(recomputed) == bool(rec["is_correct"]),
                "judge_reason": reason,
                "citation_pages": rec.get("citation_pages", []),
                "citation_valid": bool(rec.get("citation_valid")),
                "first_token_ms": rec.get("first_token_ms"),
                "total_ms": rec.get("total_ms"),
            }
        )

    corrected = [r["question_id"] for r in rows if r["is_correct_recomputed"]]
    wrong = [r["question_id"] for r in rows if not r["is_correct_recomputed"]]
    inconsistent = [r["question_id"] for r in rows if not r["consistent"]]
    accuracy = round(len(corrected) / len(rows), 4) if rows else 0.0
    recorded_accuracy = float(summaries["rag"]["accuracy"])

    ok = (not inconsistent) and abs(accuracy - recorded_accuracy) < 1e-9

    OUT.write_text(
        json.dumps(
            {
                "meta": {
                    "工单": "人工智能NLP-RAG-基于PDF文档的问答系统优化",
                    "阶段": "优化 / 基线采集（判分复核）",
                    "source_records": str(EVAL_RECORDS),
                    "judge": "工单1 app/core/evaluator.py::Evaluator.check_answer（只读副本 repro）",
                    "FUZZY_THRESHOLD": FUZZY_THRESHOLD,
                    "reproduce_cmd": "pwsh -NoProfile -File run_py.ps1 -B 优化/脚本/verify_baseline_metrics.py",
                    "说明": "answer/golden 取自 eval_records.json 的原始记录，重新过一遍判分函数；"
                    "未修改、未放宽任何阈值。",
                },
                "baseline_metrics": {
                    "accuracy": accuracy,
                    "accuracy_count": f"{len(corrected)}/{len(rows)}",
                    "correct_question_ids": corrected,
                    "wrong_question_ids": wrong,
                    "recorded_accuracy": recorded_accuracy,
                    "recorded_correct_ids": [
                        int(r["question_id"]) for r in records if r.get("mode") == "rag" and r["is_correct"]
                    ],
                    "unknown_accuracy": summaries["rag"]["unknown_accuracy"],
                    "citation_accuracy": summaries["rag"]["citation_accuracy"],
                    "citation_total": summaries["rag"]["citation_total"],
                    "citation_valid": summaries["rag"]["citation_valid"],
                    "first_token_avg_ms": summaries["rag"]["first_token_avg_ms"],
                    "first_token_p95_ms": summaries["rag"]["first_token_p95_ms"],
                    "first_token_max_ms": summaries["rag"]["first_token_max_ms"],
                    "total_avg_ms": summaries["rag"]["total_avg_ms"],
                    "rag_en": summaries.get("rag_en"),
                    "ragas": "未运行（ragas 依赖不可用且本机断网）；eval_records 中 ragas 四项指标均为 null",
                },
                "verification": {
                    "recomputed_accuracy_matches_recorded": abs(accuracy - recorded_accuracy) < 1e-9,
                    "all_per_question_judgements_reproduced": not inconsistent,
                    "overall_ok": ok,
                },
                "per_question": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"FUZZY_THRESHOLD = {FUZZY_THRESHOLD}（未改动）")
    print(f"{'题号':<6}{'记录判定':<10}{'复算判定':<10}{'一致':<6}{'引用页':<12}{'判分理由'}")
    for r in rows:
        print(
            f"{r['question_id']:<6}{('✅' if r['is_correct_recorded'] else '❌'):<8}"
            f"{('✅' if r['is_correct_recomputed'] else '❌'):<8}"
            f"{('✅' if r['consistent'] else '❌'):<4}{str(r['citation_pages'])[:10]:<12}{r['judge_reason'][:60]}"
        )
    print(f"\n复算准确率 = {accuracy} ({len(corrected)}/{len(rows)})；记录准确率 = {recorded_accuracy}")
    print(f"逐题判定全部复现: {'✅' if not inconsistent else '❌ ' + str(inconsistent)}")
    print(f"总体核验: {'✅ 通过' if ok else '❌ 不通过'}")
    print(f"输出: {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[FATAL] 基线判分复核失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
