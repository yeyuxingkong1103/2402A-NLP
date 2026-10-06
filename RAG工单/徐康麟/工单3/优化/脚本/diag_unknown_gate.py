# -*- coding: utf-8 -*-
"""T9 诊断：无关问题为何未被拒答（「不清楚」正确率的根因定位）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

背景（实测）：``测试/测试数据/unknown_questions.jsonl`` 6 条负例里只有 N-GEN1 被拒答，
其余 5 条给出了正文答案（含 N-4b 引到 PDF2 物理 340 的银行借款 300 万美元——正是红线⑨）。
本脚本逐条打印**可答性闸门的原始判据**（top_score / keyword_coverage / 阈值 / 命中块），
把「为什么没拒」变成可复算的数据，而不是猜测。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/diag_unknown_gate.py
    pwsh -NoProfile -File run_py.ps1 优化/脚本/diag_unknown_gate.py --id N-GEN2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_DIR = REPO_ROOT / "研发"
sys.path.insert(0, str(DEV_DIR))

from app.core.answerability import decide  # noqa: E402
from app.core.config import get_config  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.qa_engine import build_engine  # noqa: E402
from app.core.query_understanding import understand  # noqa: E402

UNKNOWN_FIXTURE = REPO_ROOT / "测试" / "测试数据" / "unknown_questions.jsonl"
OUT_JSON = REPO_ROOT / "优化" / "评估结果" / "过程日志" / "_diag_unknown_gate.json"


def main(argv: Sequence[str] | None = None) -> int:
    """入口：逐条负例打印闸门判据与实际回答（只读，不改产品状态）。"""
    parser = argparse.ArgumentParser(description="无关问题拒答失败的根因诊断")
    parser.add_argument("--id", default="", help="只诊断某一条（如 N-GEN2）")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("diag_unknown_gate")
    with log.enter("main", {"only": args.id, "top_k": args.top_k}) as span:
        cases = [json.loads(line) for line in UNKNOWN_FIXTURE.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if args.id:
            cases = [case for case in cases if str(case.get("id")) == args.id]
        engine = build_engine(cfg=cfg, warmup=True, logger=log)
        rows: list[dict[str, Any]] = []
        for case in cases:
            file_names = list(case.get("file_names") or []) or None
            started = time.perf_counter()
            result = engine.retrieve_only(str(case["question"]), file_names=file_names, top_k=args.top_k)
            info = understand(str(case["question"]), [], cfg=cfg, llm=None, logger=log)
            decision = decide(str(case["question"]), result, field_type=str(info.get("field_type") or "other"),
                              expects_numeric=bool(info.get("expects_numeric")), cfg=cfg, logger=log)
            answer = engine.ask(str(case["question"]), file_names=file_names, top_k=args.top_k)
            row = {
                "id": case.get("id"), "question": case["question"], "file_names": file_names,
                "expects_numeric": bool(info.get("expects_numeric")), "field_type": info.get("field_type"),
                "gate": decision.to_dict(),
                "thresholds": {"min_score": cfg.answer.min_score,
                               "min_keyword_coverage": cfg.answer.min_keyword_coverage},
                "top_chunks": [{"chunk_id": getattr(c, "chunk_id", ""), "file_name": getattr(c, "file_name", ""),
                                "page": getattr(c, "page", 0), "type": getattr(c, "type", ""),
                                "score": round(float(getattr(c, "score", 0.0)), 5),
                                "head": str(getattr(c, "content", ""))[:90].replace("\n", " ")}
                               for c in list(getattr(result, "chunks", []) or [])[:5]],
                "answer_text": answer.text, "is_unknown": bool(answer.is_unknown),
                "unknown_reason": answer.unknown_reason,
                "citations": [c.render() for c in (answer.citations or [])],
                "must_not_contain_hit": [t for t in (case.get("must_not_contain") or []) if t in answer.text],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            rows.append(row)
            flag = "✅拒答" if row["is_unknown"] else "❌作答"
            print(f"{flag} [{row['id']}] gate={row['gate']} 阈值={row['thresholds']} "
                  f"引用={row['citations']} 禁用串命中={row['must_not_contain_hit']}")
            print(f"    答：{row['answer_text'][:110]}")
            for chunk in row["top_chunks"][:3]:
                print(f"    块 {chunk['chunk_id']} {chunk['file_name']} p{chunk['page']} "
                      f"{chunk['type']} score={chunk['score']} :: {chunk['head'][:70]}")
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps({"work_order": WORK_ORDER, "rows": rows},
                                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n诊断 JSON：{OUT_JSON.relative_to(REPO_ROOT)}")
        span.set_output({"cases": len(rows), "refused": sum(1 for row in rows if row["is_unknown"])})
    shutdown_logging()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 —— 顶层兜底
        try:
            get_logger("diag_unknown_gate").log_event("diag.failed", level="ERROR",
                                                      error_type=type(exc).__name__, message=str(exc),
                                                      stack=__import__("traceback").format_exc())
        finally:
            import traceback

            traceback.print_exc()
        raise SystemExit(1)
