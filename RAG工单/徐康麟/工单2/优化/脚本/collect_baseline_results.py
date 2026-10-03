# -*- coding: utf-8 -*-
"""T6 产出①：固化优化前基线结果 → 优化/基线/baseline_results.json

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 基线采集

数据来源（**只读工单1**，严禁修改）：
    工单1\\优化\\评估结果\\eval_results\\eval_records.json   —— 逐题真值（唯一来源）
    工单1\\优化\\评估结果\\eval_results\\ragas_report.md      —— 环境信息与人工可读报告（交叉核对）
    工单1\\data\\eval\\golden_qa.jsonl                        —— 问题/参考答案

判分口径复核：用 工单1 原始 `Evaluator.check_answer`（只读副本 优化/基线/repro/app）
              对 10 条原始回答**重算一遍**，并与记录值逐题比对，确保「同一把尺子」。

复现命令（工作目录 = E:\\gao6gongdan\\工单2）：
    pwsh -NoProfile -File run_py.ps1 -B 优化/脚本/collect_baseline_results.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True

W1 = Path(r"E:\gao6gongdan\工单1")
W2 = Path(r"E:\gao6gongdan\工单2")
REPRO = W2 / "优化" / "基线" / "repro"
OUT = W2 / "优化" / "基线" / "baseline_results.json"

EVAL_RECORDS = W1 / "优化" / "评估结果" / "eval_results" / "eval_records.json"
RAGAS_REPORT = W1 / "优化" / "评估结果" / "eval_results" / "ragas_report.md"
GOLDEN = W1 / "data" / "eval" / "golden_qa.jsonl"

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


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_report_env(text: str) -> dict:
    """从 ragas_report.md 的 §0 环境表里取键值对。"""
    env: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("|") and line.count("|") >= 3:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 2 and cells[0] and cells[0] != "项目" and not set(cells[0]) <= set("-: "):
                env[cells[0]] = cells[1]
    return env


def main() -> int:
    from app.core.config import get_settings
    from app.core.evaluator import FUZZY_THRESHOLD, Evaluator

    settings = get_settings()
    assert str(settings.paths.logs).startswith(str(REPRO)), "日志路径未隔离到 repro，拒绝继续"
    for f in (EVAL_RECORDS, RAGAS_REPORT, GOLDEN):
        if not f.exists():
            print(f"[FATAL] 缺少源文件: {f}", file=sys.stderr)
            return 2

    payload = json.loads(EVAL_RECORDS.read_text(encoding="utf-8"))
    summaries = payload["summaries"]
    records = payload["records"]
    golden = {
        json.loads(l)["id"]: json.loads(l)
        for l in GOLDEN.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    report_text = RAGAS_REPORT.read_text(encoding="utf-8")
    report_env = parse_report_env(report_text)

    evaluator = Evaluator()
    rag_rows, en_rows = [], []
    inconsistent: list[int] = []
    for rec in records:
        mode = rec.get("mode")
        row = {
            "question_id": int(rec["question_id"]),
            "question": rec["question"],
            "golden": rec["golden"],
            "baseline_answer": rec["answer"],
            "is_correct": bool(rec["is_correct"]),
            "citation_pages": rec.get("citation_pages", []),
            "citation_valid": bool(rec.get("citation_valid")),
            "first_token_ms": rec.get("first_token_ms"),
            "total_ms": rec.get("total_ms"),
            "is_unknown": bool(rec.get("is_unknown")),
            "unknown_reason": rec.get("unknown_reason", ""),
            "mode": mode,
        }
        if mode == "rag":
            ok, reason = evaluator.check_answer(rec["answer"], rec["golden"])
            row["recomputed_is_correct"] = bool(ok)
            row["recomputed_reason"] = reason
            row["judgement_consistent"] = bool(ok) == bool(rec["is_correct"])
            if not row["judgement_consistent"]:
                inconsistent.append(row["question_id"])
            row["evidence_pages_in_golden"] = golden.get(row["question_id"], {}).get("evidence_pages", [])
            rag_rows.append(row)
        elif mode == "rag_en":
            en_rows.append(row)

    s = summaries["rag"]
    correct_ids = [r["question_id"] for r in rag_rows if r["is_correct"]]
    accuracy = round(len(correct_ids) / len(rag_rows), 4) if rag_rows else 0.0

    output = {
        "meta": {
            "工单": "人工智能NLP-RAG-基于PDF文档的问答系统优化",
            "阶段": "优化 / 基线采集（T6 产出①）",
            "文件用途": "固化**优化前**基线结果，供 T7/T9 对比；本文件数值全部来自工单1 真实产物 + 本机复算。",
            "采集时间": datetime.now().isoformat(timespec="seconds"),
            "来源文件": [
                {"path": str(EVAL_RECORDS), "sha1": sha1_of(EVAL_RECORDS), "用途": "逐题真值（唯一来源）"},
                {"path": str(RAGAS_REPORT), "sha1": sha1_of(RAGAS_REPORT), "用途": "环境信息与人工可读报告"},
                {"path": str(GOLDEN), "sha1": sha1_of(GOLDEN), "用途": "问题与参考答案"},
            ],
            "判分口径": {
                "实现": "工单1 app/core/evaluator.py::Evaluator.check_answer（确定性规则，不依赖 LLM 裁判）",
                "FUZZY_THRESHOLD": FUZZY_THRESHOLD,
                "阈值是否改动": False,
                "判定顺序": [
                    "1. 去标点后参考答案是答案子串（或反之）",
                    "2. 参考答案的全部金额都出现在答案中（按元折算等价）",
                    "3. 关键实体（引号内名称）有交集",
                    "4. 参考答案的全部百分比都出现在答案中",
                    "5. 字符二元组 Jaccard ≥ 阈值且不缺任何数字",
                ],
                "复核方式": "本机用上述同一函数对 10 条原始回答重算，与记录值逐题比对",
            },
            "运行环境（摘自 ragas_report.md §0，工单1 记录）": report_env,
            "复现命令": "pwsh -NoProfile -File run_py.ps1 -B 优化/脚本/collect_baseline_results.py",
            "配套产物": {
                "检索指标": "优化/基线/baseline_retrieval.json",
                "流水线上下文": "优化/基线/baseline_repro_retrieval.json",
                "回答路径审计": "优化/基线/baseline_answer_paths_generator.json",
                "说明文档": "优化/基线/基线说明.md",
            },
        },
        "summary": {
            "accuracy": accuracy,
            "accuracy_count": f"{len(correct_ids)}/{len(rag_rows)}",
            "correct_question_ids": sorted(correct_ids),
            "wrong_question_ids": sorted(r["question_id"] for r in rag_rows if not r["is_correct"]),
            "unknown_accuracy": s["unknown_accuracy"],
            "citation_accuracy": s["citation_accuracy"],
            "citation_total": s["citation_total"],
            "citation_valid": s["citation_valid"],
            "first_token_avg_ms": s["first_token_avg_ms"],
            "first_token_p95_ms": s["first_token_p95_ms"],
            "first_token_max_ms": s["first_token_max_ms"],
            "total_avg_ms": s["total_avg_ms"],
            "count": s["count"],
            "rag_en": summaries.get("rag_en"),
        },
        "verification": {
            "recomputed_accuracy": accuracy,
            "recorded_accuracy": float(s["accuracy"]),
            "accuracy_matches": abs(accuracy - float(s["accuracy"])) < 1e-9,
            "per_question_judgements_all_reproduced": not inconsistent,
            "inconsistent_question_ids": inconsistent,
            "report_md_cross_check": "ragas_report.md §3 逐题表与 eval_records.json 的 ✅/❌ 完全一致（已人工核对 10/10）",
            "ragas": "未运行（ragas 依赖不可用且本机断网）；eval_records 中 faithfulness/"
            "answer_relevancy/context_precision/context_recall 全为 null，ragas_report.md §2 亦显式标注「未运行」",
        },
        "per_question_rag": rag_rows,
        "per_question_rag_en": en_rows,
    }

    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"来源: {EVAL_RECORDS}")
    print(f"环境（ragas_report.md）: 嵌入={report_env.get('嵌入模型')} / LLM={report_env.get('LLM 模型')} @ {report_env.get('LLM 服务')} / 分块={report_env.get('分块数')}")
    print(f"\n{'题号':<6}{'判定':<6}{'复算':<6}{'一致':<6}{'引用页':<12}{'首字ms':<12}{'端到端ms':<12}")
    for r in rag_rows:
        print(
            f"{r['question_id']:<6}{('✅' if r['is_correct'] else '❌'):<4}"
            f"{('✅' if r['recomputed_is_correct'] else '❌'):<4}"
            f"{('✅' if r['judgement_consistent'] else '❌'):<4}"
            f"{str(r['citation_pages'])[:10]:<12}{r['first_token_ms']:<12}{r['total_ms']}"
        )
    print(f"\n基线准确率 = {accuracy} ({len(correct_ids)}/{len(rag_rows)})；记录值 = {s['accuracy']}")
    print(f"逐题判定全部复现: {'✅' if not inconsistent else '❌ ' + str(inconsistent)}")
    print(f"英文 5 题: accuracy={summaries['rag_en']['accuracy']} 首字均值={summaries['rag_en']['first_token_avg_ms']}ms")
    print(f"RAGAS: {output['verification']['ragas']}")
    print(f"\n输出: {OUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[FATAL] 基线结果固化失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
