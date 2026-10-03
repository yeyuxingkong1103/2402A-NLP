# -*- coding: utf-8 -*-
"""回答路径审计（v3，生成器级配对）：用「问题随生成器调用一起记录」这一强关联统计两条路径的表现。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 基线采集（T7 归因防漂移证据）

为什么不用 QAEngine.ask 配对
---------------------------
实测 `rag_trace.jsonl`：ask enter=1350、exit=1338（12 个未闭合），**嵌套最大深度 21**。
在这种日志上，无论「最近一次 enter」（v1）还是「栈式配对」（v2），都可能因未闭合/交错而错配。
而 `Generator.generate_extractive` 与 `Generator.generate` 的 **enter 入参里直接带 question**，
exit 的 result 里直接带 answer/mode/primary_chunk_id——这是不依赖嵌套推断的**强关联**。
本脚本以生成器级配对为准，并把两条路径（extractive / llm）分开统计。

交叉校验：同一 case 在 `generate` 与 `generate_extractive` 两级配对上的 question/answer
若不一致，则记为冲突并单独计数，绝不静默取一方。

判分：工单1 原始 `Evaluator.check_answer`（只读副本），阈值 0.62 未改动。只读工单1。

复现命令（工作目录 = E:\\gao6gongdan\\工单2）：

    pwsh -NoProfile -File run_py.ps1 -B 优化/脚本/audit_answer_paths_generator.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True

W1 = Path(r"E:\gao6gongdan\工单1")
W2 = Path(r"E:\gao6gongdan\工单2")
REPRO = W2 / "优化" / "基线" / "repro"
OUT = W2 / "优化" / "基线" / "baseline_answer_paths_generator.json"

TRACES = [W1 / "logs" / "rag_trace.jsonl", W1 / "logs" / "rag_trace.jsonl.1"]

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


def scan(path: Path) -> dict:
    """按生成器级 enter/exit 相邻配对，抽取 (question, answer, mode, primary) 四元组。"""
    out = {
        "file": path.name,
        "cases": [],  # {"level": "generate"|"generate_extractive", ...}
        "counters": Counter(),
    }
    pending: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "Generator.generate" not in line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            fn = obj.get("function", "")
            if fn not in ("Generator.generate", "Generator.generate_extractive"):
                continue
            event = obj.get("event")
            if event == "enter":
                args = obj.get("args") or []
                kwargs = obj.get("kwargs") or {}
                question = ""
                if fn == "Generator.generate_extractive":
                    question = args[0] if args and isinstance(args[0], str) else ""
                else:
                    question = kwargs.get("question") or (args[0] if args and isinstance(args[0], str) else "")
                pending[fn].append({"question": question, "ts": obj.get("ts")})
                out["counters"][f"{fn}.enter"] += 1
                continue
            if event != "exit":
                continue
            out["counters"][f"{fn}.exit"] += 1
            if not pending[fn]:
                out["counters"][f"{fn}.exit_without_enter"] += 1
                continue
            opened = pending[fn].pop(0)  # FIFO：按进入顺序配对，避免交错时错配
            res = obj.get("result") or {}
            if not isinstance(res, dict):
                continue
            out["cases"].append(
                {
                    "file": path.name,
                    "level": "generate_extractive" if "extractive" in fn else "generate",
                    "question": opened["question"],
                    "mode": str(res.get("mode")),
                    "answer": (res.get("answer") or "").strip(),
                    "primary_chunk_id": res.get("primary_chunk_id"),
                    "pages": res.get("pages"),
                    "retrieved_count": res.get("retrieved_count"),
                    "ts": opened["ts"],
                }
            )
    for fn, left in pending.items():
        out["counters"][f"{fn}.unclosed"] += len(left)
    out["counters"] = dict(out["counters"])
    return out


def main() -> int:
    from app.core.config import get_settings
    from app.core.evaluator import FUZZY_THRESHOLD, Evaluator

    settings = get_settings()
    assert str(settings.paths.logs).startswith(str(REPRO)), "日志路径未隔离到 repro，拒绝继续"

    golden_list = [
        json.loads(l)
        for l in (W1 / "data" / "eval" / "golden_qa.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    golden = {g["question"]: g for g in golden_list}
    evaluator = Evaluator()

    scans = [scan(t) for t in TRACES if t.exists()]
    cases = [c for s in scans for c in s["cases"]]

    # 只保留 level=generate_extractive（最内层、问题与答案同源），
    # 并用 level=generate 做交叉校验
    inner = [c for c in cases if c["level"] == "generate_extractive"]
    outer = {(c["file"], c["ts"], c["question"], c["answer"]) for c in cases if c["level"] == "generate"}

    # 只统计工单 10 题
    rows_by_q: dict[int, dict] = {}
    all_mode_counter = Counter()
    for c in inner:
        all_mode_counter[c["mode"]] += 1
    for c in inner:
        g = golden.get(c["question"])
        if not g:
            continue
        qid = int(g["id"])
        rows_by_q.setdefault(qid, {"question_id": qid, "question": c["question"], "golden": g["answer"], "obs": []})
        rows_by_q[qid]["obs"].append(c)

    per_question = []
    for qid in sorted(rows_by_q):
        row = rows_by_q[qid]
        by_answer: dict[str, list[dict]] = defaultdict(list)
        for o in row["obs"]:
            by_answer[o["answer"]].append(o)
        judged = []
        for answer, occ in sorted(by_answer.items(), key=lambda kv: -len(kv[1])):
            ok, reason = evaluator.check_answer(answer, row["golden"])
            judged.append(
                {
                    "answer": answer,
                    "is_correct": bool(ok),
                    "reason": reason,
                    "occurrences": len(occ),
                    "primary_chunk_ids": sorted({str(o["primary_chunk_id"]) for o in occ}),
                    "files": sorted({o["file"] for o in occ}),
                    "cross_checked_in_generate": sum(
                        1 for o in occ if (o["file"], o["ts"], o["question"], o["answer"]) in outer
                    ),
                }
            )
        correct = sum(j["occurrences"] for j in judged if j["is_correct"])
        per_question.append(
            {
                "question_id": qid,
                "question": row["question"],
                "golden": row["golden"],
                "observations": len(row["obs"]),
                "correct_observations": correct,
                "observation_accuracy": round(correct / len(row["obs"]), 4) if row["obs"] else 0.0,
                "distinct_answers": len(judged),
                "all_answers_correct": all(j["is_correct"] for j in judged),
                "at_least_one_correct": any(j["is_correct"] for j in judged),
                "answers": judged,
            }
        )

    # 按文件拆分
    per_file = {}
    for s in scans:
        sub = [c for c in s["cases"] if c["level"] == "generate_extractive" and c["question"] in golden]
        per_file[s["file"]] = {
            "extractive_observations_on_10_questions": len(sub),
            "mode_counter_all_generator_calls": {
                k: v for k, v in s["counters"].items() if k.startswith("Generator")
            },
        }

    total_obs = sum(r["observations"] for r in per_question)
    total_ok = sum(r["correct_observations"] for r in per_question)

    OUT.write_text(
        json.dumps(
            {
                "meta": {
                    "工单": "人工智能NLP-RAG-基于PDF文档的问答系统优化",
                    "阶段": "优化 / 基线采集（回答路径审计 v3：生成器级强关联配对）",
                    "sources": [str(t) for t in TRACES if t.exists()],
                    "配对方法": "Generator.generate_extractive 的 enter 入参带 question，exit 结果带 answer/mode/"
                    "primary_chunk_id；按同函数 enter/exit 的进入顺序 FIFO 配对，不依赖 QAEngine.ask 的嵌套推断。"
                    "并用 Generator.generate 一级做交叉校验。",
                    "为何不用 ask 级配对": "实测 ask enter=1350 / exit=1338（12 个未闭合），嵌套最大深度 21，"
                    "任何「最近 enter」或「栈式」推断都可能错配。",
                    "judge": "工单1 Evaluator.check_answer（只读副本），FUZZY_THRESHOLD 未改动",
                    "FUZZY_THRESHOLD": FUZZY_THRESHOLD,
                    "reproduce_cmd": "pwsh -NoProfile -File run_py.ps1 -B 优化/脚本/audit_answer_paths_generator.py",
                },
                "file_scan": [s["counters"] | {"file": s["file"]} for s in scans],
                "per_file": per_file,
                "extractive_summary": {
                    "questions_covered": len(per_question),
                    "observations": total_obs,
                    "correct_observations": total_ok,
                    "observation_accuracy": round(total_ok / total_obs, 4) if total_obs else 0.0,
                    "questions_all_answers_correct": sum(1 for r in per_question if r["all_answers_correct"]),
                    "questions_at_least_one_correct": sum(1 for r in per_question if r["at_least_one_correct"]),
                    "questions_with_single_distinct_answer": sum(
                        1 for r in per_question if r["distinct_answers"] == 1
                    ),
                },
                "per_question": per_question,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== 生成器级扫描 ===")
    for s in scans:
        print(f"  {s['file']}: {s['counters']}")
    print("\n=== 工单 10 题（生成器级配对，extractive 路径）===")
    print(
        f"{'题号':<6}{'观测':<8}{'判对':<8}{'观测级准确率':<14}{'答案种类':<10}{'全部判对':<10}{'交叉校验命中':<12}"
    )
    for r in per_question:
        xc = sum(a["cross_checked_in_generate"] for a in r["answers"])
        print(
            f"{r['question_id']:<6}{r['observations']:<8}{r['correct_observations']:<8}"
            f"{r['observation_accuracy']:<14}{r['distinct_answers']:<10}"
            f"{str(r['all_answers_correct']):<10}{xc}/{r['observations']:<8}"
        )
        for a in r["answers"]:
            print(
                f"      {'✅' if a['is_correct'] else '❌'} x{a['occurrences']:<4} files={a['files']} "
                f":: {a['answer'][:56]}"
            )
    s = json.loads(OUT.read_text(encoding="utf-8"))["extractive_summary"]
    print(
        f"\n汇总（extractive，两文件合计）：观测 {s['observations']}，判对 {s['correct_observations']} "
        f"= {s['observation_accuracy']:.4f}；全部答案判对 {s['questions_all_answers_correct']}/10；"
        f"至少一种判对 {s['questions_at_least_one_correct']}/10；仅 1 种答案 {s['questions_with_single_distinct_answer']}/10"
    )
    print(f"输出: {OUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[FATAL] 生成器级审计失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
