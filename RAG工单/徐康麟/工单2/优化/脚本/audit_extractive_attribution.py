# -*- coding: utf-8 -*-
"""证据脚本：抽取式记录「归属是否正确」用其自带 pages 与各题真实上下文比对判定（t6/t7 归因依据）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 脚本（证据脚本 · t6/t7 归因审计；**只读**，不改交付物）
用法：``pwsh -NoProfile -File run_py.ps1 优化/脚本/audit_extractive_attribution.py``

每条 extractive 记录自带 (answer, pages, primary_chunk_id)，三者同源、自洽。
把 pages 与 10 道题各自**真实**的上下文页码（baseline_repro_retrieval.json，来自真实日志复现）比对：

  - pages 与**归属题**的上下文一致        -> 归属正确 → 用 check_answer 判分
  - pages 与**另一道题**的上下文一致      -> 日志交错导致的归错回合（不是答错）
  - pages 与任何一题都不一致             -> 属非工单问题回合（多轮/界面测试），排除
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True

W1 = Path(r"E:\gao6gongdan\工单1")
BASE = Path(r"E:\gao6gongdan\工单2\优化\基线")
REPRO = BASE / "repro"

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

from app.core.evaluator import Evaluator  # noqa: E402

repro = json.loads((BASE / "baseline_repro_retrieval.json").read_text(encoding="utf-8"))
ctx = {r["question_id"]: r["context_pages"] for r in repro["per_question"]}
golden = {
    json.loads(l)["question"]: json.loads(l)
    for l in (W1 / "data" / "eval" / "golden_qa.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
}
evaluator = Evaluator()


def leading_pages(pages):
    if not isinstance(pages, list):
        return []
    return [p for p in pages if isinstance(p, int)]


overall = Counter()
per_file = {}
for trace in (W1 / "logs" / "rag_trace.jsonl", W1 / "logs" / "rag_trace.jsonl.1"):
    current_q = None
    c = Counter()
    details = []
    for line in trace.open(encoding="utf-8", errors="replace"):
        if "Generator.generate_extractive" not in line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("function") != "Generator.generate_extractive":
            continue
        if obj.get("event") == "enter":
            args = obj.get("args") or []
            current_q = args[0] if args and isinstance(args[0], str) else None
            continue
        if obj.get("event") != "exit" or current_q not in golden:
            continue
        res = obj.get("result") or {}
        if not isinstance(res, dict):
            continue
        qid = int(golden[current_q]["id"])
        pages = leading_pages(res.get("pages"))
        c["total"] += 1

        # 归属判定
        if pages and pages == ctx.get(qid, [])[: len(pages)]:
            label = "attributed_ok"
        else:
            others = [o for o, op in ctx.items() if o != qid and pages and pages == op[: len(pages)]]
            label = "misattributed_other_question" if others else "not_a_golden_turn"
        c[label] += 1
        if label == "attributed_ok":
            ok, reason = evaluator.check_answer((res.get("answer") or "").strip(), golden[current_q]["answer"])
            c["attributed_ok_correct"] += int(ok)
            if not ok:
                details.append((qid, (res.get("answer") or "")[:50], reason))
    per_file[trace.name] = {"counters": dict(c), "wrong_when_correctly_attributed": details[:5]}
    overall.update(c)

print("=== 抽取式路径「归属正确性」判定 ===")
for name, v in per_file.items():
    c = v["counters"]
    total = c.get("total", 0)
    ok = c.get("attributed_ok", 0)
    correct = c.get("attributed_ok_correct", 0)
    print(f"\n--- {name} ---")
    print(f"  归属到工单 10 题的抽取式记录: {total}")
    print(f"    ① 归属正确（pages 与本题真实上下文一致）: {ok}")
    print(f"       其中判对: {correct}" + (f"；判错: {ok - correct}" if ok - correct else "；判错: 0"))
    print(f"    ② 归错回合（pages 与另一道题一致）: {c.get('misattributed_other_question', 0)}")
    print(f"    ③ 属非工单问题回合（pages 与任一题都不一致）: {c.get('not_a_golden_turn', 0)}")
    if v["wrong_when_correctly_attributed"]:
        print(f"    归属正确但判错的样本: {v['wrong_when_correctly_attributed']}")

t = overall
print("\n=== 两文件合计 ===")
print(f"  总计 {t['total']}；归属正确 {t['attributed_ok']}（判对 {t['attributed_ok_correct']}）；"
      f"归错回合 {t['misattributed_other_question']}；非工单回合 {t['not_a_golden_turn']}")
print(f"  结论：在归属正确的记录里，抽取式判对率 = "
      f"{t['attributed_ok_correct']/t['attributed_ok']*100:.2f}%" if t["attributed_ok"] else "  结论：无记录")
