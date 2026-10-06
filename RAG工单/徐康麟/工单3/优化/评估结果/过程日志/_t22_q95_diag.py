# -*- coding: utf-8 -*-
"""t22 题 95 单项诊断：确认 t22k 13/14 是否源自判分器对空格（`1.0 版` vs `1.0版`）的敏感。

只读产物：读取 `优化/评估结果/accuracy_report.t22k.json` 与 golden_qa.jsonl，
生成三份候选对（原答案 / 去空格 / 短标题）交给工单1 判分器子进程。
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"E:\gao6gongdan\工单3")
GOLDEN = ROOT / "研发" / "data" / "eval" / "golden_qa.jsonl"
REPORT = ROOT / "优化" / "评估结果" / "accuracy_report.t22k.json"
PAIRS = ROOT / "优化" / "评估结果" / "过程日志" / "_t22_q95_pairs.json"
OUT = ROOT / "优化" / "评估结果" / "过程日志" / "_t22_q95_verdicts.json"
RUNNER = ROOT / "优化" / "脚本" / "ref_judge_runner.py"
REF_DEV = r"E:\gao6gongdan\工单1\研发"
PY = r"E:\gao6gongdan\工单1\.venv\Scripts\python.exe"

golden = None
for line in GOLDEN.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if int(row.get("id") or 0) == 95:
        golden = str(row["answer"])
        break
assert golden, "golden 里没有 id=95"

report = json.loads(REPORT.read_text(encoding="utf-8"))
items = report.get("results") or report.get("items") or report.get("questions") or []
row95 = next((r for r in items if int(r.get("id") or 0) == 95), None)
assert row95, "t22k 报告里没有 id=95"
answer = str(row95.get("answer_body") or row95.get("answer") or "")
print("t22k 答案 =", answer)
print("golden   =", golden[:80], "…")

variants = [
    ("t22k-原答案", answer),
    ("t22k-去空格（1.0版）", answer.replace("1.0 版", "1.0版")),
    ("t22j-短标题", "《某视频指挥系统技术规范（1.0版）》"),
    ("golden-自判", golden),
]
PAIRS.write_text(json.dumps([{"id": name, "answer": text, "golden": golden} for name, text in variants],
                            ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
proc = subprocess.run([PY, "-B", str(RUNNER), str(PAIRS), str(OUT), REF_DEV],
                      capture_output=True, text=True, encoding="utf-8", errors="replace")
print("runner exit =", proc.returncode)
for line in (proc.stdout or "").splitlines():
    print("  ", line[:300])
if proc.stderr:
    print("stderr:", proc.stderr[-500:])
verdicts = json.loads(OUT.read_text(encoding="utf-8"))
for r in verdicts["rows"]:
    print(f"  {r['id']}: ok={r['ok']} reason={r.get('reason')}")
