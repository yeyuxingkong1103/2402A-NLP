# -*- coding: utf-8 -*-
"""证据脚本：日志配对可靠性边界——①ask 嵌套深度的真实来源（并发交错）；②pages=[300] 记录的真实问题。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 脚本（证据脚本 · 日志配对边界审计；**只读**，不改交付物）
用法：``pwsh -NoProfile -File run_py.ps1 优化/脚本/audit_trace_pairing_limits.py``

利用 `QueryUnderstanding.analyze exit` 的结果自带 `original`（本轮问题的原文）——
这是与 `generate_*` 同源的强关联记录，可用于校验「问题↔答案」的归属。
"""

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True

W1 = Path(r"E:\gao6gongdan\工单1")
TRACE = W1 / "logs" / "rag_trace.jsonl"

golden = {
    json.loads(l)["question"]: int(json.loads(l)["id"])
    for l in (W1 / "data" / "eval" / "golden_qa.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
}

# ---------- ① ask 嵌套深度 ----------
stack: list[str] = []
max_depth = 0
max_at = None
depth_hist = {}
events = 0
for i, line in enumerate(TRACE.open(encoding="utf-8", errors="replace")):
    if "QAEngine.ask" not in line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    if obj.get("function") != "QAEngine.ask":
        continue
    events += 1
    if obj.get("event") == "enter":
        args = obj.get("args") or []
        stack.append(args[0] if args else "")
        depth_hist[len(stack)] = depth_hist.get(len(stack), 0) + 1
        if len(stack) > max_depth:
            max_depth = len(stack)
            max_at = (i, list(stack)[:6])
    elif obj.get("event") == "exit" and stack:
        stack.pop()

print("=== ① QAEngine.ask 嵌套深度（仅计 ask 事件）===")
print(f"  事件数={events} 最大深度={max_depth}")
print(f"  深度分布（深度: 出现次数）= {dict(sorted(depth_hist.items()))}")
if max_at:
    print(f"  最大深度出现在第 {max_at[0]} 行；当时栈顶若干问题:")
    for q in max_at[1]:
        print(f"      - {str(q)[:52]}")
print(f"  文件结束时残留栈深={len(stack)}（= 未闭合 enter 数）")

# ---------- ② pages=[300] 记录的真实问题 ----------
print("\n=== ② 抽取式 pages=[300] 记录 与 最近的 analyze.original 对照 ===")
last_analyze_original = None
current_q = None
shown = 0
for i, line in enumerate(TRACE.open(encoding="utf-8", errors="replace")):
    if "QueryUnderstanding.analyze" not in line and "Generator.generate_extractive" not in line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    fn, ev = obj.get("function"), obj.get("event")
    if fn == "QueryUnderstanding.analyze" and ev == "exit":
        res = obj.get("result") or {}
        if isinstance(res, dict):
            last_analyze_original = res.get("original")
        continue
    if fn == "Generator.generate_extractive":
        if ev == "enter":
            args = obj.get("args") or []
            current_q = args[0] if args and isinstance(args[0], str) else None
            continue
        res = obj.get("result") or {}
        if not isinstance(res, dict):
            continue
        pages = [p for p in (res.get("pages") or []) if isinstance(p, int)]
        if pages == [300] or (pages and max(pages) == 300 and len(pages) == 1):
            print(f"  [行 {i}] 生成器收到的 question = {str(current_q)[:46]}")
            print(f"          归属题号 = {golden.get(current_q, '非工单问题')}")
            print(f"          analyze.original（本轮的权威问题） = {str(last_analyze_original)[:46]}")
            print(f"          answer = {str(res.get('answer'))[:56]}")
            print(f"          intent/关键词见日志；primary={res.get('primary_chunk_id')}")
            shown += 1
            if shown >= 4:
                break
if not shown:
    print("  （未找到 pages=[300] 的抽取式记录）")
