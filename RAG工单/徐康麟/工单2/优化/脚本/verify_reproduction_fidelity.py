# -*- coding: utf-8 -*-
"""证据脚本：核对基线检索复现结果与真实日志页码是否逐题一致（t6 保真度校验），并输出 Q95/Q207 的替代判据结论。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 脚本（证据脚本 · t6 复现保真度校验；**只读**，不改交付物）
用法：``pwsh -NoProfile -File run_py.ps1 优化/脚本/verify_reproduction_fidelity.py``
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"E:\gao6gongdan\工单2\优化\基线")
repro = json.loads((BASE / "baseline_repro_retrieval.json").read_text(encoding="utf-8"))
logged = json.loads((BASE / "baseline_evalrun_context.json").read_text(encoding="utf-8"))
traps = json.loads((BASE / "baseline_retrieval.json").read_text(encoding="utf-8"))["traps"]

logged_pages = {r["question_id"]: r["上下文页(前5+计数)"] for r in logged["per_question"]}

print("=== 复现 vs 真实日志：上下文页一致性 ===")
for r in repro["per_question"]:
    qid = r["question_id"]
    lg = [p for p in (logged_pages.get(qid) or []) if isinstance(p, int)]
    mine = r["context_pages"]
    ok = lg == mine[: len(lg)]
    print(f"Q{qid:<5} 日志前{len(lg)}= {lg}")
    print(f"      复现     = {mine}   前缀一致: {'✅' if ok else '❌'}")

print("\n=== 关键题目的上下文 chunk ===")
by_id = {r["question_id"]: r for r in repro["per_question"]}
for qid in (95, 207, 957, 795, 34, 793):
    r = by_id[qid]
    print(f"Q{qid}: ctx={r['context_chunk_ids']}")
    print(f"       证据chunk(严格)={r['evidence_chunk_ids']} 在上下文={r['evidence_chunk_in_context']}")

print("\n=== Q95 替代判据：c000740(page160) 是否在上下文 ===")
print("c000740 in Q95 ctx:", "c000740" in by_id[95]["context_chunk_ids"])
print("Q95 上下文页:", by_id[95]["context_pages"])

print("\n=== Q207 替代判据：页 479/490 是否在上下文 ===")
print("Q207 上下文页:", by_id[207]["context_pages"], "含479/490:",
      bool({479, 490} & set(by_id[207]["context_pages"])))
print("Q207 上下文包含补充流动资金块的 chunk:",
      [c for c in by_id[207]["context_chunk_ids"] if c in {"c002666", "c002667", "c002668", "c002669", "c002715"}])

print("\n=== 证据在上下文中的排名（1=最前）===")
for r in repro["per_question"]:
    ranks = [s["rank"] for s in r["scores"] if s["chunk_id"] in r["evidence_chunk_ids"]]
    print(f"Q{r['question_id']:<5} 证据chunk位置={ranks or '—'}  top1={r['top1']}")

print("\n=== 陷阱命中块 ===")
print("Q531 命中块:", traps["Q531_无区分度"]["命中块(B_punct)"])
print("Q207 替代判据块:", traps["Q207_合成引用文本"]["替代判据_含补充流动资金且含15000的块"])
print("Q95 前缀块:", traps["Q95_省略号"]["前缀块(B_punct)"])
