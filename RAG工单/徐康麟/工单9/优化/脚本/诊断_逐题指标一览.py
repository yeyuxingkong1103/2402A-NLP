# -*- coding: utf-8 -*-
"""临时诊断：逐题「精度/召回/上下文块数/命中通道」一览（工单编号：人工智能NLP-RAG-Graph RAG 优化任务）。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
D = Path(r"E:\gao6gongdan\工单9\优化\评估结果")

rows = {}
for tag, name in (("before", "优化前_逐题_检索.jsonl"), ("after", "优化后_逐题_检索.jsonl")):
    rows[tag] = {}
    for line in (D / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[tag][r["id"]] = r

print(f"{'题':<5}{'题型':<8}{'前精度':>8}{'后精度':>8}{'前召回':>8}{'后召回':>8}"
      f"{'后块数':>7}{'后缺失数':>8}")
for qid in rows["after"]:
    a = rows["after"][qid]
    b = rows["before"].get(qid, {})
    am, bm = a.get("metrics") or {}, b.get("metrics") or {}
    miss = len(am.get("context_keyword_miss") or [])
    print(f"{qid:<5}{str(a.get('question_type')):<8}{bm.get('context_precision',0):>8.3f}"
          f"{am.get('context_precision',0):>8.3f}{bm.get('context_recall',0):>8.3f}"
          f"{am.get('context_recall',0):>8.3f}{am.get('context_size',0):>7}{miss:>8}")

# 通道统计
print("\n通道分布（after，证据块）：")
agg = {}
for qid, a in rows["after"].items():
    for c in (a.get("evidence_chunks") or []):
        k = c.get("retrieval_channel")
        agg[k] = agg.get(k, 0) + 1
print("   ", agg)
print("\n各题证据块通道：")
for qid, a in rows["after"].items():
    ch = {}
    for c in (a.get("evidence_chunks") or []):
        k = c.get("retrieval_channel")
        ch[k] = ch.get(k, 0) + 1
    print(f"    {qid}: {ch}")
