# -*- coding: utf-8 -*-
"""临时诊断：逐题看「哪些参考答案关键词没被上下文覆盖」（工单编号：人工智能NLP-RAG-Graph RAG 优化任务）。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
P = Path(r"E:\gao6gongdan\工单9\优化\评估结果\优化后_逐题_检索.jsonl")
refs = {}
for line in Path(r"E:\gao6gongdan\工单9\研发\评估\参考答案集.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        d = json.loads(line)
        refs[d["id"]] = d

for line in P.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    m = row.get("metrics") or {}
    miss = m.get("context_keyword_miss") or []
    if not miss:
        continue
    print(f"{row['id']} [{row.get('question_type')}] 精度={m.get('context_precision')} "
          f"召回={m.get('context_recall')} 上下文={m.get('context_size')}块")
    print(f"    缺失关键词：{miss}")
    print(f"    证据块：{[(c.get('chunk_id'), c.get('page')) for c in (row.get('evidence_chunks') or [])]}")
