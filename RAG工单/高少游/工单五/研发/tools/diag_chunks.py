# -*- coding: utf-8 -*-
"""诊断：定位关键事实所在知识块（工单编号: 人工智能 NLP-RAG-Query 理解优化任务）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge_base import load_kb   # noqa: E402

kb = load_kb(with_bm25=False)
print(f"总块数: {len(kb.chunks)}")

TARGETS = ["6,464.51", "18,780.67", "军用领域", "销售处", "大客户销售部", "组织结构图"]
for t in TARGETS:
    hits = [c for c in kb.chunks if t in c.text]
    print("=" * 90)
    print(f"[{t}] 命中 {len(hits)} 块")
    for c in hits[:4]:
        print(f"  id={c.id} {c.source} p.{c.page} {c.kind}: {c.text[:220]}")