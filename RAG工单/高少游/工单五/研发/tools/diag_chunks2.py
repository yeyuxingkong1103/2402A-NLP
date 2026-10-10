# -*- coding: utf-8 -*-
"""诊断：定位军用领域收入相关块（工单编号: 人工智能 NLP-RAG-Query 理解优化任务）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge_base import load_kb   # noqa: E402

kb = load_kb(with_bm25=False)

for t in ["军用领域的收入占比", "4,627.15", "按行业列示"]:
    hits = [c for c in kb.chunks if t in c.text]
    print("=" * 90)
    print(f"[{t}] 命中 {len(hits)} 块")
    for c in hits[:6]:
        print(f"  id={c.id} {c.source} p.{c.page} {c.kind} len={len(c.text)}")
        print("     " + c.text[:400].replace("\n", "⏎"))