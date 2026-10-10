# -*- coding: utf-8 -*-
"""诊断：Q1 候选池与重排分量（工单编号: 人工智能 NLP-RAG-Query 理解优化任务）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config   # noqa: E402
from src.knowledge_base import load_kb   # noqa: E402
from src.query_understanding import QueryUnderstanding   # noqa: E402
from src.retriever import Retriever   # noqa: E402

Q = "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"
kb = load_kb()
rt = Retriever(kb)
qu = QueryUnderstanding()
a = qu.analyze(Q)
print("core:", a.core, "| kw:", a.keywords, "| syn:", a.synonyms, "| type:", a.answer_type)

cands = rt.search(a.core, top_k=config.ANSWER_POOL, extra_queries=a.expand_queries)
print(f"融合候选 {len(cands)} 个")
targets = {c.id for c in kb.chunks if "国防客户的销售额合计分别为" in c.text}
print("目标块 id:", targets)
for rank, c in enumerate(cands[:30]):
    ch = kb.chunks[c.chunk_id]
    mark = "  <<< 目标" if c.chunk_id in targets else ""
    print(f"  #{rank} id={c.chunk_id} rrf={c.rrf:.4f} vec={c.vector_score:.3f} "
          f"bm25={c.bm25_score:.3f} {ch.source} p.{ch.page} {ch.kind}{mark}")
    if c.chunk_id in targets:
        print("     ", ch.text[:200].replace("\n", "⏎"))