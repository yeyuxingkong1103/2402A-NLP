# -*- coding: utf-8 -*-
"""检索诊断：定位 Q1 / Q5 的召回与耗时瓶颈。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务
"""
import sys
import time

sys.path.insert(0, ".")

from src import config
from src.conversation import Conversation
from src.embedding import Embedder
from src.knowledge_base import load_kb
from src.query_rewriting import QueryRewriter
from src.query_understanding import QueryUnderstanding
from src.reranker import rerank
from src.retriever import Retriever

kb = load_kb()
qu = QueryUnderstanding()
rt = Retriever(kb)

QS = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
]

for q in QS:
    print("=" * 90)
    print("Q:", q)
    a = qu.analyze(q)
    print("core:", a.core, "| type:", a.answer_type, "| doc:", a.doc_hint)
    print("kw:", a.keywords)
    print("expand:", a.retrieval_queries)

    t0 = time.time(); cands = rt.search(a.core, top_k=config.ANSWER_POOL,
                                        extra_queries=a.expand_queries)
    t1 = time.time()
    ranked = rerank(kb, cands, a, top_k=8)
    t2 = time.time()
    print(f"[retrieval {t1-t0:.2f}s | rerank {t2-t1:.2f}s | candidates {len(cands)}]")
    for r in ranked:
        c = kb.chunks[r.chunk_id]
        print(f"  {r.score:.3f} | {c.source} p.{c.page} {c.kind} | {c.text[:110]}")

    # 直接看目标句是否在库里
    target = "6,464.51" if "军用" in q else "大客户采取设立分公司"
    hits = [c for c in kb.chunks if target in c.text]
    print(f"  [目标句 '{target}' 命中块数: {len(hits)}]")
    for c in hits[:3]:
        print(f"     id={c.id} {c.source} p.{c.page} {c.kind}: {c.text[:150]}")

# 计时分解
print("=" * 90)
em = Embedder()
t0 = time.time(); em.encode_one("军用领域收入"); t1 = time.time()
print(f"单次向量编码: {t1-t0:.2f}s")
t0 = time.time(); em.encode_one("军用领域收入"); t1 = time.time()
print(f"缓存后向量编码: {t1-t0:.3f}s")