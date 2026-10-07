"""
检索排序探针（离线）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

用途：**在调任何权重之前**，先把两路召回的原始分并排打出来，用数据决定参数。

必看三项：
  1) 余弦分的**极差** —— 极差 < 0.1 说明这个 embedding 在你的语料上区分不出语义远近；
  2) BM25 的**命中来源** —— top20 里有多少条真的出自正确章节；
  3) 两路的**交集** —— 双命中的块通常就是最该排第一的。

用法：
    python scripts/probe_retrieval_rank.py "武汉兴图新科电子股份有限公司注册资本是多少"
    python scripts/probe_retrieval_rank.py --calibrate   # 跑「一相关 + 一离题」校准阈值
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src import config  # noqa: E402
from src.embedder import cosine_scores, embed_query  # noqa: E402
from src.index_store import KnowledgeBase  # noqa: E402
from src.query_norm import normalize_query  # noqa: E402
from src.retriever import retrieve  # noqa: E402


def probe(query: str, top_k: int = 10) -> None:
    kb = KnowledgeBase.get()
    q = normalize_query(query)
    print(f"\n=== 探针：{query}")
    print(f"    归一化后：{q}")

    dense = cosine_scores(kb.embeddings, embed_query(q))
    sparse = kb.bm25_scores(q)

    order = np.argsort(-dense)[:top_k]
    cos_vals = [float(dense[i]) for i in order]
    print(f"\n  [稠密] 余弦 top{top_k}：极差 {max(cos_vals)-min(cos_vals):.4f}")
    for rank, i in enumerate(order, 1):
        c = kb.chunks[i]
        print(f"    {rank:>2}. cos={dense[i]:.4f} bm25={sparse[i]:7.2f} p{c.get('page')} "
              f"{(c.get('section') or '')[:36]}")

    sorder = np.argsort(-sparse)[:top_k]
    bm_vals = [float(sparse[i]) for i in sorder]
    print(f"\n  [稀疏] BM25 top{top_k}：命中 {int((sparse>0).sum())} 条，最高 {max(bm_vals):.2f}")
    for rank, i in enumerate(sorder, 1):
        c = kb.chunks[i]
        print(f"    {rank:>2}. bm25={sparse[i]:7.2f} cos={dense[i]:.4f} p{c.get('page')} "
              f"{(c.get('section') or '')[:36]}")

    inter = set(order.tolist()) & set(sorder.tolist())
    print(f"\n  [交集] top{top_k} 双路命中 {len(inter)} 条")
    for i in sorted(inter, key=lambda x: -dense[x]):
        c = kb.chunks[i]
        print(f"    p{c.get('page')} cos={dense[i]:.4f} bm25={sparse[i]:.2f} {(c.get('section') or '')[:40]}")

    res = retrieve(query)
    print(f"\n  [最终] 闸门拦空={res.gated}，保留 {len(res.items)} 条，耗时 {res.trace.get('total_ms')} ms")
    for it in res.items:
        print(f"    score={it.score:.4f} ev={it.evidence:.4f} cos={it.cosine:.4f} "
              f"bm25n={it.bm25_norm:.3f} p{it.page} {it.section[:36]}")


def calibrate() -> None:
    """
    阈值校准：用「一个真相关问题 + 一个明确离题问题」量出依据分区间。
    两边不重叠 → 阈值取中间；有重叠 → 说明这个信号分不开它们，不要硬凑数字。
    """
    cases = [
        ("真相关-1", "武汉兴图新科电子股份有限公司的注册资本是多少？"),
        ("真相关-2", "报告期内军用领域收入是多少？"),
        ("真相关-3", "电子信息行业的下游主要包括哪些行业？"),
        ("离题-1", "如何用 Python 实现快速排序？"),
        ("离题-2", "今天北京的天气怎么样？"),
        ("离题-3", "红烧肉怎么做才好吃？"),
    ]
    print("=== 阈值校准（依据分 = 余弦 + β·BM25归一，不含主题先验）")
    print(f"    β={config.RETRIEVAL_BM25_BETA}  δ={config.RETRIEVAL_DOC_CONSENSUS_DELTA}"
          f"  当前阈值={config.RETRIEVAL_MIN_EVIDENCE}")
    for name, q in cases:
        res = retrieve(q, top_k=6, apply_gate=False)
        evs = [it.evidence for it in res.items]
        if evs:
            print(f"  {name:<8} 依据分 {min(evs):.4f} ~ {max(evs):.4f}  "
                  f"(top1 score={res.items[0].score:.4f})  {q[:30]}")
        else:
            print(f"  {name:<8} 候选池为空  {q[:30]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="武汉兴图新科电子股份有限公司的注册资本是多少")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    if args.calibrate:
        calibrate()
    else:
        probe(args.query, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
