# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-混合检索任务
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务
# 模块：retrieval —— 多策略检索编排（向量 / 全文 / 混合）+ 重排
# 说明：为工单 06《混合检索任务》提供统一入口，三种策略可配置：
#   · vector   纯向量检索（可换嵌入模型）
#   · fulltext 纯全文检索（倒排索引，见 fts.py）
#   · hybrid   混合检索（向量 + 全文，权重可调）
#   融合算法：weighted（加权平均）/ voting（投票）/ rrf（加权 RRF）
#   重排：none / llm / tfidf / feedback（见 rerank.py），对候选池重排后取 top_k

from collections import Counter

from config import (RECALL_K, TOP_K, DEFAULT_EMBED, RETRIEVAL_STRATEGY, RERANKER,
                    HYBRID_WEIGHT_VEC, HYBRID_WEIGHT_FT, FUSION_ALGO, RRF_K, RRF_LAMBDA)
import kb as kb_mod
import fts as fts_mod
import rerank as rerank_mod

STRATEGIES = ("vector", "fulltext", "hybrid", "rrf")
FUSIONS = ("weighted", "voting", "rrf")

_FTS = {"idx": None}


def get_fts():
    """懒加载全文索引（挂在 KB 的 chunk 上，与向量索引同源）。"""
    if _FTS["idx"] is None:
        km = kb_mod.get_kb()
        _FTS["idx"] = fts_mod.build_index(km.chunks)
    return _FTS["idx"]


def reset_fts():
    _FTS["idx"] = None


# ---------------- 单路检索 ----------------
def vector_search(query, k=RECALL_K, embed_model=None):
    """向量检索。可指定嵌入模型（embed_model）；换模型需用该模型重建索引。"""
    km = kb_mod.get_kb()
    return km.search_dense(query, k=k, model=embed_model)


def fulltext_search(query, k=RECALL_K, mode="auto"):
    """全文检索。mode=auto：先按布尔 AND 精确匹配，命中过少则退回 OR 宽松匹配
    （自然语言问题用 AND 会过严，容易一条都召不回）。"""
    idx = get_fts()
    if mode != "auto":
        return idx.search(query, k=k, mode=mode)
    tight = idx.search(query, k=k, mode="boolean")
    return tight if len(tight) >= max(3, k // 3) else idx.search(query, k=k, mode="any")


def bm25_search(query, k=RECALL_K):
    km = kb_mod.get_kb()
    return km.search_bm25(query, k=k)


# ---------------- 融合 ----------------
def _norm(pairs):
    if not pairs:
        return {}
    mx = max(s for _i, s in pairs) or 1.0
    return {i: (s / mx) for i, s in pairs}


def fuse(vec, ft, algo=FUSION_ALGO, wv=HYBRID_WEIGHT_VEC, wf=HYBRID_WEIGHT_FT, k=RRF_K):
    """把两路召回融合成 [(idx, score)]。algo: weighted / voting / rrf。"""
    if algo == "rrf":
        fused = {}
        for r, (i, _s) in enumerate(vec):
            fused[i] = fused.get(i, 0.0) + wv / (k + r + 1)
        for r, (i, _s) in enumerate(ft):
            fused[i] = fused.get(i, 0.0) + wf / (k + r + 1)
        return sorted(fused.items(), key=lambda x: -x[1])
    if algo == "voting":
        votes, base = Counter(), {}
        for i, s in vec:
            votes[i] += 1
            base[i] = max(base.get(i, 0.0), s)
        for i, s in ft:
            votes[i] += 1
            base[i] = max(base.get(i, 0.0), s)
        return sorted(votes.items(), key=lambda x: (-x[1], -base.get(x[0], 0.0)))
    dv, fs = _norm(vec), _norm(ft)
    keys = set(dv) | set(fs)
    return sorted(((i, wv * dv.get(i, 0.0) + wf * fs.get(i, 0.0)) for i in keys),
                  key=lambda x: -x[1])


# ---------------- 统一入口 ----------------
def search(query, strategy=None, reranker=None, top_k=TOP_K, recall_k=30,
           fusion=None, weights=None, doc=None, rerank_pool=20, dedupe=True, caliber=True,
           expand=True):
    """多策略检索。返回 hits（chunk 字典 + score 字段）；doc 可限定文档。
    expand=True：先过一遍 Query 理解（术语改写 / 跨语扩展），与主链路口径一致；
    caliber=True：应用金额/占比口径消歧。"""
    km = kb_mod.get_kb()
    strategy = (strategy or RETRIEVAL_STRATEGY).lower()
    reranker = (reranker if reranker is not None else RERANKER) or "none"
    wv, wf = weights or (HYBRID_WEIGHT_VEC, HYBRID_WEIGHT_FT)
    algo = fusion or FUSION_ALGO

    rq = query
    if expand or (doc is None) or caliber:
        import engine as _eng                     # retrieval 不参与 engine 导入，无循环依赖
        if doc is None and expand:
            doc = _eng.detect_doc(query)
        if expand:
            rq = _eng.query_understanding(query)["expanded"]

    if strategy == "vector":
        fused = vector_search(rq, recall_k)
    elif strategy == "fulltext":
        fused = fulltext_search(rq, recall_k)
    elif strategy == "rrf":
        # 与系统既有链路一致：向量 + BM25 → 加权 RRF（λ=2.5 / 1.0）
        lam = weights[0] if weights else RRF_LAMBDA
        fused = fuse(km.search_dense(rq, k=recall_k), km.search_bm25(rq, k=recall_k),
                     algo="rrf", wv=lam, wf=1.0)
    else:  # hybrid
        fused = fuse(vector_search(rq, recall_k), fulltext_search(rq, recall_k),
                     algo=algo, wv=wv, wf=wf)

    dense_sim = {i: s for i, s in km.search_dense(rq, k=recall_k)} if strategy != "vector" else dict(fused)

    hits, seen = [], set()
    for idx, sc in fused:
        c = km.get(idx)
        if doc and c.get("doc") and c.get("doc") != doc:
            continue
        key = (c["text"][:40], c.get("section"))
        if dedupe and key in seen:
            continue
        seen.add(key)
        h = dict(c)
        h["score"] = sc
        h["rrf"] = sc
        h["dense_sim"] = dense_sim.get(idx, 0.0)
        h["strategy"] = strategy
        hits.append(h)
        if len(hits) >= max(recall_k, rerank_pool):
            break

    # 口径消歧（与 engine 共用同一套规则）：问金额压“只看百分比”的块、抬含“万元”的块
    if caliber:
        import engine as _eng
        need_amt, need_ratio = _eng._hit_caliber(query, "full")
        if need_amt or need_ratio:
            for h in hits:
                t, sc = h["text"], h["score"]
                if need_amt:
                    if _eng._is_ratio_only(t):
                        sc *= _eng.AMT_PENALTY
                    elif _eng._AMT_NUM.search(t):
                        sc *= _eng.AMT_BOOST
                elif need_ratio and not _eng._is_ratio_only(t) and "%" not in t:
                    sc *= _eng.AMT_PENALTY
                h["score"] = sc
                h["rrf"] = sc
            hits.sort(key=lambda x: -x["score"])

    if reranker and reranker != "none":
        hits = rerank_mod.get_reranker(reranker).rerank(query, hits[:rerank_pool], top_k=top_k)
    else:
        hits = hits[:top_k]
    return hits


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    q = "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"
    for st in STRATEGIES:
        for rk in ("none", "tfidf"):
            hits = search(q, strategy=st, reranker=rk, top_k=5)
            print("%-9s %-7s -> %s" % (st, rk, [(h["page"], round(h.get("score") or h.get("rerank_score") or 0, 3)) for h in hits]))
