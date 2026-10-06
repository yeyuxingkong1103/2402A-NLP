# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
混合检索器（优化核心）：
  1. 向量召回想：m3e 语义相似度 Top-N；
  2. BM25 召回：倒排索引关键词 Top-N；
  3. RRF/加权融合：统一两种召回的排序；
  4. MMR 重排：在相关性与多样性间权衡，避免结果冗余。
提供 search()（优化版）与 vector_only_search()（工单一基线，用于对比）。
"""
import numpy as np
from config import (
    TOP_K, RECALL_TOP_N, RRF_K, VECTOR_WEIGHT, BM25_WEIGHT, USE_MMR, MMR_LAMBDA,
    VECTORS_FILE,
)
from embedder import embed_texts, embed_query
from bm25 import BM25
from pdf_parser import get_texts
import os


class Retriever:
    """混合检索器：向量 + BM25 + 融合 + MMR 重排"""

    def __init__(self, chunks, use_cache=True):
        self.chunks = chunks
        self.texts = get_texts(chunks)
        # ---- 向量索引 ----
        if use_cache and os.path.exists(VECTORS_FILE):
            print(f"[检索器] 从缓存加载向量: {VECTORS_FILE}")
            self.vectors = np.load(VECTORS_FILE)
        else:
            print(f"[检索器] 生成向量（{len(self.texts)} 个文本块）...")
            self.vectors = embed_texts(self.texts)
            if use_cache:
                np.save(VECTORS_FILE, self.vectors)
        # ---- BM25 倒排索引 ----
        print("[检索器] 构建 BM25 倒排索引...")
        self.bm25 = BM25(self.texts)

    # ---------- 向量召回 ----------
    def _vector_recall(self, query, top_n):
        qv = embed_query(query)
        scores = np.dot(self.vectors, qv)
        idx = np.argsort(scores)[::-1][:top_n]
        return [(int(i), float(scores[i])) for i in idx]

    # ---------- 融合 ----------
    def _fuse(self, vec_ranked, bm25_ranked):
        """加权归一化融合：各自 min-max 归一化后按权重相加"""
        def norm(ranked):
            if not ranked:
                return {}
            vals = [s for _, s in ranked]
            lo, hi = min(vals), max(vals)
            span = (hi - lo) or 1.0
            return {i: (s - lo) / span for i, s in ranked}

        vn, bn = norm(vec_ranked), norm(bm25_ranked)
        fused = {}
        for i in set(vn) | set(bn):
            fused[i] = VECTOR_WEIGHT * vn.get(i, 0.0) + BM25_WEIGHT * bn.get(i, 0.0)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    # ---------- MMR 重排 ----------
    def _mmr(self, ranked, top_k, lam=MMR_LAMBDA):
        if not ranked:
            return []
        cand = [i for i, _ in ranked]
        score_map = dict(ranked)
        vecs = {i: self.vectors[i] for i in cand}
        selected, remaining = [], list(cand)
        while remaining and len(selected) < top_k:
            if not selected:
                best = max(remaining, key=lambda i: score_map[i])
            else:
                def mmr_score(i):
                    sim_to_sel = max(float(np.dot(vecs[i], vecs[j])) for j in selected)
                    return lam * score_map[i] - (1 - lam) * sim_to_sel
                best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)
        return [(i, score_map[i]) for i in selected]

    # ---------- 对外接口 ----------
    def _recall_idx(self, query, top_n):
        """内部：混合召回，返回 [(chunk_index, score), ...]"""
        vec_ranked = self._vector_recall(query, top_n)
        bm25_ranked = self.bm25.search(query, top_n)
        return self._fuse(vec_ranked, bm25_ranked)[:top_n]

    def recall(self, query, top_n=RECALL_TOP_N):
        """第一阶段：混合召回，返回较大候选集 [(chunk_dict, score), ...]（供重排使用）"""
        return [(self.chunks[i], s) for i, s in self._recall_idx(query, top_n)]

    def search(self, query, top_k=TOP_K):
        """优化版混合检索（召回+融合+MMR），返回 [(chunk_dict, score), ...]"""
        fused = self._recall_idx(query, RECALL_TOP_N)
        fused = self._mmr(fused, top_k) if USE_MMR else fused[:top_k]
        return [(self.chunks[i], s) for i, s in fused]

    def vector_only_search(self, query, top_k=TOP_K):
        """工单一基线：仅向量检索，返回 [(chunk_dict, score), ...]"""
        return [(self.chunks[i], s) for i, s in self._vector_recall(query, top_k)]


if __name__ == "__main__":
    from pdf_parser import build_chunks, load_chunks
    from config import CHUNKS_FILE

    cs = load_chunks() if os.path.exists(CHUNKS_FILE) else build_chunks()
    r = Retriever(cs)
    q = "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"
    print(f"\n查询: {q}")
    for i, (c, s) in enumerate(r.search(q), 1):
        print(f"[{i}] 页{c['page']} 分数{s:.4f}: {c['text'][:120]}")
