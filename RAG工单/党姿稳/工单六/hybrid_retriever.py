# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
混合检索器：统一封装三种检索策略
  1. 向量检索（召回 + 重排）  : m3e/bge 向量召回 Top-N → 重排器（LLM / TF-IDF / 用户反馈）→ Top-K
  2. 全文检索               : 倒排索引，支持布尔 / 短语 / 模糊 / 多字段
  3. 混合检索               : 同时执行两者，按 权重加权平均 或 投票(RRF) 融合
支持动态调整向量/全文权重与融合算法，并支持切换嵌入模型。
"""
import numpy as np

from config import RECALL_TOP_N, TOP_K, VECTOR_WEIGHT, BM25_WEIGHT, VECTORS_FILE
from embedder import embed_texts, embed_query
from fulltext import FullTextIndex
from rerankers import build_reranker

STRATEGIES = ["vector", "fulltext", "hybrid"]
FUSIONS = ["weighted", "vote"]


class HybridRetriever:
    """混合检索器（向量 + 全文 + 融合）"""

    def __init__(self, chunks, embed_model=None, reranker="llm",
                 vector_weight=VECTOR_WEIGHT, fulltext_weight=None, fusion="weighted",
                 vectors_file=VECTORS_FILE, use_cache=True, feedback_alpha=0.3):
        self.chunks = chunks
        self.embed_model = embed_model
        self.reranker_name = reranker
        self.reranker = build_reranker(reranker, feedback_alpha)
        self.vector_weight = vector_weight
        self.fulltext_weight = BM25_WEIGHT if fulltext_weight is None else fulltext_weight
        self.fusion = fusion

        texts = [c["text"] if isinstance(c, dict) else c for c in chunks]
        import os
        if use_cache and os.path.exists(vectors_file):
            print(f"[混合检索] 加载向量缓存: {vectors_file}")
            self.vectors = np.load(vectors_file)
            if self.vectors.shape[0] != len(texts):
                print(f"[混合检索] 缓存块数不符({self.vectors.shape[0]}≠{len(texts)})，重新编码")
                self.vectors = embed_texts(texts, model_name=embed_model)
                np.save(vectors_file, self.vectors)
        else:
            print(f"[混合检索] 向量编码 {len(texts)} 块（模型：{embed_model or '默认 m3e'}）...")
            self.vectors = embed_texts(texts, model_name=embed_model)
            if use_cache:
                np.save(vectors_file, self.vectors)

        print("[混合检索] 构建多字段倒排索引...")
        self.fulltext = FullTextIndex(chunks)

    # ---------- 1. 向量检索（召回+重排） ----------
    def vector_recall(self, query, top_n=RECALL_TOP_N):
        qv = embed_query(query, model_name=self.embed_model)
        scores = self.vectors @ qv
        idx = np.argsort(scores)[::-1][:top_n]
        return [(int(i), float(scores[i])) for i in idx]

    def vector_search(self, query, top_k=TOP_K, top_n=RECALL_TOP_N):
        """向量召回 + 重排，返回 [(chunk, score), ...]"""
        recall = [(self.chunks[i], s) for i, s in self.vector_recall(query, top_n)]
        if self.reranker is not None:
            return self.reranker.rerank(query, recall, top_k)
        return recall[:top_k]

    # ---------- 2. 全文检索 ----------
    def fulltext_search(self, query, top_k=TOP_K, mode="auto"):
        """倒排索引全文检索，返回 [(chunk, score), ...]"""
        return [(self.chunks[i], s) for i, s in self.fulltext.search(query, top_k, mode)]

    # ---------- 3. 混合检索 ----------
    @staticmethod
    def _rrf(rank_lists, k=60):
        """投票/排名融合（Reciprocal Rank Fusion）"""
        fused = {}
        for ranked in rank_lists:
            for rank, (i, _s) in enumerate(ranked, 1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _weighted(rank_lists, weights):
        """加权平均融合：各召回分数 min-max 归一化后按权重相加"""
        fused = {}
        for ranked, w in zip(rank_lists, weights):
            if not ranked:
                continue
            vals = [s for _, s in ranked]
            lo, hi = min(vals), max(vals)
            span = (hi - lo) or 1.0
            for i, s in ranked:
                fused[i] = fused.get(i, 0.0) + w * (s - lo) / span
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    def hybrid_search(self, query, top_k=TOP_K, top_n=RECALL_TOP_N,
                      vector_weight=None, fulltext_weight=None, fusion=None, ft_mode="auto"):
        """同时执行向量检索与全文检索，并按配置融合"""
        vw = self.vector_weight if vector_weight is None else vector_weight
        fw = self.fulltext_weight if fulltext_weight is None else fulltext_weight
        fu = self.fusion if fusion is None else fusion
        v_ranked = self.vector_recall(query, top_n)
        f_ranked = self.fulltext.search(query, top_n, ft_mode)
        if fu == "vote":
            fused = self._rrf([v_ranked, f_ranked])
        else:
            fused = self._weighted([v_ranked, f_ranked], [vw, fw])
        # 混合后可再叠加重排（LLM 成本高，默认只对候选做一次重排）
        cands = [(self.chunks[i], s) for i, s in fused[:max(top_n, top_k)]]
        if self.reranker is not None and self.reranker_name in ("llm", "feedback"):
            return self.reranker.rerank(query, cands, top_k)
        return cands[:top_k]

    # ---------- 统一入口 ----------
    def search(self, query, strategy="hybrid", top_k=TOP_K, **kw):
        """按策略检索：vector / fulltext / hybrid"""
        if strategy == "vector":
            return self.vector_search(query, top_k, kw.get("top_n", RECALL_TOP_N))
        if strategy == "fulltext":
            return self.fulltext_search(query, top_k, kw.get("ft_mode", "auto"))
        return self.hybrid_search(query, top_k, **kw)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    from kb import load_all_chunks
    allc, _, _ = load_all_chunks()
    r = HybridRetriever(allc)
    q = "武汉兴图新科电子股份有限公司法定代表人是谁？"
    for st in STRATEGIES:
        print(f"\n===== 策略：{st} =====")
        for i, (c, s) in enumerate(r.search(q, strategy=st, top_k=3), 1):
            print(f"  [{i}] 页{c['page']} {s:.4f}: {c['text'][:70]}")
