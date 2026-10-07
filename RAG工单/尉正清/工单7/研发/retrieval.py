# 工单编号：人工智能NLP-RAG-功能测试及评估
"""三种检索策略的统一入口：向量检索 / 全文检索 / 混合检索

工单6 要求「提供向量检索、全文检索以及两者同时执行的混合检索的检索策略的
配置及应用的功能」，并支持**权重调整**与**多种融合算法**。

    mode=vector    只走向量召回（BGE-M3 稠密+稀疏 RRF），可选重排
    mode=fulltext  只走倒排索引（布尔/短语/模糊）
    mode=hybrid    两路都跑，按 alpha 权重融合（alpha 是向量那一路的权重）

融合算法三选一（fusion）：

    weighted  归一化后加权平均 —— 默认，分数可比，便于调权重
    rrf       按名次融合 —— 与分数量纲无关，两路分数量级差异大时更稳
    vote      投票制 —— 两路各自提名，同时被两路提名的排前面
"""
from config import HYBRID_FUSION, RERANK_ALPHA, RECALL_K, TOP_K
from rerank import build as build_reranker


def _minmax(pairs):
    """把分数压到 [0,1]，便于两路加权。全相等时统一给 1。"""
    if not pairs:
        return []
    scores = [s for _, s in pairs]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [(c, 1.0) for c, _ in pairs]
    return [(c, (s - lo) / (hi - lo)) for c, s in pairs]


def fuse(vector_hits, fulltext_hits, alpha=0.5, method=HYBRID_FUSION, top_k=TOP_K):
    """把两路结果合成一路。

    alpha 是向量那一路的权重（0~1），全文那一路取 1-alpha。
    """
    def key(chunk):
        return (chunk.get("page"), chunk.get("text", "")[:40])

    vec, full = _minmax(vector_hits), _minmax(fulltext_hits)

    if method == "rrf":
        # 按名次融合。RRF_K=60 取自常用经验值，削弱头名之间的差距
        scores = {}
        for hits, weight in ((vec, alpha), (full, 1 - alpha)):
            for rank, (chunk, _) in enumerate(hits):
                scores[key(chunk)] = scores.get(key(chunk), 0.0) + weight / (60 + rank)
        pool = {key(c): c for c, _ in list(vec) + list(full)}
        ranked = sorted(pool.items(), key=lambda kv: -scores.get(kv[0], 0.0))
        return [(c, scores.get(k, 0.0)) for k, c in ranked[:top_k]]

    if method == "vote":
        # 投票制：各路提名自己的前 N，同时被两路提名的优先
        pool, votes, detail = {}, {}, {}
        for hits, weight in ((vec, alpha), (full, 1 - alpha)):
            for chunk, score in hits:
                k = key(chunk)
                pool[k] = chunk
                votes[k] = votes.get(k, 0) + 1
                detail[k] = detail.get(k, 0.0) + weight * score
        ranked = sorted(pool.items(),
                        key=lambda kv: (-votes[kv[0]], -detail.get(kv[0], 0.0)))
        return [(c, float(votes[k])) for k, c in ranked[:top_k]]

    # weighted：归一化后加权平均
    merged = {}
    for hits, weight in ((vec, alpha), (full, 1 - alpha)):
        for chunk, score in hits:
            k = key(chunk)
            merged.setdefault(k, [chunk, 0.0])
            merged[k][1] += weight * score
    ranked = sorted(merged.values(), key=lambda x: -x[1])
    return [(c, float(s)) for c, s in ranked[:top_k]]


class Retriever:
    """一个知识库上的检索策略执行器。"""

    def __init__(self, store, index, rerank_name="none", chat=None):
        self.store = store
        self.index = index
        self.rerank_name = rerank_name
        self.chat = chat
        self._reranker = None

    # ---------- 各路召回 ----------
    def _vector(self, query, k):
        return self.store.search(query, top_k=k)

    def _fulltext(self, query, k):
        if self.index is None:
            return []
        return self.index.search(query, top_k=k, mode="and") or \
            self.index.search(query, top_k=k, mode="or")

    # ---------- 统一入口 ----------
    def search(self, query, mode="hybrid", alpha=RERANK_ALPHA, top_k=TOP_K,
               recall_k=RECALL_K, fusion=HYBRID_FUSION):
        """按指定策略检索。返回 [(chunk, score)]。"""
        if mode == "vector":
            hits = self._vector(query, top_k)
        elif mode == "fulltext":
            hits = self._fulltext(query, top_k)
        else:
            hits = fuse(self._vector(query, recall_k),
                        self._fulltext(query, recall_k),
                        alpha=alpha, method=fusion, top_k=recall_k)
            hits = hits[:top_k]
        return self._apply_rerank(query, hits, top_k)

    def _apply_rerank(self, query, hits, top_k):
        """按配置的重排算法重排。交叉编码器那条走 vector_store 的懒加载。"""
        if not hits or self.rerank_name in (None, "none"):
            return hits[:top_k]
        if self.rerank_name == "cross":
            from vector_store import rerank as cross_rerank
            return cross_rerank(query, hits, top_k)
        if self._reranker is None:
            self._reranker = build_reranker(self.rerank_name, chat=self.chat,
                                            index=self.index)
        if self._reranker is None:
            return hits[:top_k]
        return self._reranker.rerank(query, hits, top_k)
