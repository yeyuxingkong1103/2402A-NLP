# -*- coding: utf-8 -*-
"""
统一检索模块（向量检索 / 全文检索 / 混合检索）
工单编号：人工智能NLP-RAG-混合检索任务

工单要求提供三种可配置、可切换的检索策略：
  1. vector —— 向量检索（召回 + 重排）
  2. fulltext —— 全文检索（BM25 倒排索引）
  3. hybrid —— 混合检索（同时执行前两者，按权重/投票融合）

融合算法实现三种（工单要求「加权平均、投票机制等」）：
  · weighted  —— 加权平均（归一化后线性加权）
  · rrf       —— Reciprocal Rank Fusion 倒数排名融合（对分数量纲不敏感，最稳健）
  · vote      —— 投票机制（两个通道都命中的片段获得额外加分）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from . import config
from .bm25 import BM25Retriever
from .rerank import CascadeReranker, Reranker, get_reranker
from .vectorstore import VectorStore

FusionMode = Literal["weighted", "rrf", "vote"]


@dataclass
class RetrievalResult:
    """一次检索的完整痕迹，便于工单13 做性能分析与工单07/09 做评估。"""
    query: str
    docs: list[dict]
    strategy: str
    fusion: str | None = None
    reranker: str | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query, "strategy": self.strategy,
            "fusion": self.fusion, "reranker": self.reranker,
            "timings": {k: round(v, 4) for k, v in self.timings.items()},
            "n_docs": len(self.docs),
            "docs": self.docs,
        }


class Retriever:
    """
    统一检索器。

    用法：
        r = Retriever("prospectus1")
        res = r.retrieve("军用领域收入", strategy="hybrid", fusion="rrf",
                         reranker="cascade", top_k=5)
    """

    def __init__(self, collection: str, bm25_path=None):
        self.collection = collection
        self.vs = VectorStore(collection)
        self.bm25: BM25Retriever | None = None
        self._bm25_path = bm25_path
        self._rerankers: dict[str, Reranker] = {}

    # -- 索引装载 -----------------------------------------------------------
    def load_bm25(self) -> BM25Retriever:
        if self.bm25 is None:
            self.bm25 = BM25Retriever.load(self._bm25_path)
        return self.bm25

    def get_reranker(self, name: str) -> Reranker | None:
        if name in ("none", None):
            return None
        if name not in self._rerankers:
            r = get_reranker(name)
            if isinstance(r, CascadeReranker):
                # 级联重排需要 IDF 统计，用 BM25 索引里的语料拟合
                try:
                    r.fit(self.load_bm25().docs)
                except Exception:
                    pass
            self._rerankers[name] = r
        return self._rerankers[name]

    # -- 三种检索策略 -------------------------------------------------------
    def vector_search(self, query: str, top_k: int) -> list[dict]:
        docs = self.vs.search(query, top_k=top_k)
        for d in docs:
            d.setdefault("source", "vector")
        return docs

    def fulltext_search(self, query: str, top_k: int,
                        boolean: str | None = None) -> list[dict]:
        return self.load_bm25().search(query, top_k=top_k, boolean=boolean)

    # -- 融合算法 -----------------------------------------------------------
    @staticmethod
    def _normalize(docs: list[dict]) -> list[dict]:
        """把分数归一化到 [0,1]，消除向量相似度与 BM25 分数的量纲差异。"""
        if not docs:
            return []
        scores = [d.get("score", 0.0) for d in docs]
        lo, hi = min(scores), max(scores)
        span = hi - lo
        out = []
        for d, s in zip(docs, scores):
            nd = dict(d)
            nd["norm_score"] = (s - lo) / span if span > 1e-9 else 1.0
            out.append(nd)
        return out

    def _fuse_weighted(self, vec: list[dict], bm: list[dict],
                       alpha: float) -> list[dict]:
        """加权平均：alpha 为向量权重，(1-alpha) 为全文权重。"""
        pool: dict[str, dict] = {}
        for d in self._normalize(vec):
            pool[d["chunk_id"]] = {**d, "final_score": alpha * d["norm_score"]}
        for d in self._normalize(bm):
            if d["chunk_id"] in pool:
                pool[d["chunk_id"]]["final_score"] += (1 - alpha) * d["norm_score"]
                pool[d["chunk_id"]]["source"] = "hybrid"
            else:
                pool[d["chunk_id"]] = {**d, "final_score": (1 - alpha) * d["norm_score"]}
        return sorted(pool.values(), key=lambda x: -x["final_score"])

    def _fuse_rrf(self, vec: list[dict], bm: list[dict], k: int = config.RRF_K) -> list[dict]:
        """
        Reciprocal Rank Fusion：score = Σ 1/(k + rank)
        优点：只看排名不看分数，天然免疫量纲问题，工业界标配。
        """
        pool: dict[str, dict] = {}
        for rank, d in enumerate(vec):
            pool[d["chunk_id"]] = {**d, "final_score": 1.0 / (k + rank + 1),
                                   "ranks": {"vector": rank + 1}}
        for rank, d in enumerate(bm):
            if d["chunk_id"] in pool:
                pool[d["chunk_id"]]["final_score"] += 1.0 / (k + rank + 1)
                pool[d["chunk_id"]]["ranks"]["bm25"] = rank + 1
                pool[d["chunk_id"]]["source"] = "hybrid"
            else:
                pool[d["chunk_id"]] = {**d, "final_score": 1.0 / (k + rank + 1),
                                       "ranks": {"bm25": rank + 1}}
        return sorted(pool.values(), key=lambda x: -x["final_score"])

    def _fuse_vote(self, vec: list[dict], bm: list[dict]) -> list[dict]:
        """投票机制：命中两个通道的片段记 2 票，仅命中一个记 1 票，票同则比归一化分数和。"""
        votes: dict[str, int] = {}
        pool: dict[str, dict] = {}
        for d in self._normalize(vec):
            votes[d["chunk_id"]] = votes.get(d["chunk_id"], 0) + 1
            pool[d["chunk_id"]] = {**d, "final_score": d["norm_score"]}
        for d in self._normalize(bm):
            votes[d["chunk_id"]] = votes.get(d["chunk_id"], 0) + 1
            if d["chunk_id"] in pool:
                pool[d["chunk_id"]]["final_score"] += d["norm_score"]
                pool[d["chunk_id"]]["source"] = "hybrid"
            else:
                pool[d["chunk_id"]] = {**d, "final_score": d["norm_score"]}
        for cid, v in votes.items():
            pool[cid]["votes"] = v
            pool[cid]["final_score"] += v          # 投票数作为主排序键
        return sorted(pool.values(), key=lambda x: (-x.get("votes", 0), -x["final_score"]))

    # -- 主入口 -------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        strategy: Literal["vector", "fulltext", "hybrid"] = "hybrid",
        top_k: int = config.TOP_K_RERANK,
        recall_k: int = config.TOP_K_RECALL,
        reranker: str = "none",
        fusion: FusionMode = "rrf",
        alpha: float = config.HYBRID_ALPHA,
        boolean: str | None = None,
    ) -> RetrievalResult:
        """
        执行检索。

        Args:
            strategy: vector / fulltext / hybrid
            top_k: 最终返回条数
            recall_k: 召回阶段条数（重排前）
            reranker: none / llm / tfidf / adaptive / cascade
            fusion: hybrid 时的融合算法 weighted / rrf / vote
            alpha: weighted 融合下向量权重
        """
        import time
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        if strategy == "vector":
            t = time.perf_counter()
            docs = self.vector_search(query, recall_k)
            timings["vector_recall"] = time.perf_counter() - t
        elif strategy == "fulltext":
            t = time.perf_counter()
            docs = self.fulltext_search(query, recall_k, boolean=boolean)
            timings["fulltext_recall"] = time.perf_counter() - t
        elif strategy == "hybrid":
            t = time.perf_counter()
            vec = self.vector_search(query, recall_k)
            timings["vector_recall"] = time.perf_counter() - t

            t = time.perf_counter()
            bm = self.fulltext_search(query, recall_k, boolean=boolean)
            timings["fulltext_recall"] = time.perf_counter() - t

            t = time.perf_counter()
            if fusion == "weighted":
                docs = self._fuse_weighted(vec, bm, alpha)
            elif fusion == "rrf":
                docs = self._fuse_rrf(vec, bm)
            elif fusion == "vote":
                docs = self._fuse_vote(vec, bm)
            else:
                raise ValueError(f"未知融合算法：{fusion}")
            timings["fusion"] = time.perf_counter() - t
        else:
            raise ValueError(f"未知检索策略：{strategy}")

        # 重排
        rr = self.get_reranker(reranker)
        if rr is not None and docs:
            t = time.perf_counter()
            docs = rr.rerank(query, docs, top_k=top_k)
            timings["rerank"] = time.perf_counter() - t
        else:
            docs = docs[:top_k]

        timings["total"] = time.perf_counter() - t0
        return RetrievalResult(
            query=query, docs=docs, strategy=strategy,
            fusion=fusion if strategy == "hybrid" else None,
            reranker=reranker if rr else None, timings=timings,
        )

    # -- 便捷方法 -----------------------------------------------------------
    def search(self, query: str, **kw) -> list[dict]:
        return self.retrieve(query, **kw).docs

    def format_context(self, docs: list[dict], max_chars: int = 4000) -> str:
        """把检索结果拼成给 LLM 的上下文，带来源标注便于溯源。"""
        parts, used = [], 0
        for i, d in enumerate(docs, 1):
            header = f"[片段{i}] 来源：《{d.get('doc', '')}》第{d.get('page', '?')}页"
            if d.get("section"):
                header += f" 章节：{d['section']}"
            if d.get("type") == "table":
                header += " （表格）"
            elif d.get("type") == "image":
                header += " （图表）"
            piece = f"{header}\n{d.get('text', '')}"
            if used + len(piece) > max_chars:
                break
            parts.append(piece)
            used += len(piece)
        return "\n\n---\n\n".join(parts)
