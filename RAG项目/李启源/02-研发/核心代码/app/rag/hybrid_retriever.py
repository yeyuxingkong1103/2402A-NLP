"""Dense + BM25 混合检索与 RRF 排名融合。

Dense 检索擅长语义相似问题，BM25 擅长订单号、型号、政策名称等关键词匹配。
二者原始分数不在同一量纲，因此这里不直接相加，而是按各自排名做 RRF 融合。
"""

from __future__ import annotations

import logging
from typing import Any

from app.rag.bm25 import BM25Scorer
from app.rag.retrieval_models import HybridSearchResult, SearchResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine dense retrieval with optional BM25 candidates."""

    def __init__(
        self,
        vector_retriever: Any,
        embedding_client: Any,
        *,
        enable_bm25: bool = True,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        candidate_provider: Any | None = None,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.embedding_client = embedding_client
        self.enable_bm25 = enable_bm25
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.candidate_provider = candidate_provider
        self.bm25_scorer = BM25Scorer() if enable_bm25 else None

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        final_top_k: int = 10,
        filters: dict[str, Any] | None = None,
        bm25_keywords: list[str] | None = None,
    ) -> HybridSearchResult:
        """执行两路召回并按块 ID 融合，返回统一的检索结果模型。"""
        # 第一路：问题向量化后从 Milvus 召回语义相近片段。
        dense_results = self._dense_search(query, top_k=top_k, filters=filters)

        # 第二路：只有启用 BM25、提取到关键词且存在候选数据源时才执行。
        sparse_results = (
            self._sparse_search(bm25_keywords, top_k=top_k, filters=filters)
            if self.enable_bm25 and bm25_keywords
            else tuple()
        )
        # 稀疏检索不可用时自然退化为 Dense-only，外部依赖异常不会让整次问答崩溃。
        if sparse_results:
            results = self._fuse_results(dense_results, sparse_results, final_top_k)
            fusion_method = "rrf"
        else:
            results = dense_results[:final_top_k]
            fusion_method = "dense_only"
        return HybridSearchResult(
            results=results,
            dense_results=dense_results[:final_top_k],
            sparse_results=sparse_results[:final_top_k],
            fusion_method=fusion_method,
            total_found=len(dense_results) + len(sparse_results),
        )

    def _dense_search(
        self, query: str, *, top_k: int, filters: dict[str, Any] | None
    ) -> tuple[SearchResult, ...]:
        """Embed the query and normalize vector-store records."""
        try:
            vectors = self.embedding_client.embed_texts([query])
            if not vectors:
                return tuple()
            raw_results = self.vector_retriever.search(
                vectors[0], top_k=top_k, filters=filters
            )
            return tuple(
                SearchResult(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    source=item["source"],
                    score=item["score"],
                    summary=item.get("summary", ""),
                    doc_id=item.get("doc_id"),
                    metadata={"dense_score": item["score"]},
                    search_type="dense",
                )
                for item in raw_results
            )
        except Exception as exc:
            logger.error("Dense search failed: %s", exc)
            return tuple()

    def _sparse_search(
        self,
        keywords: list[str],
        *,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> tuple[SearchResult, ...]:
        """Score one consistently filtered candidate corpus with BM25."""
        if not self.bm25_scorer or not keywords or not self.candidate_provider:
            return tuple()
        try:
            candidates = self._get_bm25_candidates(keywords, filters=filters, top_k=top_k)
            corpus = [str(item.get("text", "")) for item in candidates]
            scored = [
                (item, self.bm25_scorer.score(keywords, item["text"], documents=corpus))
                for item in candidates
            ]
            scored = sorted((item for item in scored if item[1] > 0), key=lambda x: x[1], reverse=True)
            return tuple(
                SearchResult(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    source=item["source"],
                    score=score,
                    summary=item.get("summary", ""),
                    doc_id=item.get("doc_id"),
                    metadata={"bm25_score": score},
                    search_type="sparse",
                )
                for item, score in scored[:top_k]
            )
        except Exception as exc:
            logger.error("Sparse search failed: %s", exc)
            return tuple()

    def _get_bm25_candidates(
        self, keywords: list[str], *, filters: dict[str, Any] | None, top_k: int
    ) -> list[dict[str, Any]]:
        """Call both current and legacy candidate-provider signatures."""
        try:
            return list(self.candidate_provider(keywords=keywords, filters=filters, top_k=top_k))
        except TypeError:
            return list(self.candidate_provider(filters))

    def _fuse_results(
        self,
        dense_results: tuple[SearchResult, ...],
        sparse_results: tuple[SearchResult, ...],
        top_k: int,
    ) -> tuple[SearchResult, ...]:
        """用加权 RRF 融合两种排名。

        公式为 ``weight / (60 + rank)``。常数 60 用于减小头部名次之间的剧烈差异；
        同一个块如果被两路同时召回，会同时获得 Dense 和 BM25 的排名贡献。
        """
        # 先把结果转换为 chunk_id -> 排名，chunk_id 是两路结果去重和对齐的主键。
        dense_ranks = {result.chunk_id: index for index, result in enumerate(dense_results)}
        sparse_ranks = {result.chunk_id: index for index, result in enumerate(sparse_results)}
        ranks = set(dense_ranks) | set(sparse_ranks)
        rrf_scores = {
            chunk_id: self.dense_weight / (60 + dense_ranks[chunk_id] + 1)
            if chunk_id in dense_ranks else 0.0
            for chunk_id in ranks
        }
        for chunk_id in ranks:
            if chunk_id in sparse_ranks:
                rrf_scores[chunk_id] += self.bm25_weight / (60 + sparse_ranks[chunk_id] + 1)

        # 保留原始分数和排名到 metadata，方便调参、问题定位和离线评测。
        chunk_map = {result.chunk_id: result for result in (*dense_results, *sparse_results)}
        ordered = sorted(ranks, key=rrf_scores.get, reverse=True)[:top_k]
        return tuple(
            SearchResult(
                chunk_id=chunk_map[chunk_id].chunk_id,
                text=chunk_map[chunk_id].text,
                source=chunk_map[chunk_id].source,
                score=rrf_scores[chunk_id],
                summary=chunk_map[chunk_id].summary,
                doc_id=chunk_map[chunk_id].doc_id,
                metadata={
                    "rrf_score": rrf_scores[chunk_id],
                    "dense_rank": dense_ranks.get(chunk_id, -1),
                    "sparse_rank": sparse_ranks.get(chunk_id, -1),
                    "original_dense_score": chunk_map[chunk_id].metadata.get("dense_score")
                    if chunk_id in dense_ranks and chunk_map[chunk_id].metadata else None,
                    "original_bm25_score": chunk_map[chunk_id].metadata.get("bm25_score")
                    if chunk_id in sparse_ranks and chunk_map[chunk_id].metadata else None,
                },
                search_type="hybrid",
            )
            for chunk_id in ordered
        )


def build_hybrid_retriever(
    embedding_client: Any,
    vector_retriever: Any,
    *,
    enable_bm25: bool = True,
    candidate_provider: Any | None = None,
) -> HybridRetriever:
    """Build the default 0.7 dense / 0.3 sparse retriever."""
    return HybridRetriever(
        vector_retriever=vector_retriever,
        embedding_client=embedding_client,
        enable_bm25=enable_bm25,
        bm25_weight=0.3,
        dense_weight=0.7,
        candidate_provider=candidate_provider,
    )


__all__ = ["HybridRetriever", "HybridSearchResult", "SearchResult", "build_hybrid_retriever"]
