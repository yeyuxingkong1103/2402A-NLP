# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
src/retrieval/hybrid_retriever_v6.py —— 工单六 统一可配置混合检索器

按 RetrievalConfig 执行三种检索策略：
  vector   ：bge-m3 向量召回（Milvus 余弦）→ 可配置重排器
  fulltext ：倒排索引全文检索（布尔/短语/模糊/多字段 TF-IDF）→ 可配置重排器
  hybrid   ：向量 + 全文并行召回 → RRF 投票 / 加权平均融合 → 可配置重排器

输出与 TextRetrieverV3 同构（source="text"），可直接被 rag_engine_v6 消费。
"""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from loguru import logger

from src.perf_v13 import current_request_id, stage, with_request_id
from src.retrieval.fusion import rrf_fuse, weighted_fuse
from src.retrieval.fulltext_retriever import FulltextRetriever
from src.retrieval.rerankers import get_reranker
from src.retrieval.retrieval_config import (
    FUSION_WEIGHTED, MODE_FULLTEXT, MODE_HYBRID, MODE_VECTOR,
    RetrievalConfig,
)

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"


class HybridRetrieverV6:
    """工单六：统一混合检索器（策略可配置）"""

    def __init__(self, config: Optional[RetrievalConfig] = None,
                 vector_store=None, embedder=None):
        self.config = config or RetrievalConfig()
        self._vector_store = vector_store
        self._embedder = embedder
        self._fulltext: Optional[FulltextRetriever] = None
        self._rerankers: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _get_vector_store(self):
        if self._vector_store is None:
            from src.vector_store import VectorStore
            self._vector_store = VectorStore()
        return self._vector_store

    def _get_embedder(self):
        if self._embedder is None:
            from src.embedding import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _get_fulltext(self) -> FulltextRetriever:
        if self._fulltext is None:
            self._fulltext = FulltextRetriever.get_instance()
        return self._fulltext

    def _get_reranker(self, name: str):
        """工单六：重排器进程内复用（LLM 重排器本身即全局单例）"""
        if name not in self._rerankers:
            self._rerankers[name] = get_reranker(name)
        return self._rerankers[name]

    # ------------------------------------------------------------------
    def vector_recall(self, query: str, doc_id: Optional[str],
                      recall_k: int) -> List[Dict[str, Any]]:
        """工单六：向量召回（bge-m3 + Milvus 余弦相似度）
        工单十三：细分嵌入耗时与 Milvus 检索耗时两个子阶段"""
        emb = self._get_embedder()
        with stage("vector_embed") as _se:
            qvec = emb.encode([query], show_progress_bar=False)[0]
        with stage("vector_milvus", recall_k=recall_k) as _sm:
            hits = self._get_vector_store().search(qvec, top_k=recall_k,
                                                   doc_id=doc_id)
        out = []
        for h in hits:
            out.append({
                "doc_id": h.get("doc_id", ""), "chunk_id": h.get("chunk_id", ""),
                "content": h.get("content") or h.get("text") or "",
                "page": h.get("page", 0), "metadata": h.get("metadata", {}),
                "score": float(h.get("score", 0.0)),
                "source": "text", "search_path": "vector",
            })
        return out

    def fulltext_recall(self, query: str, doc_id: Optional[str],
                        recall_k: int, cfg: RetrievalConfig) -> List[Dict[str, Any]]:
        """工单六：全文召回（倒排索引）"""
        return self._get_fulltext().search(
            query, top_k=recall_k, doc_id=doc_id, match=cfg.match,
            fields=cfg.fields, field_weights=cfg.field_weights)

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None,
                 doc_id: Optional[str] = None,
                 config: Optional[RetrievalConfig] = None,
                 rid: Optional[str] = None) -> Dict[str, Any]:
        """工单六：按配置执行检索 → 融合 → 重排

        工单十三：rid 为可选请求 ID（线程池工作线程恢复日志追踪用）；
        各子阶段（向量嵌入/Milvus/全文/融合/重排）写入结构化性能日志。

        Returns:
            {results, mode, fusion, reranker, vector_hits, fulltext_hits, elapsed_ms}
        """
        t0 = time.perf_counter()
        cfg = config or self.config
        k = top_k or cfg.top_k
        mode = cfg.mode
        vec_hits: List[Dict[str, Any]] = []
        ft_hits: List[Dict[str, Any]] = []
        fused: List[Dict[str, Any]] = []

        if mode == MODE_VECTOR:
            vec_hits = self.vector_recall(query, doc_id, cfg.vector_recall_k)
            fused = [dict(h, rrf_score=h["score"]) for h in vec_hits]
        elif mode == MODE_FULLTEXT:
            with stage("fulltext_recall") as _s:
                ft_hits = self.fulltext_recall(query, doc_id,
                                               cfg.fulltext_recall_k, cfg)
            fused = [dict(h, rrf_score=h["score"]) for h in ft_hits]
        else:  # 工单六：hybrid —— 两路并行召回
            # 工单十三：工作线程恢复请求 ID，保证子阶段日志可追踪
            with ThreadPoolExecutor(max_workers=2) as pool:
                fv = pool.submit(with_request_id(rid or current_request_id(),
                                                 self.vector_recall),
                                 query, doc_id, cfg.vector_recall_k)
                ff = pool.submit(with_request_id(rid or current_request_id(),
                                                 self.fulltext_recall),
                                 query, doc_id, cfg.fulltext_recall_k, cfg)
                with stage("vector_recall_wall") as _sv:
                    vec_hits = fv.result()
                with stage("fulltext_recall_wall") as _sf:
                    ft_hits = ff.result()
            if cfg.fusion == FUSION_WEIGHTED:
                with stage("fusion", vec=len(vec_hits), ft=len(ft_hits)) as _su:
                    fused = weighted_fuse(vec_hits, ft_hits,
                                          cfg.vector_weight, cfg.fulltext_weight,
                                          top_k=cfg.vector_recall_k + cfg.fulltext_recall_k)
                for h in fused:
                    h["rrf_score"] = h.get("weighted_score", 0.0)
            else:
                with stage("fusion", vec=len(vec_hits), ft=len(ft_hits)) as _su:
                    fused = rrf_fuse(vec_hits, ft_hits, rrf_k=cfg.rrf_k,
                                     top_k=cfg.vector_recall_k + cfg.fulltext_recall_k)

        # 工单六：可配置重排（llm / tfidf / adaptive）
        # 工单十三：重排是重点怀疑瓶颈，单独计时并记录候选数
        reranker = self._get_reranker(cfg.reranker)
        with stage("rerank", reranker=cfg.reranker,
                   candidates=len(fused)) as _sr:
            results = reranker.rerank(query, fused, top_k=k, content_key="content")

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"[hybrid_v6] mode={mode} fusion={cfg.fusion} "
                    f"reranker={cfg.reranker} vec={len(vec_hits)} "
                    f"ft={len(ft_hits)} out={len(results)} {elapsed:.0f}ms")
        return {
            "results": results, "mode": mode, "fusion": cfg.fusion,
            "reranker": cfg.reranker, "match": cfg.match,
            "vector_weight": cfg.vector_weight,
            "fulltext_weight": cfg.fulltext_weight,
            "vector_hits": len(vec_hits), "fulltext_hits": len(ft_hits),
            "elapsed_ms": round(elapsed, 1),
        }
