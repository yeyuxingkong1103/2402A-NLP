# -*- coding: utf-8 -*-
"""
工单十五：混合检索器（跨模态检索优化）
  - 文本召回：BM25 + bge-m3 向量 → RRF 融合 → bge-reranker 重排
  - 跨模态增强：检测到视觉引用(图号/页码)时，注入图纸结构化描述做二次召回与重排
"""
import os
import time
from typing import Any, Dict, List
from loguru import logger
import numpy as np
import jieba

from src.embedding import get_embedder
from src.vector_store import VectorStore
from src.query_router import detect_figure_refs, rewrite_query_for_retrieval
from src.image_knowledge import ImageKnowledgeBase

_bm25_index = None
_bm25_chunks = None

STOPWORDS = set("的 了 和 是 在 有 我 他 她 它 这 那 就 不 也 都 一 个 上 下 中 到 说 去 你 好 为 什么 怎么 哪 谁 几 多 少 吗 呢 吧 啊 哦 嗯 会 能 可以 应该 需要 希望 想 看 听 问 答 知道 了解".split())


def _build_bm25_index(chunks_list):
    global _bm25_index, _bm25_chunks
    from rank_bm25 import BM25Okapi
    corpus = []
    for c in chunks_list:
        text = c.get("content", "") or c.get("text", "")
        tokens = [t for t in jieba.lcut(text) if len(t.strip()) > 1 and t not in STOPWORDS]
        corpus.append(tokens)
    _bm25_index = BM25Okapi(corpus)
    _bm25_chunks = chunks_list
    logger.info(f"BM25 索引构建完成: {len(chunks_list)} chunks")


class Retriever:
    def __init__(self, vector_store=None, top_k=5, use_rerank=True, use_bm25=True,
                 bm25_top_k=30, vector_top_k=30, rrf_k=60):
        self.vs = vector_store or VectorStore()
        self.top_k = top_k
        self.use_rerank = use_rerank
        self.use_bm25 = use_bm25
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.embedder = get_embedder()
        self.img_kb = ImageKnowledgeBase()

        self._reranker = None
        if use_rerank:
            try:
                from FlagEmbedding import FlagReranker
                self._reranker = FlagReranker(os.getenv("RERANKER_PATH", "/home/dabaie/models/bge-reranker-v2-m3"), use_fp16=True)
                logger.info("Reranker 加载完成")
            except Exception as e:
                logger.warning(f"Reranker 加载失败: {e}")
                self.use_rerank = False

        if use_bm25:
            self._ensure_bm25_index()

    def _ensure_bm25_index(self):
        global _bm25_index
        if _bm25_index is not None:
            return
        try:
            chunks = self.vs.get_all_chunks()
            if chunks:
                _build_bm25_index(chunks)
        except Exception as e:
            logger.warning(f"BM25 索引构建失败: {e}")
            self.use_bm25 = False

    def retrieve(self, query, top_k=None, doc_id=None):
        """跨模态检索：检测视觉引用 → 注入图纸描述 → 多路召回 → RRF → 重排"""
        k = top_k or self.top_k
        t0 = time.time()
        refs = detect_figure_refs(query)

        # 跨模态增强：若引用了某图，把该图的结构化描述拼入查询做增强召回
        fig_desc = ""
        if refs["fig_no"] is not None or refs["page"] is not None:
            fig_desc = self.img_kb.get_description(fig_no=refs["fig_no"], page=refs["page"])
        enhanced_query = rewrite_query_for_retrieval(query, fig_desc if fig_desc else None)

        # 向量召回（用增强后的查询）
        query_vec = self.embedder.encode([enhanced_query], show_progress_bar=False)[0]
        vec_raw = self.vs.search(query_vec, top_k=self.vector_top_k, doc_id=doc_id) or []
        vec_ms = (time.time() - t0) * 1000

        # BM25 召回（用原始查询，关键词匹配更准）
        bm25_raw = []
        if self.use_bm25 and _bm25_index is not None:
            t1 = time.time()
            tokens = [t for t in jieba.lcut(query) if len(t.strip()) > 1]
            scores = _bm25_index.get_scores(tokens)
            top_idx = np.argsort(scores)[::-1][:self.bm25_top_k]
            for i in top_idx:
                if scores[i] > 0 and i < len(_bm25_chunks):
                    c = _bm25_chunks[i]
                    bm25_raw.append({
                        "chunk_id": c.get("chunk_id", f"bm25_{i}"),
                        "content": c.get("content", ""),
                        "page": c.get("page", 0),
                        "metadata": c.get("metadata", {}),
                        "bm25_score": float(scores[i]),
                    })
            bm25_ms = (time.time() - t1) * 1000
        else:
            bm25_ms = 0

        fused = self._rrf_fuse(vec_raw, bm25_raw)

        # 重排
        if self.use_rerank and self._reranker is not None and len(fused) > 0:
            pairs = [(query, c.get("content", "")) for c in fused[:min(len(fused), 20)]]
            scores = self._reranker.compute_score(pairs)
            if not isinstance(scores, list):
                scores = [scores]
            for i, c in enumerate(fused[:len(scores)]):
                c["rerank_score"] = float(scores[i])
            fused.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        for r in fused:
            r["score"] = round(max(r.get("rerank_score", 0), r.get("rrf_score", 0) / 100, 0), 4)
            r["preview"] = r.get("content", "")[:120]

        logger.info(f"检索完成: {len(fused)} chunks | fig={refs} | vec={vec_ms:.0f}ms bm25={bm25_ms:.0f}ms")
        return fused[:k]

    def get_figure_context(self, query: str) -> Dict[str, str]:
        """返回查询引用的图纸上下文（描述 + 路径），供 RAG 引擎拼入 Prompt"""
        refs = detect_figure_refs(query)
        info = self.img_kb.get(fig_no=refs["fig_no"], page=refs["page"])
        if not info:
            return {}
        return {
            "fig_no": info.get("fig_no"),
            "page": info.get("page"),
            "caption": info.get("caption", ""),
            "description": self.img_kb.get_description(fig_no=info.get("fig_no"), page=info.get("page")),
            "image_path": info.get("image_path", ""),
        }

    def _rrf_fuse(self, vec_results, bm25_results):
        scores = {}
        docs = {}
        for rank, d in enumerate(vec_results, 1):
            cid = d.get("chunk_id") or f"v_{rank}"
            if cid not in scores:
                scores[cid] = 0.0
                docs[cid] = d
            scores[cid] += 1.0 / (self.rrf_k + rank)
        for rank, d in enumerate(bm25_results, 1):
            cid = d.get("chunk_id") or f"b_{rank}"
            if cid not in scores:
                scores[cid] = 0.0
                docs[cid] = d
            scores[cid] += 1.0 / (self.rrf_k + rank)
        sorted_cids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        out = []
        for cid in sorted_cids:
            d = docs[cid]
            d["chunk_id"] = cid
            d["rrf_score"] = scores[cid]
            out.append(d)
        return out
