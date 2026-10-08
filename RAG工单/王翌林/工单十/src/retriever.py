# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-基于PDF文档的问答系统
src/retriever.py — 检索器（优化版 v2）

优化项：
  1. BM25 + 向量检索混合召回 → RRF 融合
  2. bge-reranker-v2-m3 重排序（默认开启）
  3. 查询级缓存（diskcache）
  4. 引用溯源增强（page + chunk_id + score + preview）

融合策略：
  - 向量召回 top_k=50（扩大候选池）
  - BM25 召回 top_k=50（关键词精确匹配）
  - RRF k=60 融合去重（用 chunk_id 作主键，避免 content md5 误合并）
  - reranker 重排后取 top_k
"""
import hashlib, os, time
from typing import Any, Dict, List, Optional
from loguru import logger
import numpy as np
import jieba  # 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— BM25 中文分词

from src.embedding import get_embedder
from src.vector_store import VectorStore

# ========== 工单：人工智能NLP-RAG-基于PDF文档的问答系统 ==========
# BM25 索引缓存（进程级，避免每次检索重复加载）
_bm25_index = None
_bm25_chunks = None  # 全量 chunks 列表，用于 BM25 打分
_BM25_CACHE_TS = 0

def _build_bm25_index(chunks_list):
    """工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 构建 BM25 索引"""
    global _bm25_index, _bm25_chunks, _BM25_CACHE_TS
    from rank_bm25 import BM25Okapi
    # 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— jieba 中文分词 + 停用词过滤
    STOPWORDS = set("的 了 和 是 在 有 我 他 她 它 这 那 就 不 也 都 一 个 上 下 中 到 说 去 你 好 为 什么 怎么 哪 谁 几 多 少 吗 呢 吧 啊 哦 嗯 会 能 可以 应该 需要 希望 想 看 听 问 答 知道 了解".split())
    corpus = []
    for c in chunks_list:
        text = c.get("content", "") or c.get("text", "")
        tokens = [t for t in jieba.lcut(text) if len(t.strip()) > 1 and t not in STOPWORDS]
        corpus.append(tokens)
    _bm25_index = BM25Okapi(corpus)
    _bm25_chunks = chunks_list
    _BM25_CACHE_TS = time.time()
    logger.info(f"✅ BM25 索引构建完成: {len(chunks_list)} chunks")

class Retriever:
    """工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 优化版检索器"""
    def __init__(self, vector_store=None, top_k=5,
                 use_rerank=True, use_bm25=True, use_cache=True,
                 reranker_model=None, bm25_top_k=50, vector_top_k=50, rrf_k=60):
        self.vs = vector_store or VectorStore()
        self.top_k = top_k
        self.use_rerank = use_rerank
        self.use_bm25 = use_bm25
        self.vector_top_k = vector_top_k  # 工单：扩大向量召回候选池
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.embedder = get_embedder()

        # ========== Reranker ==========
        self._reranker = None
        if use_rerank:
            try:
                from FlagEmbedding import FlagReranker
                model_path = reranker_model or os.getenv("RERANKER_PATH", "/home/dabaie/models/bge-reranker-v2-m3")
                self._reranker = FlagReranker(model_path, use_fp16=True)
                logger.info(f"✅ Reranker 加载完成: {model_path}")
            except Exception as e:
                logger.warning(f"Reranker 加载失败（跳过）: {e}"); self.use_rerank = False

        # ========== BM25 ==========
        if use_bm25:
            try:
                self._ensure_bm25_index()
            except Exception as e:
                logger.warning(f"BM25 索引构建失败（跳过）: {e}"); self.use_bm25 = False

        # ========== 缓存 ==========
        self._cache = None
        if use_cache:
            try:
                from diskcache import Cache
                self._cache = Cache("./data/retriever_cache")
                self._cache_ttl = 3600  # 1h
                logger.info("✅ 检索缓存已启用 (diskcache)")
            except Exception as e:
                logger.warning(f"缓存初始化失败: {e}"); self._cache = None

    # ========== BM25 索引构建 ==========
    def _ensure_bm25_index(self):
        """工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 懒加载 BM25 索引"""
        global _bm25_index
        if _bm25_index is not None: return
        # 从 Milvus 拉全量 chunks（仅 content + chunk_id + page + metadata）
        try:
            stats = self.vs.get_stats()
            if not stats.get("exists") or not stats.get("num_entities", 0):
                logger.warning("Milvus 为空，无法构建 BM25"); return
            # 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 用 Milvus scan 拉数据
            from pymilvus import Collection, utility
            col = Collection(self.vs.collection)
            col.load()
            chunks = []
            for batch in col.query(expr="chunk_id != ''", output_fields=["chunk_id", "content", "page", "metadata"], limit=2000):
                chunks.extend(batch)
            if len(chunks) > 2000: chunks = chunks[:2000]  # 安全上限
            _build_bm25_index(chunks)
        except Exception as e:
            logger.warning(f"BM25 索引构建跳过（非关键）: {e}")

    # ========== 检索主入口 ==========
    def retrieve(self, query, top_k=None, doc_id=None, rewritten_query=None):
        """工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 混合检索 + RRF + Rerank"""
        k = top_k or self.top_k
        effective_query = rewritten_query or query

        # ========== 缓存命中 ==========
        cache_key = None
        if self._cache is not None:
            cache_key = hashlib.md5(f"{effective_query}|{k}|{doc_id or ''}".encode()).hexdigest()
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info(f"📦 检索缓存命中: {effective_query[:40]}")
                return cached

        # ========== 向量召回 ==========
        t0 = time.time()
        query_vec = self.embedder.encode([effective_query], show_progress_bar=False)[0]
        vec_raw = self.vs.search(query_vec, top_k=self.vector_top_k, doc_id=doc_id) or []
        vec_ms = (time.time() - t0) * 1000
        logger.info(f"🔍 向量召回: {len(vec_raw)} chunks in {vec_ms:.0f}ms")

        # ========== BM25 召回 ==========
        bm25_raw = []
        bm25_ms = 0
        if self.use_bm25 and _bm25_index is not None:
            t1 = time.time()
            tokens = [t for t in jieba.lcut(effective_query) if len(t.strip()) > 1]
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
            logger.info(f"🔍 BM25 召回: {len(bm25_raw)} chunks in {bm25_ms:.0f}ms")

        # ========== RRF 融合（chunk_id 作主键）==========
        t2 = time.time()
        fused = self._rrf_fuse(vec_raw, bm25_raw, k=k)
        fuse_ms = (time.time() - t2) * 1000
        logger.info(f"🔀 RRF 融合: {len(fused)} chunks in {fuse_ms:.0f}ms")

        # ========== Rerank 重排序 ==========
        if self.use_rerank and self._reranker is not None and len(fused) > 0:
            t3 = time.time()
            pairs = [(effective_query, c.get("content", "")) for c in fused[:min(len(fused), 30)]]
            scores = self._reranker.compute_score(pairs)
            if not isinstance(scores, list): scores = [scores]
            for i, c in enumerate(fused[:len(scores)]):
                c["rerank_score"] = float(scores[i])
            fused.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            fused = fused[:k]  # rerank 后取最终 top_k
            rerank_ms = (time.time() - t3) * 1000
            logger.info(f"📊 Rerank 重排: top {k} in {rerank_ms:.0f}ms")

        # ========== 归一化 score + 引用溯源 ==========
        for r in fused:
            # 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 统一 score 语义（越大越好）
            r["score"] = round(max(
                r.get("rerank_score", 0),
                r.get("rrf_score", 0) / 100,  # RRF 分数归一化
                r.get("distance", 0),
                0
            ), 4)
            # 工单：引用溯源增强
            content = r.get("content", "")
            r["preview"] = content[:120]  # 延长 preview 到 120 字
            r["char_count"] = len(content)

        # ========== 写缓存 ==========
        if self._cache is not None and cache_key is not None:
            self._cache.set(cache_key, fused, expire=self._cache_ttl)

        total_ms = vec_ms + bm25_ms + fuse_ms
        logger.info(f"✅ 检索完成: {len(fused)} chunks | vec={vec_ms:.0f}ms bm25={bm25_ms:.0f}ms fuse={fuse_ms:.0f}ms")
        return fused[:k]

    # ========== RRF 融合实现 ==========
    def _rrf_fuse(self, vec_results, bm25_results, k=60):
        """
        工单：人工智能NLP-RAG-基于PDF文档的问答系统
        Reciprocal Rank Fusion — 用 chunk_id 作稳定主键去重
        RRF(d) = Σ 1/(k + rank_i)  其中 k 是平滑常数（默认 60）
        """
        scores = {}
        docs = {}

        # 向量路（rank 从 1 开始）
        for rank, d in enumerate(vec_results, 1):
            cid = d.get("chunk_id") or d.get("id") or f"v_{rank}"
            if cid not in scores:
                scores[cid] = 0.0; docs[cid] = d
            scores[cid] += 1.0 / (k + rank)

        # BM25 路
        for rank, d in enumerate(bm25_results, 1):
            cid = d.get("chunk_id") or f"b_{rank}"
            if cid not in scores:
                scores[cid] = 0.0; docs[cid] = d
            scores[cid] += 1.0 / (k + rank)

        # 排序 + 去重（chunk_id 作主键）
        sorted_cids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        out = []
        for cid in sorted_cids:
            d = docs[cid]
            d["chunk_id"] = cid  # 确保 chunk_id 存在
            d["rrf_score"] = scores[cid]
            out.append(d)
        return out

    # ========== Prompt 格式化 ==========
    @staticmethod
    def format_for_prompt(chunks):
        """工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 增强版引用溯源"""
        lines = []
        for i, c in enumerate(chunks, 1):
            page = c.get("page", "?")
            content = c.get("content", "").strip()
            chunk_id = c.get("chunk_id", "?")
            lines.append(f"[资料{i}] (第{page}页, chunk={chunk_id})\n{content}")
        return "\n\n".join(lines)
