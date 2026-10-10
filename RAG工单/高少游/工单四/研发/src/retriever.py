# -*- coding: utf-8 -*-
"""检索模块（图像内容解析及检索优化版）：向量检索 + BM25 + RRF 融合 + 内容级重排
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

在 02 工单「多路查询 + 双路召回 + RRF + 重排」基础上，针对表格类问题：
- 召回侧：表格块（table）与键值对块（table_kv）均参与向量与 BM25 召回；
- 融合侧：同一表格的 Markdown 块与键值对块合并去重，避免同表多块挤占名额；
- 排序侧：交由 Reranker 对表格块给予额外加成（见 reranker.py）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src import config
from src.knowledge_base import get_all_documents, load_kb
from src.query_understanding import QueryAnalysis
from src.reranker import Reranker, ScoredDoc

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    """检索结果（兼容基线字段）。"""

    doc: Document
    score: float


def tokenize(text: str) -> List[str]:
    """中英文混合分词（jieba 搜索引擎模式）。"""
    tokens: List[str] = []
    for seg in jieba.cut_for_search(text or ""):
        seg = seg.strip().lower()
        if seg:
            tokens.extend(seg.split())
    return tokens


class BM25Index:
    """BM25 全局索引（构建一次，复用查询）。"""

    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.corpus = [tokenize(d.page_content) for d in docs]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def search(self, query: str, top_k: int) -> List[tuple[Document, float]]:
        if not self.bm25:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        return [(self.docs[i], float(scores[i])) for i in ranked[:top_k]]


# ---------------- 优化版检索器 -------------------------------------------------
class HybridRetriever:
    """向量 + BM25 混合检索 + 重排（优化版）。"""

    def __init__(self, store=None, corpus: Iterable[Document] | None = None,
                 vector_top_k: int = config.VECTOR_TOP_K,
                 bm25_top_k: int = config.BM25_TOP_K,
                 rrf_k: int = config.RANK_FUSION_K,
                 reranker: Reranker | None = None):
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.store = store or load_kb()
        docs = list(corpus) if corpus is not None else get_all_documents()
        self.bm25_index = BM25Index(docs)
        self.reranker = reranker or Reranker()
        # 【本工单新增】CLIP 跨模态图像索引（以文搜图）；不可用时优雅降级
        self.image_index = None
        if config.CLIP_ENABLE:
            try:
                from src.image_index import get_image_index

                self.image_index = get_image_index()
            except Exception as exc:
                logger.warning("图像索引加载失败（降级为纯文本检索）: %s", exc)

    def _vector_retrieve(self, query: str) -> List[tuple[Document, float]]:
        try:
            pairs = self.store.similarity_search_with_score(query, k=self.vector_top_k)
        except Exception as exc:
            logger.warning("vector retrieve failed: %s", exc)
            return []
        out = []
        for doc, dist in pairs:
            sim = 1.0 / (1.0 + float(dist))   # L2 距离 → 相似度
            out.append((doc, sim))
        return out

    @staticmethod
    def _rrf(rank_lists: List[List[Document]], rrf_k: int) -> Dict[str, float]:
        agg: Dict[str, float] = {}
        for rl in rank_lists:
            for rank, doc in enumerate(rl):
                key = doc.page_content
                agg[key] = agg.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        return agg

    def retrieve_candidates(self, query: str) -> List[tuple[Document, float, float]]:
        """返回 [(文档, 向量相似度, BM25分)] 的候选集合。"""
        vec = self._vector_retrieve(query)
        bm = self.bm25_index.search(query, self.bm25_top_k)

        vec_map = {d.page_content: s for d, s in vec}
        bm_map = {d.page_content: s for d, s in bm}

        rrf = self._rrf([[d for d, _ in vec], [d for d, _ in bm]], self.rrf_k)
        ranked_keys = sorted(rrf.keys(), key=lambda k: rrf[k], reverse=True)[
            : self.vector_top_k + self.bm25_top_k]

        doc_map: Dict[str, Document] = {d.page_content: d for d, _ in vec}
        doc_map.update({d.page_content: d for d, _ in bm})

        return [(doc_map[k], vec_map.get(k, 0.0), bm_map.get(k, 0.0)) for k in ranked_keys]

    def retrieve(self, analysis: QueryAnalysis, k: int = config.TOP_K) -> List[ScoredDoc]:
        """多路查询召回 → 合并 → 重排 → 去重（按父块）。"""
        merged: Dict[str, tuple[Document, float, float]] = {}
        for q in analysis.retrieval_queries:
            for doc, v, b in self.retrieve_candidates(q):
                key = doc.page_content
                if key not in merged:
                    merged[key] = (doc, v, b)
                else:
                    ov, ob = merged[key][1], merged[key][2]
                    merged[key] = (doc, max(ov, v), max(ob, b))

        # ---- CLIP 跨模态「以文搜图」（本工单新增） ----
        # 查询文本经 Chinese-CLIP 编码后与图像向量做余弦相似度，命中「答案所在图形」，
        # 再把对应图像语义块注入候选池，并按相似度计入重排的 CLIP 分项。
        clip_map: Dict[str, float] = {}
        if self.image_index is not None and getattr(self.image_index, "ready", False):
            try:
                hits = self.image_index.search(
                    analysis.core or analysis.question,
                    top_k=config.IMAGE_TOP_K,
                    source=analysis.doc_hint or None,
                )
            except Exception as exc:
                logger.debug("以文搜图失败: %s", exc)
                hits = []
            fig_by_sp = {(d.metadata.get("source"), d.metadata.get("page")): d
                         for d in self.bm25_index.docs
                         if d.metadata.get("ctype") == "figure"}
            for h in hits:
                if h.score < config.FIGURE_ROUTE_MIN_SIM:
                    continue
                doc = fig_by_sp.get((h.meta.get("source"), h.meta.get("page")))
                if doc is None:
                    continue
                key = doc.page_content
                clip_map[key] = max(clip_map.get(key, 0.0), float(h.score))
                if key not in merged:
                    merged[key] = (doc, float(h.score), 0.0)

        # ---- 文档级路由（本工单新增）：多文档消歧 ----
        # 两份招股说明书存在同构表格，若不按公司名限定文档，会出现跨文档串味
        # （问力源却召回兴图新科的“发行股数/募集资金”表）。此处按 doc_hint 过滤；
        # 若过滤后候选过少（说明该文档命中不足），则退回全量候选，保证召回不塌陷。
        # 过滤阈值固定为「至少 5 个候选」，不能用答案池大小（否则阈值过高导致过滤
        # 永不生效，串味问题依旧）。
        if config.DOC_FILTER_ENABLE and analysis.doc_hint:
            filtered = {k: v for k, v in merged.items()
                        if v[0].metadata.get("source") == analysis.doc_hint}
            if len(filtered) >= 5:
                merged = filtered

        scored = self.reranker.rerank(analysis, list(merged.values()),
                                      top_n=max(k * 3, 15), clip_map=clip_map)

        # 去重：同一小节 / 同一表格的「同类型」块只保留得分最高者。
        # 注意：表格的 Markdown 块（table）与键值对块（table_kv）内容互补，
        # 必须分别保留（故去重键带 ctype），否则数值型问题会丢失键值对线索。
        seen, out = set(), []
        for sd in scored:
            parent = sd.doc.metadata.get("parent") or sd.doc.page_content
            ctype = sd.doc.metadata.get("ctype", "text")
            key = f"{parent}|{ctype}"
            if key in seen:
                continue
            seen.add(key)
            out.append(sd)
            if len(out) >= k:
                break
        return out


# ---------------- 基线检索器（复刻 01/02 工单，用于对比） ---------------------
class BaselineRetriever:
    """基线混合检索器：向量 + BM25 + RRF，无重排（与 01/02 工单一致）。"""

    def __init__(self, store=None, corpus: Iterable[Document] | None = None,
                 vector_top_k: int = config.BASE_TOP_K,
                 bm25_top_k: int = config.BASE_BM25_TOP_K,
                 rrf_k: int = config.BASE_RRF_K):
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.store = store or load_kb()
        docs = list(corpus) if corpus is not None else get_all_documents()
        self._docs = docs
        self._corpus_tokens = [tokenize(d.page_content) for d in docs]

    def _vector_retrieve(self, query: str) -> List[Document]:
        try:
            return self.store.similarity_search(query, k=self.vector_top_k)
        except Exception as exc:
            logger.warning("vector retrieve failed: %s", exc)
            return []

    def _bm25_retrieve(self, query: str) -> List[Document]:
        if not self._corpus_tokens:
            return []
        q = tokenize(query)
        if not q:
            return []
        bm25 = BM25Okapi(self._corpus_tokens)
        scores = bm25.get_scores(q)
        ranked = sorted(range(len(self._docs)), key=lambda i: scores[i], reverse=True)
        return [self._docs[i] for i in ranked[: self.bm25_top_k]]

    @staticmethod
    def _rrf(rankings: List[List[Document]], rrf_k: int) -> List[RetrievedDoc]:
        agg: Dict[str, float] = {}
        by_id: Dict[str, Document] = {}
        for rl in rankings:
            for rank, doc in enumerate(rl):
                cid = doc.page_content
                by_id.setdefault(cid, doc)
                agg[cid] = agg.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = sorted(by_id.keys(), key=lambda cid: agg[cid], reverse=True)
        return [RetrievedDoc(by_id[cid], agg[cid]) for cid in ranked]

    def retrieve(self, query: str, k: int = config.BASE_TOP_K) -> List[RetrievedDoc]:
        vec = self._vector_retrieve(query)
        bm = self._bm25_retrieve(query)
        fused = self._rrf([vec, bm], self.rrf_k)
        return fused[: max(k, self.vector_top_k)]


def format_contexts(results: Iterable) -> str:
    """把检索结果格式化为提示词上下文文本。"""
    blocks = []
    for i, r in enumerate(results, 1):
        doc = r.doc if hasattr(r, "doc") else r
        page = doc.metadata.get("page", "?")
        src = doc.metadata.get("source", "")
        blocks.append(f"[文档{i}][{src} 第{page}页]\n{doc.page_content}")
    return "\n\n".join(blocks)