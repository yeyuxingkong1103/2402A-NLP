# -*- coding: utf-8 -*-
"""混合检索：向量召回 + BM25 召回 + RRF 融合。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

沿用 02/03 工单的混合检索思想，并针对多轮场景补充：
    - 文档级路由：改写后的问句若含公司名，则优先召回对应招股说明书；
    - 话题实体注入：把会话话题实体并入检索查询，缓解省略式追问的信息缺失；
    - **批量向量编码**：一次 ask 的多路查询（核心问句 + 扩展查询）合并为单次
      Ollama 调用，避免逐条编码带来的线性耗时（本工单性能优化关键）。

性能说明：bge-m3 单次 HTTP 编码存在约 2s 的固定开销，若按「每路查询一次调用」
实现，4 路查询将耗时 ~8s，超出工单 ≤3s 的响应要求；改为批量后固定开销只付一次。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src import config
from src.embedding import Embedder, normalize
from src.knowledge_base import KnowledgeBase, _tokenize


@dataclass
class Retrieved:
    """检索结果条目。"""
    chunk_id: int
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rrf: float = 0.0


class Retriever:
    """向量 + BM25 + RRF 混合检索器。"""

    def __init__(self, kb: KnowledgeBase, embedder: Embedder | None = None):
        self.kb = kb
        self.embedder = embedder or Embedder()
        self._qcache: Dict[str, np.ndarray] = {}      # 查询向量缓存（跨轮复用）

    # -- 查询编码（批量 + 缓存） ---------------------------------------------
    def _encode_queries(self, queries: List[str]) -> np.ndarray:
        """批量编码查询：缺失项一次 HTTP 调用补齐，命中缓存则零开销。"""
        missing = [q for q in queries if q not in self._qcache]
        if missing:
            vecs = normalize(self.embedder.encode(missing))
            for q, v in zip(missing, vecs):
                self._qcache[q] = v.astype(np.float32)
        return np.vstack([self._qcache[q] for q in queries])

    # -- 单路召回 ------------------------------------------------------------
    def _vector_search(self, qv: np.ndarray, top_k: int,
                       allow: set[int] | None = None) -> List[Tuple[int, float]]:
        n = float(np.linalg.norm(qv))
        if n == 0:
            return []
        qv = qv / n
        sims = self.kb.vectors @ qv                      # 余弦相似度（向量已归一化）
        order = np.argsort(-sims)
        out: List[Tuple[int, float]] = []
        for idx in order:
            i = int(idx)
            if allow is not None and i not in allow:
                continue
            out.append((i, float(sims[i])))
            if len(out) >= top_k:
                break
        return out

    def _bm25_search(self, query: str, top_k: int,
                     allow: set[int] | None = None) -> List[Tuple[int, float]]:
        if self.kb.bm25 is None:
            return []
        scores = self.kb.bm25.get_scores(_tokenize(query))
        order = np.argsort(-scores)
        out: List[Tuple[int, float]] = []
        for idx in order:
            i = int(idx)
            if allow is not None and i not in allow:
                continue
            if scores[i] <= 0:
                break
            out.append((i, float(scores[i])))
            if len(out) >= top_k:
                break
        return out

    # -- 融合 ----------------------------------------------------------------
    @staticmethod
    def _rrf(rankings: List[List[Tuple[int, float]]], k: int = 60) -> Dict[int, float]:
        fused: Dict[int, float] = {}
        for ranking in rankings:
            for rank, (idx, _s) in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return fused

    def _doc_filter(self, query: str) -> set[int] | None:
        """文档级路由：命中公司名则只在该文档内召回。"""
        if not config.DOC_FILTER_ENABLE:
            return None
        matched = None
        best = ""
        for src, aliases in config.DOC_COMPANY.items():
            for alias in aliases:
                if alias in (query or "") and len(alias) > len(best):
                    matched, best = src, alias
        if not matched:
            return None
        return {i for i, c in enumerate(self.kb.chunks) if c.source == matched}

    def search(self, query: str, top_k: int | None = None,
               extra_queries: List[str] | None = None) -> List[Retrieved]:
        """混合检索，返回融合后的候选列表。"""
        top_k = top_k or config.TOP_K
        queries: List[str] = []
        for q in [query] + list(extra_queries or []):
            q = (q or "").strip()
            if q and q not in queries:
                queries.append(q)
        if not queries:
            return []
        allow = self._doc_filter(query)
        qvecs = self._encode_queries(queries)            # 单次批量编码

        rankings: List[List[Tuple[int, float]]] = []
        vec_map: Dict[int, float] = {}
        bm_map: Dict[int, float] = {}
        for qi, q in enumerate(queries):
            vr = self._vector_search(qvecs[qi], config.VECTOR_TOP_K, allow)
            br = self._bm25_search(q, config.BM25_TOP_K, allow)
            rankings.append(vr)
            rankings.append(br)
            for i, s in vr:
                vec_map[i] = max(vec_map.get(i, -1e9), s)
            for i, s in br:
                bm_map[i] = max(bm_map.get(i, -1e9), s)

        fused = self._rrf(rankings, config.RANK_FUSION_K)
        results: List[Retrieved] = []
        for i, rrf in sorted(fused.items(), key=lambda kv: -kv[1]):
            results.append(Retrieved(chunk_id=i, score=rrf, rrf=rrf,
                                     vector_score=vec_map.get(i, 0.0),
                                     bm25_score=bm_map.get(i, 0.0)))
        return results