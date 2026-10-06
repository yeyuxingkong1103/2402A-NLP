# -*- coding: utf-8 -*-
"""检索模块：向量检索 + BM25 混合检索（倒数排名融合）
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

说明：
- 向量检索：基于语义相似度（BGE-M3 嵌入 + Chroma）；
- 词法检索：基于 BM25（rank-bm25），对专有名词、数字、公司名等精确命中友好；
- 融合：采用 Reciprocal Rank Fusion (RRF)，兼顾语义与精确匹配，提升召回质量。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import jieba

from src import config
from src.knowledge_base import get_all_documents, load_kb

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    """单条检索结果。"""

    doc: Document
    score: float   # RRF 融合分


class HybridRetriever:
    """向量 + BM25 混合检索器。"""

    def __init__(
        self,
        store=None,
        corpus: Iterable[Document] | None = None,
        vector_top_k: int = config.TOP_K,
        bm25_top_k: int = config.BM25_TOP_K,
        rrf_k: int = config.RANK_FUSION_K,
    ):
        """构造混合检索器。

        Args:
            store: 向量检索器（object; FAISS/Chroma 均可）。默认从磁盘加载。
            corpus: 全量文档块（用于 BM25）。为 None 时尝试从磁盘加载。
        """
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self._corpus = corpus
        self.store = store or load_kb()

    # ---- 向量检索 -----------------------------------------------------------
    def _vector_retrieve(self, query: str) -> List[Document]:
        try:
            return self.store.similarity_search(query, k=self.vector_top_k)
        except Exception as exc:  # 向量检索失败时降级为空，交由融合兜底
            logger.warning("vector retrieve failed: %s", exc)
            return []

    # ---- BM25 词法检索 --------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # 中英文混合分词；英文小写按空格切分，中文按 jieba 切分
        tokens: List[str] = []
        for seg in jieba.cut_for_search(text):
            seg = seg.strip().lower()
            if seg and seg not in (" ", ""):
                tokens.extend(seg.split())
        return tokens

    def _bm25_retrieve(self, query: str, all_docs: List[Document]) -> List[Document]:
        corpus = [self._tokenize(d.page_content) for d in all_docs]
        q_tokens = self._tokenize(query)
        if not corpus or not q_tokens:
            return []
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(q_tokens)
        ranked = sorted(
            range(len(all_docs)), key=lambda i: scores[i], reverse=True
        )
        return [all_docs[i] for i in ranked[: self.bm25_top_k]]

    # ---- 松耦合：保证 BM25 也有全局语料 -------------------------------------
    def _all_chunks(self) -> List[Document]:
        if self._corpus is not None:
            return list(self._corpus)
        return get_all_documents()

    # ---- RRF 融合 --------------------------------------------------------------
    @staticmethod
    def _rrf(rankings: List[List[Document]], rrf_k: int) -> List[RetrievedDoc]:
        agg: dict[str, float] = {}
        doc_by_id: dict[str, Document] = {}
        for rank_list in rankings:
            for rank, doc in enumerate(rank_list):
                cid = doc.page_content  # 以内容作为文档唯一标识
                if cid not in doc_by_id:
                    doc_by_id[cid] = doc
                agg[cid] = agg.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = sorted(doc_by_id.keys(), key=lambda cid: agg[cid], reverse=True)
        return [RetrievedDoc(doc_by_id[cid], agg[cid]) for cid in ranked]

    def retrieve(self, query: str, k: int = config.TOP_K) -> List[RetrievedDoc]:
        """执行混合检索，返回 top-k 融合结果。"""
        vec = self._vector_retrieve(query)
        bm = self._bm25_retrieve(query, self._all_chunks())
        fused = self._rrf([vec, bm], self.rrf_k)
        return fused[: max(k, self.vector_top_k)]


def format_contexts(results: Iterable[RetrievedDoc]) -> str:
    """把检索结果格式化为模型提示词所用的上下文文本。"""
    blocks = []
    for i, r in enumerate(results, 1):
        page = r.doc.metadata.get("page", "?")
        blocks.append(f"[文档{i}][第{page}页]\n{r.doc.page_content}")
    return "\n\n".join(blocks)