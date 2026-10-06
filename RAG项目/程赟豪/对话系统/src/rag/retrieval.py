"""检索模块

在线检索链路（对应文档.txt 的「在线部分」）：
  问题向量化 -> 检索（混合检索 / 多路召回：向量 + BM25）-> 相似度过滤
  -> 重排序（精排）-> 送入生成器。
"""
from typing import List, Dict, Any, Optional
import numpy as np

from src.config import config
from src.utils.logger import logger
from src.rag.vector_store import VectorStore
from src.rag.embedding import EmbeddingModel
from src.rag.rerank import get_reranker, Reranker
from src.rag.bm25 import BM25Index
from src.rag.query_rewriter import QueryRewriter


class Retriever:
    """检索器（默认即混合检索：向量召回 + BM25 召回 + 重排序）"""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        reranker: Optional[Reranker] = None,
        bm25_index: Optional[BM25Index] = None,
        query_rewriter: Optional[QueryRewriter] = None
    ):
        self.vector_store = vector_store or VectorStore()
        self.embedding_model = embedding_model or EmbeddingModel()
        # 默认启用精排：无 BGE 模型时退回本地关键词精排
        self.reranker = reranker if reranker is not None else get_reranker('auto')
        self._bm25 = bm25_index
        self._bm25_count = -1
        self.query_rewriter = query_rewriter
        self.top_k = config.get('retrieval.top_k', 5)
        self.similarity_threshold = config.get('retrieval.similarity_threshold', 0.5)
        self.use_bm25 = config.get('retrieval.use_bm25', True)
        self.use_rewrite = config.get('retrieval.query_rewrite', True)

    # ---- BM25 稀疏路（多路召回之一） ----
    def _ensure_bm25(self) -> BM25Index:
        """按需构建 BM25 索引；知识库条数变化时自动重建。"""
        try:
            count = self.vector_store.get_count()
        except Exception as e:
            logger.warning(f"获取知识库数量失败，跳过 BM25 召回: {e}")
            return BM25Index()

        if self._bm25 is None or self._bm25_count != count:
            self._bm25 = BM25Index()
            self._bm25.build(self.vector_store.get_all_chunks())
            self._bm25_count = count
        return self._bm25

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_rerank: bool = True,
        filter_source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        混合检索 + 多路召回 + 重排序。

        Returns:
            [{chunk_id, text, source, page_number, summary, distance, final_score, ...}]
        """
        top_k = top_k or self.top_k

        # 1. 问题向量化
        logger.info(f"检索: {query[:50]}...")
        query_vector = self.embedding_model.encode(query)

        # 2. 向量召回（Milvus）
        expr = f'source == "{filter_source}"' if filter_source else None
        dense_results = self.vector_store.search(query_vector, top_k * 3, expr)

        # 3. 相似度阈值过滤（余弦相似度得分的过滤）
        dense_results = [
            r for r in dense_results
            if r.get('distance', 0) >= self.similarity_threshold
        ]

        # 4. BM25 稀疏召回（多路召回）
        sparse_results = []
        if self.use_bm25:
            sparse_results = self._ensure_bm25().search(query, top_k * 3)
            if filter_source:
                sparse_results = [
                    r for r in sparse_results if r.get('source') == filter_source
                ]

        # 5. 多路结果融合（RRF）
        merged = self._rrf_fusion(dense_results, sparse_results, top_k * 2)

        if not merged:
            logger.warning("未检索到相关文档")
            return []

        # 6. 重排序（精排）
        if use_rerank and self.reranker and len(merged) > 1:
            merged = self.reranker.rerank(query, merged, top_k)
        else:
            merged = merged[:top_k]

        logger.info(f"检索到 {len(merged)} 条结果")
        return merged

    def retrieve_with_context(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """带上下文的检索：先做 Query 改写（共指消解），再混合检索。"""
        if conversation_history and self.use_rewrite:
            if self.query_rewriter is not None:
                query = self.query_rewriter.rewrite(query, conversation_history)
            else:
                query = self._build_context_query(query, conversation_history)
        return self.retrieve(query, **kwargs)

    @staticmethod
    def _rrf_fusion(
        dense: List[Dict[str, Any]],
        sparse: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion 多路结果融合（同一 chunk 出现在多路时得分累加）。"""
        scored: Dict[str, Dict[str, Any]] = {}

        def add(results: List[Dict[str, Any]]):
            for rank, doc in enumerate(results):
                key = doc.get('chunk_id') or str(doc.get('id'))
                if not key:
                    continue
                entry = scored.get(key)
                if entry is None:
                    entry = {'doc': dict(doc), 'score': 0.0}
                    scored[key] = entry
                entry['score'] += 1.0 / (k + rank + 1)

        add(dense)
        add(sparse)

        ranked = sorted(scored.values(), key=lambda e: e['score'], reverse=True)
        for e in ranked:
            e['doc']['final_score'] = e['score']
        return [e['doc'] for e in ranked]

    def _build_context_query(
        self,
        query: str,
        history: List[Dict[str, Any]]
    ) -> str:
        """无改写器时的兜底：拼接最近几轮对话作为上下文查询。"""
        recent = history[-3:]
        parts = [
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in recent if m.get('content')
        ]
        context_text = " ".join(parts)
        return f"{context_text} {query}" if context_text else query


class HybridRetriever(Retriever):
    """混合检索器（兼容旧名，行为与 Retriever 一致）"""
    pass
