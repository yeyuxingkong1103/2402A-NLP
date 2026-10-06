# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/text_retriever_v3.py —— 工单三多文档文本检索器

职责：
  1. 在 Milvus rag_chunks collection 中向量检索文本 chunk
  2. 支持按 doc_id 过滤（多文档隔离）
  3. BM25 关键词召回（Milvus LIKE 兜底）
  4. RRF 融合向量 + 关键词结果
  5. 兼容工单二 HybridRetriever 的 retrieve 接口（无 doc_id 参数时降级）

与工单二 HybridRetriever 的区别：
  - 支持 doc_id 过滤（多文档）
  - 不依赖父块映射（简化，直接用 chunk 内容）
  - 复用 bge-m3 + bge-reranker
"""
import time
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from src.embedding import get_embedder
from src.reranker import Reranker
from src.vector_store import VectorStore


class TextRetrieverV3:
    """工单三：多文档文本检索器"""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder=None,
        reranker: Optional[Reranker] = None,
        use_rerank: bool = True,
        top_k: int = 8,
        raw_multiplier: int = 3,
    ):
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or get_embedder()
        self.top_k = top_k
        self.raw_multiplier = raw_multiplier
        self._reranker = reranker
        self.use_rerank = use_rerank
        if use_rerank and self._reranker is None:
            try:
                self._reranker = Reranker()
            except Exception as e:
                logger.warning(f"[text_retriever_v3] Reranker 初始化失败: {e}")
                self._reranker = None
                self.use_rerank = False

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """工单三：文本检索主入口

        Args:
            query: 查询文本
            top_k: 返回条数
            doc_id: 按文档过滤（招股说明书1 / 招股说明书2）

        Returns:
            [{doc_id, chunk_id, content, page, score, source:"text", ...}, ...]
        """
        t0 = time.time()
        k = top_k or self.top_k

        # 1) 向量检索
        qvec = self.embedder.encode([query], show_progress_bar=False)[0]
        raw_k = k * self.raw_multiplier
        vec_hits = self.vector_store.search(
            qvec, top_k=raw_k, doc_id=doc_id,
        )
        for h in vec_hits:
            h["source"] = "text"
            h["content"] = h.get("content") or h.get("text") or ""
            h["search_path"] = "vector"

        # 2) BM25-like 关键词召回（Milvus LIKE）
        kw_hits = self._search_by_keyword(query, doc_id=doc_id, top_k=raw_k)

        # 3) 融合
        fused = self._merge_results(vec_hits, kw_hits)

        # 4) 重排序
        if self.use_rerank and self._reranker and fused:
            try:
                fused = self._reranker.rerank(
                    query, fused, top_k=k, content_key="content",
                )
            except Exception as e:
                logger.warning(f"[text_retriever_v3] Rerank 失败: {e}")
                fused = sorted(fused, key=lambda x: x.get("score", 0),
                               reverse=True)[:k]
        else:
            fused = sorted(fused, key=lambda x: x.get("score", 0),
                           reverse=True)[:k]

        elapsed = (time.time() - t0) * 1000
        logger.info(f"[text_retriever_v3] retrieve: "
                    f"vec={len(vec_hits)} kw={len(kw_hits)} "
                    f"fused={len(fused)} {elapsed:.0f}ms")
        return fused

    def _search_by_keyword(
        self, query: str, doc_id: Optional[str] = None, top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """工单三：BM25-like 关键词召回（Milvus LIKE）"""
        import jieba
        STOPWORDS = set(
            "的 了 和 是 在 有 我 他 她 它 这 那 就 不 也 都 一 个 上 下 "
            "中 到 说 去 你 好 为 什么 怎么 哪 谁 几 多 少 吗 呢 吧 啊 哦 嗯 "
            "会 能 可以 应该 需要 希望 想 看 听 问 答 知道 了解 多少 是".split()
        )
        tokens = [t for t in jieba.lcut(query)
                   if len(t.strip()) >= 2 and t not in STOPWORDS]
        if not tokens:
            return []

        like_exprs = [f'content like "%{t}%"' for t in tokens[:5]]
        filter_parts = " or ".join(like_exprs)
        if doc_id:
            filter_parts = f'(doc_id == "{doc_id}") and ({filter_parts})'
        try:
            results = self.vector_store.client.query(
                collection_name=self.vector_store.collection,
                filter=filter_parts,
                output_fields=["doc_id", "chunk_id", "content", "page",
                               "metadata"],
                limit=top_k,
            )
            out = []
            for r in (results or []):
                content = r.get("content", "")
                kw_hits = sum(1 for t in tokens if t in content)
                out.append({
                    "doc_id": r.get("doc_id", ""),
                    "chunk_id": r.get("chunk_id", ""),
                    "content": content,
                    "page": r.get("page", 0),
                    "metadata": r.get("metadata", {}),
                    "score": 0.5 + kw_hits * 0.1,
                    "kw_hits": kw_hits,
                    "source": "text",
                    "search_path": "keyword",
                })
            out.sort(key=lambda x: x.get("kw_hits", 0), reverse=True)
            return out[:top_k]
        except Exception as e:
            logger.warning(f"[text_retriever_v3] 关键词查询失败: {e}")
            return []

    @staticmethod
    def _merge_results(
        vec_hits: List[Dict], kw_hits: List[Dict],
    ) -> List[Dict]:
        """工单三：合并向量 + 关键词结果（保向量分数 + 关键词保底）"""
        seen: Dict[str, Dict] = {}
        for h in vec_hits:
            key = f"{h.get('doc_id','')}|{h.get('chunk_id','')}"
            h["search_path"] = "vector"
            seen[key] = h
        for h in kw_hits:
            key = f"{h.get('doc_id','')}|{h.get('chunk_id','')}"
            if key in seen:
                seen[key]["search_path"] = "vector+keyword"
            else:
                h["search_path"] = "keyword"
                seen[key] = h
        return sorted(seen.values(), key=lambda x: x.get("score", 0),
                       reverse=True)
