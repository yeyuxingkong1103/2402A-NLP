# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
src/image_parser/image_retriever.py —— 工单四图像检索器（新增文件）

流程：query → bge-m3 文本向量 → Milvus rag_images 检索 → 关键词加权重排
     （短 caption 场景 BM25-like 精确匹配补偿，沿用项目 Lessons）。
支持 doc_ids 过滤；返回 path/caption/ocr_text/vqa_text/image_id/page/score。
"""
from typing import Any, Dict, List, Optional

from loguru import logger

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"
KEYWORD_BOOST = 0.08          # 工单四：query 词命中 caption/ocr/vqa 每词加分


class ImageRetriever:
    """工单四：图像检索器（文本通道为主，CLIP 通道可选）"""

    def __init__(self, store: Optional[Any] = None,
                 embedder: Optional[Any] = None):
        self._store = store
        self._embedder = embedder

    def _get_store(self):
        if self._store is None:
            from src.image_parser.image_store import ImageStore
            self._store = ImageStore()
            self._store.ensure_collection()
        return self._store

    def _get_embedder(self):
        if self._embedder is None:
            from src.image_parser.image_embedding import ImageTextEmbedder
            self._embedder = ImageTextEmbedder()
        return self._embedder

    # ------------------------------------------------------------------
    @staticmethod
    def _keyword_boost(query: str, rec: Dict[str, Any]) -> int:
        """工单四：query 中 ≥2 字词在图像描述文本中的命中计数（BM25-like 简化）"""
        text = f"{rec.get('caption', '')} {rec.get('ocr_text', '')} {rec.get('vqa_text', '')}"
        terms = [t for t in query.strip().split() if len(t) >= 2] or \
                [query[i:i + 2] for i in range(0, len(query) - 1, 2)]
        return sum(1 for t in terms if t and t in text)

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5,
                 doc_ids: Optional[List[str]] = None,
                 use_clip: bool = False) -> List[Dict[str, Any]]:
        """工单四：图像检索主入口（向量召回 + 关键词加权重排）"""
        store, emb = self._get_store(), self._get_embedder()
        qvec = emb.embed_text(query)
        hits = store.search_text(qvec, top_k=top_k * 3, doc_ids=doc_ids,
                                 use_clip=use_clip)   # 工单四：超采样后重排
        for h in hits:
            h["kw_hits"] = self._keyword_boost(query, h)
            h["final_score"] = round(h["score"] + KEYWORD_BOOST * h["kw_hits"], 4)
        hits.sort(key=lambda x: x["final_score"], reverse=True)
        logger.info(f"[image_retriever] '{query[:30]}' → {len(hits[:top_k])} hits")
        return hits[:top_k]

    def close(self) -> None:
        """工单四：释放 Milvus 连接"""
        if self._store is not None and getattr(self._store, "client", None):
            self._store.client.close()
            self._store = None
