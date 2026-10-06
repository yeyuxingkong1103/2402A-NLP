"""重排序模块"""
from typing import List, Dict, Any, Optional
import numpy as np

from src.config import config
from src.utils.logger import logger


class Reranker:
    """重排序模型基类"""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or config.get('rerank.model', 'BAAI/bge-rerank-base')
        self.device = device or config.get('rerank.device', 'cuda')
        self.model = None
    
    def load(self):
        """加载模型"""
        raise NotImplementedError
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """重排序"""
        raise NotImplementedError


class BGELocalReranker(Reranker):
    """BGE本地重排序器"""
    
    def load(self):
        """加载模型"""
        if self.model is None:
            try:
                from FlagEmbedding import FlagReranker
                logger.info(f"加载重排序模型: {self.model_name}")
                self.model = FlagReranker(self.model_name, use_fp16=True)
                logger.info("重排序模型加载完成")
            except ImportError:
                logger.warning("FlagEmbedding未安装，使用简单重排序")
                self.model = None
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """重排序"""
        if not documents:
            return []
        
        if self.model is None:
            self.load()
        
        if self.model is None:
            # 简单重排序：按原始距离排序
            return documents[:top_k] if top_k else documents
        
        # 提取文本
        texts = [doc['text'] for doc in documents]
        
        # 计算得分
        scores = self.model.compute_score([[query, text] for text in texts])
        
        # 按得分排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 添加得分
        reranked = []
        for doc, score in scored_docs:
            doc = doc.copy()
            doc['rerank_score'] = score
            reranked.append(doc)
        
        top_k = top_k or len(reranked)
        return reranked[:top_k]


class SimpleReranker(Reranker):
    """简单重排序器（基于中英混合分词的重叠打分）

    不依赖外部模型，作为 BGE-rerank 不可用时的兜底精排。
    """

    def load(self):
        pass

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """简单重排序"""
        if not documents:
            return []

        from src.rag.bm25 import BM25Index

        query_tokens = set(BM25Index.tokenize(query))
        if not query_tokens:
            return documents[:top_k] if top_k else documents

        for doc in documents:
            text_tokens = set(BM25Index.tokenize(doc.get('text', '')))
            overlap = len(query_tokens & text_tokens)
            # 重叠率 + 命中词个数的组合打分
            doc['rerank_score'] = overlap / max(len(query_tokens), 1) + overlap * 0.01

        documents.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        return documents[:top_k] if top_k else documents


def get_reranker(method: str = "auto", **kwargs) -> Reranker:
    """获取重排序器。

    method: auto / bge / simple
      - auto: 有 FlagEmbedding 且能加载 bge-rerank 则用 bge，否则退回 simple；
      - bge:   强制 BGE（加载失败时内部降级为按距离排序）；
      - simple: 本地关键词精排。
    """
    if method == "auto":
        try:
            import importlib.util
            if importlib.util.find_spec("FlagEmbedding") is not None:
                return BGELocalReranker(**kwargs)
        except Exception:
            pass
        return SimpleReranker(**kwargs)

    rerankers = {
        'bge': BGELocalReranker,
        'simple': SimpleReranker,
    }
    return rerankers.get(method, SimpleReranker)(**kwargs)
