# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：向量检索模块（优化版）
功能：使用 bge-base 中文模型 + FAISS 检索
"""

import numpy as np
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer


class VectorRetriever:
    """向量检索器（优化版）"""

    def __init__(self, model_name: str = "BAAI/bge-base-zh-v1.5"):
        print(f"正在加载嵌入模型：{model_name}")
        self.encoder = SentenceTransformer(
            model_name,
            model_kwargs={"use_safetensors": True}
        )
        try:
            self.dimension = self.encoder.get_embedding_dimension()
        except AttributeError:
            self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        print(f"✅ 模型加载完成，向量维度：{self.dimension}")

    def build_index(self, chunks: List[Dict[str, Any]]):
        """构建向量索引"""
        self.chunks = chunks
        texts = [c["content"] for c in chunks]

        print(f"正在向量化 {len(texts)} 个文本块...")
        embeddings = self.encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print(f"✅ 索引构建完成：{len(chunks)} 块，维度 {self.dimension}")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """检索 Top-K 相关块"""
        if self.index is None:
            raise ValueError("请先调用 build_index()")

        qvec = self.encoder.encode(
            [query],
            normalize_embeddings=True
        )
        qvec = np.array(qvec).astype("float32")

        scores, indices = self.index.search(qvec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                item = dict(self.chunks[idx])
                item["score"] = float(score)
                results.append(item)
        return results
