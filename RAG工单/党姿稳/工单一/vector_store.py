# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
向量存储与检索模块：用 numpy 实现向量存储和余弦相似度检索
"""
import numpy as np
import os
from config import VECTORS_FILE, TOP_K
from embedder import embed_texts, embed_query


class VectorStore:
    """
    简易向量数据库：
    - 存储所有文本块的向量
    - 通过余弦相似度检索最相关的文本块
    """

    def __init__(self, chunks, use_cache=True):
        self.chunks = chunks              # 文本块列表
        self.vectors = None               # 向量矩阵，shape=(n_chunks, dim)
        self._build_or_load(use_cache)

    def _build_or_load(self, use_cache=True):
        """构建向量索引（若已有缓存则直接加载）"""
        if use_cache and os.path.exists(VECTORS_FILE):
            print(f"[向量库] 从缓存加载向量: {VECTORS_FILE}")
            self.vectors = np.load(VECTORS_FILE)
            print(f"[向量库] 向量矩阵形状: {self.vectors.shape}")
        else:
            print(f"[向量库] 正在为 {len(self.chunks)} 个文本块生成向量...")
            self.vectors = embed_texts(self.chunks)
            if use_cache:
                np.save(VECTORS_FILE, self.vectors)
                print(f"[向量库] 向量已保存到: {VECTORS_FILE}")

    def search(self, query, top_k=TOP_K):
        """
        检索与查询最相关的 top_k 个文本块
        返回: [(chunk_text, score), ...] 按相似度降序
        """
        query_vec = embed_query(query)
        # 点积 = 余弦相似度（因为向量已L2归一化）
        scores = np.dot(self.vectors, query_vec)
        # 取 top_k 个最大值的索引
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))
        return results


if __name__ == "__main__":
    from pdf_parser import build_chunks, load_chunks
    import os
    from config import CHUNKS_FILE

    # 加载或构建文本块
    if os.path.exists(CHUNKS_FILE):
        chunks = load_chunks()
    else:
        chunks = build_chunks()

    # 构建向量库
    store = VectorStore(chunks)

    # 测试检索
    test_query = "武汉兴图新科电子股份有限公司注册资本是多少？"
    results = store.search(test_query, top_k=3)
    print(f"\n查询: {test_query}")
    print("检索结果:")
    for i, (chunk, score) in enumerate(results):
        print(f"\n--- Top {i+1} (相似度: {score:.4f}) ---")
        print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
