# -*- coding: utf-8 -*-
"""
进阶优化: sentence-transformers 向量模型升级
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

说明:
    - 当前 V2 使用 BM25 + TF-IDF 混合检索, 已内置轻量实现
    - 若需进一步提升准确率, 可将 TF-IDF 替换为预训练向量模型
    - 本文件演示如何用 sentence-transformers 替换 vector_retrieval_v2 中的 TF-IDF 部分
    - 需要额外安装: pip install sentence-transformers
    - 模型首次使用会自动下载 (~100MB)

使用方法:
    1. pip install sentence-transformers
    2. 修改 vector_retrieval_v2.py 中的 VectorRetriever 使用本模块的 encode()
    3. 重新构建索引
"""
import os
import sys
import logging
from typing import List

logger = logging.getLogger(__name__)

# 推荐模型 (中文效果好, 体积小)
RECOMMENDED_MODELS = [
    "BAAI/bge-small-zh",          # 30MB, 中文效果好
    "shibing624/text2vec-base-chinese",  # 40MB
    "BAAI/bge-base-zh",           # 100MB, 效果更好
]


def encode_with_stmodel(texts: List[str], model_name: str = "BAAI/bge-small-zh"):
    """
    使用 sentence-transformers 对文本列表编码

    Returns:
        numpy.ndarray 形状 (n, dim)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "请先安装 sentence-transformers:\n"
            "pip install sentence-transformers\n"
            "或继续使用内置的 TF-IDF + BM25 混合检索 (已足够达标)"
        )
    logger.info(f"加载模型: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    logger.info(f"编码完成: {embeddings.shape}")
    return embeddings


def cosine_search(query_embedding, doc_embeddings, top_k: int = 5):
    """向量余弦检索"""
    import numpy as np
    scores = np.dot(doc_embeddings, query_embedding.T).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    return top_indices, scores[top_indices]


def hybrid_with_stmodel(query: str, chunks: List[Dict],
                        model_name: str = "BAAI/bge-small-zh",
                        bm25_weight: float = 0.3, st_weight: float = 0.7,
                        top_k: int = 5) -> List[Dict]:
    """
    BM25 + sentence-transformers 混合检索 (替换 TF-IDF)

    使用方法:
        from 进阶优化向量模型 import hybrid_with_stmodel
        results = hybrid_with_stmodel("注册资本", chunks)
    """
    import numpy as np
    from vector_retrieval_v2 import BM25Retriever, _tokenize

    texts = [c["text"] for c in chunks]
    tokenized = [_tokenize(t) for t in texts]

    # BM25
    bm25 = BM25Retriever()
    bm25.fit(tokenized)
    query_tokens = _tokenize(query)
    bm25_scores = np.array(bm25.score(query_tokens))
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    # sentence-transformers
    import sentence_transformers as st
    model = st.SentenceTransformer(model_name)
    query_emb = model.encode([query], normalize_embeddings=True)
    doc_embs = model.encode(texts, normalize_embeddings=True)
    st_scores = np.dot(doc_embs, query_emb.T).flatten()

    # 融合
    final = bm25_weight * bm25_scores + st_weight * st_scores
    top_idx = final.argsort()[-top_k:][::-1]

    results = []
    for idx in top_idx:
        chunk = chunks[idx]
        results.append({
            "id": chunk["id"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": round(float(final[idx]), 4),
            "type": chunk.get("type", "text"),
            "bm25_score": round(float(bm25_scores[idx]), 4),
            "st_score": round(float(st_scores[idx]), 4),
        })
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("进阶优化模块 - sentence-transformers 向量升级")
    print(f"推荐模型: {RECOMMENDED_MODELS[0]}")
    print("安装命令: pip install sentence-transformers")
