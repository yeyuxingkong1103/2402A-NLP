# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
向量化模块：使用 sentence-transformers 加载本地 m3e-base 模型，将文本转为向量
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_PATH, EMBED_DIM

# 全局模型实例（懒加载，只加载一次）
_model = None


def get_model():
    """获取嵌入模型单例，避免重复加载"""
    global _model
    if _model is None:
        print(f"[向量化] 正在加载嵌入模型: {EMBED_MODEL_PATH}")
        _model = SentenceTransformer(EMBED_MODEL_PATH)
        print(f"[向量化] 模型加载完成，输出维度: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(texts, batch_size=32):
    """
    将一批文本转为向量
    texts: 文本列表
    返回: numpy 数组，shape=(len(texts), EMBED_DIM)
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2归一化，后续点积即余弦相似度
        convert_to_numpy=True
    )
    return embeddings


def embed_query(query):
    """将单个查询文本转为向量（用于检索时）"""
    model = get_model()
    vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return vec[0]  # 返回一维向量


if __name__ == "__main__":
    # 测试嵌入模型
    model = get_model()
    test_texts = ["武汉兴图新科电子股份有限公司", "军用领域收入"]
    vecs = embed_texts(test_texts)
    print(f"向量形状: {vecs.shape}")
    print(f"向量维度: {vecs.shape[1]}")
    # 计算两个向量的余弦相似度（归一化后点积）
    sim = np.dot(vecs[0], vecs[1])
    print(f"两文本余弦相似度: {sim:.4f}")
