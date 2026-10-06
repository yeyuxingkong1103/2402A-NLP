# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
向量化模块：加载本地 m3e-base 中文嵌入模型，将文本转为 L2 归一化向量
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_PATH

_model = None


def get_model():
    """获取嵌入模型单例（懒加载，只加载一次）"""
    global _model
    if _model is None:
        print(f"[向量化] 正在加载嵌入模型: {EMBED_MODEL_PATH}")
        _model = SentenceTransformer(EMBED_MODEL_PATH)
        print(f"[向量化] 模型加载完成，维度: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(texts, batch_size=32):
    """批量文本向量化，返回 shape=(n, dim) 的归一化向量矩阵"""
    model = get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def embed_query(query):
    """单条查询向量化，返回一维归一化向量"""
    model = get_model()
    return model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]


if __name__ == "__main__":
    vecs = embed_texts(["武汉兴图新科电子股份有限公司", "军用领域收入"])
    print(f"向量形状: {vecs.shape}")
    print(f"余弦相似度: {float(np.dot(vecs[0], vecs[1])):.4f}")
