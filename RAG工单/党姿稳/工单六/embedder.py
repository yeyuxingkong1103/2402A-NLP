# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
向量化模块：支持多种嵌入模型（m3e / bge / 其他 sentence-transformers 模型）。
默认使用本地 m3e-base，输出 L2 归一化向量；通过 model_name 参数可切换模型。
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_PATH, EMBED_MODELS

_cache = {}


def get_model(model_name=None):
    """获取嵌入模型单例（按模型名缓存）"""
    key = model_name or EMBED_MODEL_PATH
    if key not in _cache:
        path = EMBED_MODELS.get(model_name, key) if model_name else EMBED_MODEL_PATH
        print(f"[向量化] 正在加载嵌入模型: {path}")
        _cache[key] = SentenceTransformer(path)
        print(f"[向量化] 加载完成，维度: {_cache[key].get_sentence_embedding_dimension()}")
    return _cache[key]


def embed_texts(texts, batch_size=32, model_name=None):
    """批量文本向量化，返回 shape=(n, dim) 的归一化向量矩阵"""
    model = get_model(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def embed_query(query, model_name=None):
    """单条查询向量化，返回一维归一化向量"""
    model = get_model(model_name)
    return model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]


if __name__ == "__main__":
    vecs = embed_texts(["武汉兴图新科电子股份有限公司", "军用领域收入"])
    print(f"向量形状: {vecs.shape}")
    print(f"余弦相似度: {float(np.dot(vecs[0], vecs[1])):.4f}")
