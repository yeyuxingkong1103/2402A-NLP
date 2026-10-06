# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
嵌入模块：将文本向量化。
  - TfidfEmbedder：轻量离线（默认）
  - SentenceTransformerEmbedder：语义嵌入（bge / m3e，可选）
"""
import math
import numpy as np

try:
    import jieba
except ImportError:
    jieba = None


def tokenize(text):
    text = text.lower()
    if jieba is not None:
        return [t.strip() for t in jieba.cut(text) if t.strip()]
    chars = [c for c in text if not c.isspace()]
    return chars + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


class TfidfEmbedder:
    """TF-IDF 稀疏向量嵌入。"""

    def __init__(self):
        self.vocab = {}
        self.idf = None
        self.n = 0

    def fit(self, docs):
        self.n = len(docs)
        df = {}
        for d in docs:
            for t in set(tokenize(d)):
                df[t] = df.get(t, 0) + 1
        self.vocab = {t: i for i, t in enumerate(sorted(df.keys()))}
        self.idf = np.zeros(len(self.vocab))
        for t, i in self.vocab.items():
            self.idf[i] = math.log((1 + self.n) / (1 + df[t])) + 1.0

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        mat = np.zeros((len(texts), len(self.vocab)))
        for di, t in enumerate(texts):
            tf = {}
            for tok in tokenize(t):
                if tok in self.vocab:
                    tf[tok] = tf.get(tok, 0) + 1
            for tok, c in tf.items():
                mat[di, self.vocab[tok]] = (1 + math.log(c)) * self.idf[self.vocab[tok]]
            norm = np.linalg.norm(mat[di])
            if norm > 0:
                mat[di] /= norm
        return mat


class SentenceTransformerEmbedder:
    """语义嵌入（bge / m3e），需 sentence-transformers。"""

    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception:
            pass

    def available(self):
        return self.model is not None

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")


def get_embedder(model_name=None):
    """优先语义模型，不可用时回退 TF-IDF。"""
    if model_name:
        sem = SentenceTransformerEmbedder(model_name)
        if sem.available():
            return sem
    return TfidfEmbedder()
