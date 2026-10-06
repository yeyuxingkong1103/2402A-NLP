# -*- coding: utf-8 -*-
# 【索引存储模块 · vector_store.py】TF-IDF稠密向量 + jieba BM25 倒排索引，支持持久化
# 工单编号：人工智能NLP-RAG-Query理解优化任务

"""索引层：

- TfidfVectorStore：字符 ngram TF-IDF 矩阵，点积即余弦，百页招股书规模下毫秒级检索；
- BM25Store：jieba 中文分词 + 英文按词的 BM25 倒排，兜住数字/专名类查询；
- IndexStore：聚合两类索引与 Chunk 元数据，提供 numpy/pickle 持久化，启动单例加载。
"""
import json
import math
import os
import pickle
from collections import Counter
from typing import List, Tuple

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from chunker import Chunk


def tokenize(text: str) -> List[str]:
    """中英文混合分词：jieba 切中文，英文/数字整词保留。"""
    tokens = [t.strip() for t in jieba.lcut(text.lower()) if t.strip()]
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


class TfidfVectorStore:
    """TF-IDF 稠密向量库（已 L2 归一化，点积即余弦）。"""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        scores = self.matrix @ query_vec
        k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        ranked = sorted(((int(i), float(scores[i])) for i in idx),
                        key=lambda x: x[1], reverse=True)
        return ranked


class BM25Store:
    """BM25 稀疏检索：纯 Python 倒排表，含 IDF、文档长度归一。"""

    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5,
                 b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.tf: List[Counter] = [Counter(d) for d in docs_tokens]
        self.df: Counter = Counter()
        for doc in self.tf:
            self.df.update(doc.keys())
        self.idf = {
            word: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for word, freq in self.df.items()
        }

    def search(self, query_tokens: List[str], top_k: int) -> List[Tuple[int, float]]:
        scores = np.zeros(self.n, dtype=np.float32)
        for word in query_tokens:
            if word not in self.idf:
                continue
            idf = self.idf[word]
            for i, tf in enumerate(self.tf):
                freq = tf.get(word, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * (self.doc_len[i] / (self.avgdl or 1.0)))
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        k = min(top_k, self.n)
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        return sorted(((int(i), float(scores[i])) for i in idx if scores[i] > 0),
                      key=lambda x: x[1], reverse=True)


class IndexStore:
    """聚合向量库、BM25 与块元数据，提供构建/保存/加载/多路召回。"""

    def __init__(self, chunks: List[Chunk], vectorizer: TfidfVectorizer,
                 dense: TfidfVectorStore, sparse: BM25Store) -> None:
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.dense = dense
        self.sparse = sparse

    @classmethod
    def build(cls, chunks: List[Chunk]) -> "IndexStore":
        """用全部块文本构建 TF-IDF + BM25 双路索引。"""
        texts = [c.text for c in chunks]
        vectorizer = TfidfVectorizer(analyzer="char_wb",
                                     ngram_range=(2, 4),
                                     max_features=30000)
        matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        matrix = matrix / norms
        dense = TfidfVectorStore(matrix)
        sparse = BM25Store([tokenize(t) for t in texts])
        return cls(chunks, vectorizer, dense, sparse)

    def dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        vec = self.vectorizer.transform([query]).toarray().astype(np.float32)
        norm = np.linalg.norm(vec) or 1e-12
        qv = (vec / norm).flatten()
        return self.dense.search(qv, top_k)

    def sparse_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        return self.sparse.search(tokenize(query), top_k)

    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, "vectors.npy"), self.dense.matrix)
        with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(self.sparse, f)
        with open(os.path.join(index_dir, "tfidf.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        meta = {"embedding": "tfidf", "chunk_count": len(self.chunks)}
        with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str) -> "IndexStore":
        matrix = np.load(os.path.join(index_dir, "vectors.npy"))
        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            sparse = pickle.load(f)
        with open(os.path.join(index_dir, "tfidf.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        return cls(chunks, vectorizer, TfidfVectorStore(matrix), sparse)
