# -*- coding: utf-8 -*-
# 【索引存储模块 · vector_store.py】稠密向量库（numpy余弦）+ BM25倒排索引，支持持久化
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

"""索引层：

- DenseVectorStore：归一化向量矩阵 + 点积即余弦，百页招股书规模下毫秒级检索；
- BM25Store：jieba 中文分词 + 英文按词的 BM25 倒排，兜住数字/专名类查询；
- IndexStore：聚合两类索引与 Chunk 元数据，提供 numpy 持久化，启动单例加载。
"""
import json
import math
import os
import pickle
from collections import Counter
from typing import List, Tuple

import jieba
import numpy as np

from chunker import Chunk
from embeddings import BaseEmbedder, TfidfEmbedder


def tokenize(text: str) -> List[str]:
    """中英文混合分词：jieba 切中文，英文/数字整词保留，过滤空白与单标点。

    :param text: 原始文本
    :return: 词项列表
    """
    tokens = [t.strip() for t in jieba.lcut(text.lower()) if t.strip()]
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


class DenseVectorStore:
    """内存稠密向量库：矩阵化余弦相似度 Top-K 检索。"""

    def __init__(self, matrix: np.ndarray) -> None:
        """保存已归一化的文档向量矩阵。

        :param matrix: shape=(n, dim) 的 L2 归一化矩阵
        """
        self.matrix = matrix

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """向量检索：点积即余弦相似度。

        :param query_vec: shape=(dim,) 归一化查询向量
        :param top_k: 返回前 K 条
        :return: [(chunk_index, score), ...] 按分数降序
        """
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
        """构建词项倒排表与文档频率统计。

        :param docs_tokens: 每个文档块的分词列表
        :param k1: 词频饱和参数
        :param b: 长度归一参数
        """
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
        """对查询计算所有文档的 BM25 得分并取 Top-K。

        :param query_tokens: 查询分词
        :param top_k: 返回条数
        :return: [(chunk_index, score), ...] 降序
        """
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

    def __init__(self, chunks: List[Chunk], embedder: BaseEmbedder,
                 dense: DenseVectorStore, sparse: BM25Store) -> None:
        """保存索引四要素。

        :param chunks: 检索块列表（按下标对齐向量行/倒排文档）
        :param embedder: 已拟合的嵌入器（TF-IDF 需随索引持久化）
        :param dense: 稠密向量库
        :param sparse: BM25 倒排
        """
        self.chunks = chunks
        self.embedder = embedder
        self.dense = dense
        self.sparse = sparse

    @classmethod
    def build(cls, chunks: List[Chunk], embedder: BaseEmbedder) -> "IndexStore":
        """用全部块文本构建双路索引。

        :param chunks: 检索块列表
        :param embedder: 嵌入器（若为 TF-IDF 会先拟合）
        :return: 可检索的 IndexStore
        """
        texts = [c.text for c in chunks]
        if isinstance(embedder, TfidfEmbedder):
            embedder.fit(texts)
        matrix = embedder.encode_documents(texts)
        dense = DenseVectorStore(matrix)
        sparse = BM25Store([tokenize(t) for t in texts])
        return cls(chunks, embedder, dense, sparse)

    def dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """稠密召回。

        :param query: 用户查询
        :param top_k: 返回条数
        """
        qv = self.embedder.encode_queries([query])[0]
        return self.dense.search(qv, top_k)

    def sparse_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """稀疏关键词召回。

        :param query: 用户查询
        :param top_k: 返回条数
        """
        return self.sparse.search(tokenize(query), top_k)

    def save(self, index_dir: str) -> None:
        """持久化索引到目录（向量矩阵 + 块元数据 + 嵌入器）。

        :param index_dir: 索引目录
        """
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, "vectors.npy"), self.dense.matrix)
        with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(self.sparse, f)
        if isinstance(self.embedder, TfidfEmbedder):
            with open(os.path.join(index_dir, "tfidf.pkl"), "wb") as f:
                pickle.dump(self.embedder.vectorizer, f)
        meta = {"embedding": "tfidf" if isinstance(self.embedder, TfidfEmbedder)
                else "bge", "chunk_count": len(self.chunks)}
        with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str, embedder: BaseEmbedder) -> "IndexStore":
        """从目录加载索引。

        :param index_dir: 索引目录
        :param embedder: 查询编码使用的嵌入器（TF-IDF 时由索引文件恢复）
        """
        matrix = np.load(os.path.join(index_dir, "vectors.npy"))
        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            sparse = pickle.load(f)
        tfidf_path = os.path.join(index_dir, "tfidf.pkl")
        if os.path.exists(tfidf_path):
            with open(tfidf_path, "rb") as f:
                embedder = TfidfEmbedder()
                embedder.vectorizer = pickle.load(f)
                embedder.dim = matrix.shape[1]
        return cls(chunks, embedder, DenseVectorStore(matrix), sparse)
