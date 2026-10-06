# -*- coding: utf-8 -*-
# 【索引存储模块 · vector_store.py】TF-IDF稀疏向量库 + jieba BM25倒排，含BGE接口与无缓存自动降级
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

"""索引层（对应设计文档第 5 章）：

- TfidfEmbedder：sklearn 字符级 2~4gram TF-IDF，保持稀疏矩阵，
  对中文专名、数字串区分度好，数千块内存占用仅数十 MB；
- TfidfStore：稀疏矩阵乘法做余弦 Top-K，毫秒级；
- BM25Store：jieba 分词 + postings 倒排表，只累加命中文档；
- BGEEmbedder：可选语义升级接口，本地无模型缓存时由工厂自动降级 TF-IDF，
  绝不联网下载；
- IndexStore：聚合索引与 Chunk 元数据，提供构建/保存/加载。
"""
import glob
import json
import math
import os
import pickle
from collections import Counter
from typing import List, Optional, Tuple

import jieba
import numpy as np
import scipy.sparse as sp

from chunker import Chunk
from config import CONFIG


def tokenize(text: str) -> List[str]:
    """中英文混合分词：jieba 切中文，英文/数字整词保留，过滤纯标点。

    :param text: 原始文本
    :return: 词项列表
    """
    tokens = [t.strip() for t in jieba.lcut(text.lower()) if t.strip()]
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


# ---------------------------------------------------------------------------
# 嵌入器
# ---------------------------------------------------------------------------
class TfidfEmbedder:
    """离线嵌入：字符级 2~4gram TF-IDF（稀疏、L2 归一、sublinear）。"""

    def __init__(self) -> None:
        """初始化向量化器（fit 后维度确定）。"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=CONFIG.tfidf_max_features,
            sublinear_tf=True, norm="l2")
        self.dim = 0
        self.kind = "tfidf"

    def fit(self, texts: List[str]) -> None:
        """拟合 IDF 并记录维度。

        :param texts: 文档块文本
        """
        matrix = self.vectorizer.fit_transform(texts)
        self.dim = matrix.shape[1]

    def encode_documents(self, texts: List[str]) -> sp.csr_matrix:
        """编码文档块为稀疏 L2 归一矩阵。

        :param texts: 文档文本
        :return: csr 矩阵
        """
        return self.vectorizer.transform(texts).tocsr()

    def encode_queries(self, texts: List[str]) -> sp.csr_matrix:
        """编码查询（TF-IDF 无需指令前缀）。

        :param texts: 查询文本
        :return: csr 矩阵
        """
        return self.vectorizer.transform(texts).tocsr()


class BGEEmbedder:
    """可选 BGE 双语稠密嵌入接口；仅在本地已缓存模型时可用。"""

    def __init__(self) -> None:
        """延迟加载本地 BGE 模型。"""
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(CONFIG.embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.kind = "bge"

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """编码文档块。"""
        return self.model.encode(
            texts, normalize_embeddings=True,
            convert_to_numpy=True, batch_size=64).astype(np.float32)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """编码查询。"""
        return self.model.encode(
            texts, normalize_embeddings=True,
            convert_to_numpy=True, batch_size=64).astype(np.float32)


def model_available_locally(model_name: str) -> bool:
    """检测 HF 模型是否已在本地缓存（避免离线环境联网等待）。

    :param model_name: HF 模型 ID
    :return: 本地可用返回 True
    """
    if os.path.exists(model_name):
        return True
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    repo = "models--" + model_name.replace("/", "--")
    return bool(glob.glob(os.path.join(hub, repo, "snapshots", "*")))


def create_embedder(doc_texts: Optional[List[str]] = None):
    """嵌入器工厂：本地有 BGE 缓存则升级，否则离线 TF-IDF（不联网下载）。

    :param doc_texts: 文档块文本（TF-IDF 拟合用）
    :return: 已就绪嵌入器
    """
    if CONFIG.fallback_embedding != "tfidf" and model_available_locally(
            CONFIG.embedding_model):
        try:
            print("[嵌入] 检测到本地 BGE 缓存，启用语义向量。")
            return BGEEmbedder()
        except Exception as exc:
            print(f"[嵌入] BGE 加载失败({exc})，降级 TF-IDF。")
    print("[嵌入] 未检测到本地模型缓存，启用离线 TF-IDF 字符向量（不联网下载）。")
    embedder = TfidfEmbedder()
    if doc_texts and any(t.strip() for t in doc_texts):
        embedder.fit(doc_texts)
    return embedder


# ---------------------------------------------------------------------------
# 向量库与 BM25
# ---------------------------------------------------------------------------
class TfidfStore:
    """稀疏 TF-IDF 向量库：矩阵乘查询向量即余弦相似度。"""

    def __init__(self, matrix: sp.spmatrix) -> None:
        """保存文档稀疏矩阵。

        :param matrix: shape=(n, dim) 的 L2 归一稀疏矩阵
        """
        self.matrix = matrix.tocsr()

    def search(self, q_vec: sp.spmatrix, top_k: int,
               subset: Optional[np.ndarray] = None) -> List[Tuple[int, float]]:
        """稀疏余弦检索 Top-K，可限定在公司路由后的子集下标内。

        :param q_vec: 查询稀疏向量 (1, dim)
        :param top_k: 返回条数
        :param subset: 允许返回的块下标数组（公司路由）
        :return: [(块下标, 分数)] 降序
        """
        scores = (self.matrix @ q_vec.T).toarray().ravel()
        if subset is not None:
            mask = np.full(scores.shape[0], -np.inf, dtype=np.float32)
            mask[subset] = 0.0
            scores = scores + mask
        k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        ranked = sorted(((int(i), float(scores[i])) for i in idx
                         if scores[i] > 0),
                        key=lambda x: x[1], reverse=True)
        return ranked


class BM25Store:
    """BM25 稀疏检索：jieba 分词，postings 倒排 + 文档长度归一。"""

    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5,
                 b: float = 0.75) -> None:
        """构建倒排表与统计量。

        :param docs_tokens: 每块分词列表
        :param k1: 词频饱和参数
        :param b: 长度归一参数
        """
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens],
                                dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.tf: List[Counter] = [Counter(d) for d in docs_tokens]
        df: Counter = Counter()
        for doc in self.tf:
            df.update(doc.keys())
        self.idf = {
            word: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for word, freq in df.items()}
        # postings：词 → [(文档下标, 词频)]，检索时只遍历命中文档
        self.postings: dict = {}
        for i, doc in enumerate(self.tf):
            for word, freq in doc.items():
                self.postings.setdefault(word, []).append((i, freq))

    def search(self, query_tokens: List[str], top_k: int,
               subset: Optional[set] = None) -> List[Tuple[int, float]]:
        """计算 BM25 得分并取 Top-K。

        :param query_tokens: 查询分词
        :param top_k: 返回条数
        :param subset: 公司路由允许的块下标集合
        :return: [(块下标, 分数)] 降序
        """
        scores: dict = {}
        for word in query_tokens:
            idf = self.idf.get(word)
            if idf is None:
                continue
            for doc_i, freq in self.postings.get(word, []):
                if subset is not None and doc_i not in subset:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * (
                        self.doc_len[doc_i] / (self.avgdl or 1.0)))
                scores[doc_i] = scores.get(doc_i, 0.0) + \
                    idf * (freq * (self.k1 + 1)) / denom
        if not scores:
            return []
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(i, float(s)) for i, s in items[:top_k]]


class IndexStore:
    """聚合向量库、BM25 与块元数据，提供构建/保存/加载/多路召回。"""

    def __init__(self, chunks: List[Chunk], embedder, dense: TfidfStore,
                 sparse: BM25Store) -> None:
        """保存索引四要素。

        :param chunks: 检索块列表（下标对齐向量行/倒排文档）
        :param embedder: 已拟合嵌入器
        :param dense: TF-IDF 向量库
        :param sparse: BM25 倒排
        """
        self.chunks = chunks
        self.embedder = embedder
        self.dense = dense
        self.sparse = sparse

    @classmethod
    def build(cls, chunks: List[Chunk], embedder) -> "IndexStore":
        """用全部块文本构建双路索引。

        :param chunks: 检索块列表
        :param embedder: TF-IDF 嵌入器（先拟合）
        :return: 可检索 IndexStore
        """
        texts = [c.text for c in chunks]
        if hasattr(embedder, "fit"):
            embedder.fit(texts)
        matrix = embedder.encode_documents(texts)
        dense = TfidfStore(matrix)
        sparse = BM25Store([tokenize(t) for t in texts])
        return cls(chunks, embedder, dense, sparse)

    def dense_search(self, query: str, top_k: int,
                     subset: Optional[np.ndarray] = None):
        """TF-IDF 召回。"""
        qv = self.embedder.encode_queries([query])
        return self.dense.search(qv, top_k, subset)

    def sparse_search(self, query: str, top_k: int,
                      subset: Optional[set] = None):
        """BM25 关键词召回。"""
        return self.sparse.search(tokenize(query), top_k, subset)

    def save(self, index_dir: str) -> None:
        """持久化索引（稀疏向量、块、BM25、TF-IDF 向量化器、元信息）。

        :param index_dir: 索引目录
        """
        os.makedirs(index_dir, exist_ok=True)
        sp.save_npz(os.path.join(index_dir, "vectors.npz"), self.dense.matrix)
        with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(self.sparse, f)
        if getattr(self.embedder, "kind", "") == "tfidf":
            with open(os.path.join(index_dir, "tfidf.pkl"), "wb") as f:
                pickle.dump(self.embedder.vectorizer, f)
        companies = Counter(c.company for c in self.chunks)
        meta = {"embedding": getattr(self.embedder, "kind", "tfidf"),
                "chunk_count": len(self.chunks),
                "table_chunks": sum(
                    1 for c in self.chunks if c.chunk_type == "table"),
                "table_row_chunks": sum(
                    1 for c in self.chunks if c.chunk_type == "table_row"),
                "companies": dict(companies)}
        with open(os.path.join(index_dir, "meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str, embedder=None) -> "IndexStore":
        """从目录加载索引。

        :param index_dir: 索引目录
        :param embedder: 可选外部嵌入器；TF-IDF 索引会自行恢复向量化器
        """
        matrix = sp.load_npz(os.path.join(index_dir, "vectors.npz"))
        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            sparse = pickle.load(f)
        tfidf_path = os.path.join(index_dir, "tfidf.pkl")
        if os.path.exists(tfidf_path):
            embedder = TfidfEmbedder()
            with open(tfidf_path, "rb") as f:
                embedder.vectorizer = pickle.load(f)
            embedder.dim = matrix.shape[1]
        return cls(chunks, embedder, TfidfStore(matrix), sparse)
