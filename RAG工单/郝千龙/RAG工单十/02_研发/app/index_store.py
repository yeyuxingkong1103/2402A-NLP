# -*- coding: utf-8 -*-
# 【索引存储模块 · index_store.py】TF-IDF稀疏向量库 + BM25倒排 + 卷持久化与来源清单校验
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 说明：面向容器内存限制（默认 1GB）将 TF-IDF 矩阵以 scipy 稀疏 CSR 保存，
#       索引目录即 Docker 卷挂载点；通过 meta.json 来源清单实现“换容器不重建”。

"""索引层：

- tokenize：jieba 中文分词 + 英文/数字整词保留；
- TfidfSparseStore：scikit-learn 字符 2~4 gram TF-IDF，CSR 稀疏矩阵 + L2 归一，
  点积即余弦相似度，百页招股书规模下内存占用仅稠密方案的十分之一；
- BM25Store：纯 Python 倒排表，含 IDF 与文档长度归一，兜住数字/专名类查询；
- IndexStore：聚合两类索引与 Chunk 元数据，提供构建/保存/加载，并比对 PDF
  来源清单（文件名、大小、修改时间）与运行时版本决定是否复用卷内索引。
"""
import json
import logging
import math
import os
import pickle
import platform
import tempfile
from collections import Counter
from datetime import datetime
from typing import List, Tuple

import jieba
import numpy as np
import scipy
import sklearn
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from chunker import Chunk
from config import CONFIG

logger = logging.getLogger(__name__)

# 显式指定 jieba 词典缓存到系统临时目录：
# 容器内以非 root 用户运行，默认缓存路径在某些基础镜像下不可写，
# 统一定位到 /tmp（Windows 本机直跑时定位到 %TEMP%），避免启动期告警
_JIEBA_CACHE = os.path.join(tempfile.gettempdir(), "finqa_jieba.cache")
try:
    jieba.dt.cache_file = _JIEBA_CACHE
    jieba.setLogLevel(logging.WARNING)  # 抑制 jieba 首次建缓存的常规提示
except Exception:  # pragma: no cover - 缓存设置失败不影响分词功能
    pass


def tokenize(text: str) -> List[str]:
    """中英文混合分词：jieba 切中文，英文/数字整词保留，过滤空白与单标点。

    :param text: 原始文本
    :return: 词项列表
    """
    tokens = [t.strip() for t in jieba.lcut(text.lower()) if t.strip()]
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


class TfidfSparseStore:
    """字符 ngram TF-IDF 稀疏向量库：CSR 矩阵行归一化，点积即余弦。"""

    def __init__(self, vectorizer: TfidfVectorizer, matrix: csr_matrix) -> None:
        """保存已拟合的向量化器与归一化文档矩阵。

        :param vectorizer: 已 fit 的 TfidfVectorizer
        :param matrix: shape=(n, vocab) 的 L2 归一化 CSR 矩阵
        """
        self.vectorizer = vectorizer
        self.matrix = matrix

    @classmethod
    def build(cls, texts: List[str]) -> "TfidfSparseStore":
        """拟合 TF-IDF 并生成归一化稀疏文档矩阵。

        :param texts: 全部检索块文本
        :return: 可检索的稀疏向量库
        """
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=CONFIG.max_features)
        matrix = vectorizer.fit_transform(texts).astype(np.float32)
        # 行归一化后稀疏点积等价于余弦相似度
        matrix = normalize(matrix, norm="l2", copy=False)
        return cls(vectorizer, matrix.tocsr())

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """检索与查询最相似的 Top-K 文档块。

        :param query: 用户问题
        :param top_k: 返回条数
        :return: [(块下标, 余弦分), ...] 降序
        """
        q = normalize(self.vectorizer.transform([query]).astype(np.float32),
                      norm="l2", copy=False)
        scores = np.asarray((self.matrix @ q.T).todense()).ravel()
        k = min(top_k, scores.shape[0])
        # argpartition 先取 Top-K 再精确排序，避免全量排序
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        return sorted(((int(i), float(scores[i])) for i in idx),
                      key=lambda x: x[1], reverse=True)


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
        :return: [(块下标, 得分), ...] 降序
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


def scan_sources(data_dir: str) -> List[dict]:
    """扫描数据目录下全部 PDF，生成来源清单（名称/大小/修改时间）。

    :param data_dir: PDF 挂载目录
    :return: 按文件名排序的来源描述列表
    """
    sources = []
    if not os.path.isdir(data_dir):
        return sources
    for name in sorted(os.listdir(data_dir)):
        if not name.lower().endswith(".pdf"):
            continue
        full = os.path.join(data_dir, name)
        if not os.path.isfile(full):
            continue
        stat = os.stat(full)
        sources.append({"name": name, "size": stat.st_size,
                        "mtime": int(stat.st_mtime)})
    return sources


class IndexStore:
    """聚合 TF-IDF 向量库、BM25 与块元数据，提供构建/保存/加载/双路召回。"""

    META_NAME = "meta.json"

    def __init__(self, chunks: List[Chunk], dense: TfidfSparseStore,
                 sparse: BM25Store, sources: List[dict]) -> None:
        """保存索引要素。

        :param chunks: 检索块列表（下标对齐向量行/倒排文档）
        :param dense: TF-IDF 稀疏向量库
        :param sparse: BM25 倒排
        :param sources: 建库 PDF 来源清单
        """
        self.chunks = chunks
        self.dense = dense
        self.sparse = sparse
        self.sources = sources

    # ---------- 构建 ----------
    @classmethod
    def build(cls, chunks: List[Chunk], sources: List[dict]) -> "IndexStore":
        """用全部块文本构建 TF-IDF + BM25 双路索引。

        :param chunks: 检索块列表
        :param sources: PDF 来源清单
        :return: 可检索的 IndexStore
        """
        texts = [c.text for c in chunks]
        logger.info("开始拟合 TF-IDF 字符 ngram 向量库（块数=%d）...", len(texts))
        dense = TfidfSparseStore.build(texts)
        logger.info("开始构建 BM25 倒排索引...")
        sparse = BM25Store([tokenize(t) for t in texts])
        return cls(chunks, dense, sparse, sources)

    # ---------- 双路召回 ----------
    def dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """TF-IDF 向量召回。

        :param query: 用户查询
        :param top_k: 返回条数
        """
        return self.dense.search(query, top_k)

    def sparse_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """BM25 关键词召回。

        :param query: 用户查询
        :param top_k: 返回条数
        """
        return self.sparse.search(tokenize(query), top_k)

    # ---------- 持久化 ----------
    def _meta(self) -> dict:
        """生成索引元数据（来源清单 + 运行时版本，用于缓存命中判断）。"""
        return {
            "index_version": CONFIG.index_version,
            "embedding": "tfidf-char-2-4gram",
            "chunk_count": len(self.chunks),
            "sources": self.sources,
            "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "platform": platform.platform(),
            },
        }

    def save(self, index_dir: str) -> None:
        """持久化索引到目录（Docker 卷挂载点）。

        写入文件：vectors.npz（稀疏矩阵）、tfidf.pkl、bm25.pkl、
        chunks.pkl、meta.json。先写临时文件再替换，避免容器被 kill 时产生半截索引。

        :param index_dir: 索引目录
        """
        os.makedirs(index_dir, exist_ok=True)
        # 清理上次异常退出可能遗留的临时文件，避免污染卷目录
        for stale in os.listdir(index_dir):
            if stale.endswith(".tmp") or stale.endswith(".tmp.npz"):
                try:
                    os.remove(os.path.join(index_dir, stale))
                except OSError:
                    pass
        tmp_suffix = ".tmp"
        # 注意：scipy.sparse.save_npz 对非 .npz 结尾的路径会自动补 .npz 后缀，
        # 因此临时文件实际落盘名为 vectors.npz.tmp.npz，改名时需按实际路径处理
        save_npz(os.path.join(index_dir, "vectors.npz" + tmp_suffix),
                 self.dense.matrix)
        tmp_files = {
            "vectors.npz": os.path.join(index_dir, "vectors.npz.tmp.npz"),
        }
        with open(os.path.join(index_dir, "tfidf.pkl" + tmp_suffix), "wb") as f:
            pickle.dump(self.dense.vectorizer, f)
        tmp_files["tfidf.pkl"] = os.path.join(index_dir, "tfidf.pkl" + tmp_suffix)
        with open(os.path.join(index_dir, "bm25.pkl" + tmp_suffix), "wb") as f:
            pickle.dump(self.sparse, f)
        tmp_files["bm25.pkl"] = os.path.join(index_dir, "bm25.pkl" + tmp_suffix)
        with open(os.path.join(index_dir, "chunks.pkl" + tmp_suffix), "wb") as f:
            pickle.dump(self.chunks, f)
        tmp_files["chunks.pkl"] = os.path.join(index_dir, "chunks.pkl" + tmp_suffix)
        with open(os.path.join(index_dir, self.META_NAME + tmp_suffix), "w",
                  encoding="utf-8") as f:
            json.dump(self._meta(), f, ensure_ascii=False, indent=2)
        tmp_files[self.META_NAME] = os.path.join(
            index_dir, self.META_NAME + tmp_suffix)
        # 全部写完后原子改名，保证加载侧永远看到完整索引
        for name, tmp_path in tmp_files.items():
            os.replace(tmp_path, os.path.join(index_dir, name))
        logger.info("索引已持久化到卷目录: %s", index_dir)

    @classmethod
    def is_cache_fresh(cls, index_dir: str, sources: List[dict]) -> Tuple[bool, str]:
        """校验卷内索引是否可直接复用。

        规则：meta 存在、index_version 一致、PDF 来源清单（名/大小/时间）一致、
        运行时关键库版本一致（避免跨版本 pickle/稀疏矩阵不兼容）。

        :param index_dir: 索引目录
        :param sources: 当前扫描到的 PDF 来源清单
        :return: (是否命中缓存, 原因说明)
        """
        meta_path = os.path.join(index_dir, cls.META_NAME)
        if not os.path.exists(meta_path):
            return False, "卷内无 meta.json，判定为首次建库"
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            return False, f"meta.json 无法读取: {exc}"
        if meta.get("index_version") != CONFIG.index_version:
            return False, "索引算法版本不一致，需要重建"
        if meta.get("sources") != sources:
            return False, "PDF 来源清单（文件名/大小/时间）变化，需要重建"
        runtime = meta.get("runtime", {})
        if runtime.get("scikit_learn") != sklearn.__version__:
            return False, ("scikit-learn 版本变化（卷内 "
                           f"{runtime.get('scikit_learn')} / 当前 {sklearn.__version__}），需要重建")
        if runtime.get("numpy") != np.__version__:
            return False, "numpy 版本变化，需要重建"
        required = ("vectors.npz", "tfidf.pkl", "bm25.pkl", "chunks.pkl")
        missing = [n for n in required
                   if not os.path.exists(os.path.join(index_dir, n))]
        if missing:
            return False, f"卷内缺少索引文件 {missing}，需要重建"
        return True, f"卷内索引命中缓存（{meta.get('built_at')} 建库）"

    @classmethod
    def load(cls, index_dir: str, sources: List[dict]) -> "IndexStore":
        """从卷目录加载完整索引。

        :param index_dir: 索引目录
        :param sources: 当前 PDF 来源清单（仅做登记透传）
        """
        matrix = load_npz(os.path.join(index_dir, "vectors.npz")).tocsr()
        with open(os.path.join(index_dir, "tfidf.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            sparse = pickle.load(f)
        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
        return cls(chunks, TfidfSparseStore(vectorizer, matrix), sparse, sources)
