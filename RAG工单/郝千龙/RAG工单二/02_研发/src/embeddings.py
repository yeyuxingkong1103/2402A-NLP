# -*- coding: utf-8 -*-
# 【嵌入模型封装 · embeddings.py】BGE双语嵌入，断网/无GPU时自动降级为TF-IDF向量
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

"""嵌入层：统一接口封装稠密向量编码，保证离线可运行。

- 在线/有依赖：sentence-transformers 加载 BAAI/bge-base-zh-v1.5（中英文同空间）；
- 离线降级：scikit-learn TF-IDF（字符 ngram，对中文专名/数字友好）。
查询侧按 BGE 官方建议追加 instruction 前缀。
"""
import logging
from typing import List, Optional

import numpy as np

from config import CONFIG

logger = logging.getLogger(__name__)


class BaseEmbedder:
    """嵌入器统一接口（所有实现必须返回 L2 归一化后的向量）。"""

    dim: int = 0

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """编码文档块。

        :param texts: 文档文本列表
        :return: shape=(n, dim) 的归一化浮点矩阵
        """
        raise NotImplementedError

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """编码用户查询（可追加检索指令前缀）。

        :param texts: 查询文本列表
        :return: shape=(n, dim) 的归一化浮点矩阵
        """
        raise NotImplementedError

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """对向量按行做 L2 归一化，使余弦相似度等价于点积。

        :param vectors: 原始向量矩阵
        :return: 归一化后的矩阵
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms


class BGEEmbedder(BaseEmbedder):
    """基于 sentence-transformers 的 BGE 双语嵌入实现。"""

    def __init__(self, model_name: Optional[str] = None) -> None:
        """加载本地/远程 BGE 模型。

        :param model_name: HuggingFace 模型名或本地路径
        """
        from sentence_transformers import SentenceTransformer  # 延迟导入
        self.model = SentenceTransformer(model_name or CONFIG.embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """编码文档块。"""
        vec = self.model.encode(texts, normalize_embeddings=True,
                                convert_to_numpy=True, batch_size=32)
        return vec.astype(np.float32)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """编码查询：中文 BGE 加检索指令前缀以提升召回。"""
        queries = [f"{CONFIG.query_instruction}{t}" for t in texts]
        vec = self.model.encode(queries, normalize_embeddings=True,
                                convert_to_numpy=True, batch_size=32)
        return vec.astype(np.float32)


class TfidfEmbedder(BaseEmbedder):
    """离线降级嵌入：字符级 2~4 gram TF-IDF（对中文数字/专名区分度好）。"""

    def __init__(self) -> None:
        """初始化 TF-IDF 向量化器（在 fit 后维度才确定）。"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(analyzer="char_wb",
                                          ngram_range=(2, 4),
                                          max_features=20000)
        self.dim = 0

    def fit(self, texts: List[str]) -> None:
        """用全部文档块拟合 IDF。

        :param texts: 文档文本列表
        """
        matrix = self.vectorizer.fit_transform(texts)
        self.dim = matrix.shape[1]

    def _encode(self, texts: List[str]) -> np.ndarray:
        """统一编码并转为稠密归一化矩阵。"""
        vec = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return self._l2_normalize(vec)

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """编码文档块。"""
        return self._encode(texts)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """编码查询（TF-IDF 无需指令前缀）。"""
        return self._encode(texts)


def create_embedder(doc_texts: Optional[List[str]] = None) -> BaseEmbedder:
    """嵌入器工厂：优先本地 BGE，不可用自动降级 TF-IDF，保证离线可服务。

    :param doc_texts: 文档块文本，TF-IDF 降级时用于拟合
    :return: 已就绪的嵌入器实例
    """
    from retriever import model_available_locally  # 复用本地缓存检测

    if CONFIG.fallback_embedding != "tfidf" and model_available_locally(
            CONFIG.embedding_model):
        try:
            logger.info("加载本地 BGE 嵌入模型: %s", CONFIG.embedding_model)
            return BGEEmbedder()
        except Exception as exc:  # 显存不足等运行时异常
            logger.warning("BGE 模型加载失败，降级 TF-IDF：%s", exc)
    elif CONFIG.fallback_embedding != "tfidf":
        logger.info("未检测到 %s 本地缓存，直接启用 TF-IDF 离线嵌入（不联网下载）",
                    CONFIG.embedding_model)
    embedder = TfidfEmbedder()
    # 仅在提供有效文档时拟合；仅用于查询加载场景时由索引文件恢复 vectorizer
    if doc_texts and any(t.strip() for t in doc_texts):
        embedder.fit(doc_texts)
    logger.info("已启用 TF-IDF 离线嵌入，维度=%d", embedder.dim)
    return embedder
