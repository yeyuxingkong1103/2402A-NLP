# -*- coding: utf-8 -*-
# 【多嵌入模型封装 · embeddings.py】统一封装 bge/m3e 接口，本地无缓存时 TF-IDF 离线降级，禁止联网下载
# 工单编号：人工智能NLP-RAG-混合检索任务

"""嵌入层：统一接口封装稠密向量编码，满足“支持多种嵌入模型（bge、m3e 等）”要求。

- bge 后端：sentence-transformers 加载本地缓存的 BAAI/bge-base-zh-v1.5（中英双语），
  查询侧按官方建议追加 instruction 前缀；
- m3e 后端：sentence-transformers 加载本地缓存的 moka-ai/m3e-base（中文），查询不加指令；
- 离线降级：本地无任何模型缓存（或显式指定 tfidf）时使用 scikit-learn 字符 ngram TF-IDF，
  全程不发起任何网络请求（local_files_only=True）。
"""
import glob
import logging
import os
from typing import List, Optional

# 离线环境：禁止 HuggingFace 任何联网探测（否则 sentence-transformers
# 即使 local_files_only=True 仍会发 HEAD 请求，在无外网时卡数十秒到两分钟）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

from config import EMBEDDER_MODELS, CONFIG

logger = logging.getLogger(__name__)


def model_available_locally(model_name: str) -> bool:
    """检测 HF 模型是否已在本地缓存或就是本地路径（离线环境严禁联网下载）。

    :param model_name: HF 模型 ID 或本地路径
    :return: 本地可用返回 True
    """
    if not model_name or model_name == "tfidf":
        return False
    if os.path.exists(model_name):
        return True
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    repo = "models--" + model_name.replace("/", "--")
    return bool(glob.glob(os.path.join(hub, repo, "snapshots", "*")))


class BaseEmbedder:
    """嵌入器统一接口（所有实现返回 L2 归一化后的向量）。"""

    dim: int = 0
    backend: str = "base"

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """编码文档块。

        :param texts: 文档文本列表
        :return: shape=(n, dim) 的归一化浮点矩阵
        """
        raise NotImplementedError

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """编码用户查询。

        :param texts: 查询文本列表
        :return: shape=(n, dim) 的归一化浮点矩阵
        """
        raise NotImplementedError

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """按行 L2 归一化，使余弦相似度等价于点积。

        :param vectors: 原始向量矩阵
        :return: 归一化矩阵
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms


class STEmbedder(BaseEmbedder):
    """基于 sentence-transformers 的本地稠密嵌入实现（bge / m3e 共用）。"""

    def __init__(self, model_name: str, backend: str,
                 add_instruction: bool = False) -> None:
        """以离线方式加载本地缓存模型（禁止联网）。

        :param model_name: HF 模型 ID（必须已在本地缓存）
        :param backend: 业务后端名（bge / m3e）
        :param add_instruction: 查询侧是否追加 bge 检索指令
        """
        from sentence_transformers import SentenceTransformer
        # local_files_only=True 双保险：环境无外网时也不会卡住或下载
        self.model = SentenceTransformer(model_name, device="cpu",
                                         local_files_only=True)
        self.model_name = model_name
        self.backend = backend
        self.add_instruction = add_instruction
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """编码文档块。"""
        vec = self.model.encode(texts, normalize_embeddings=True,
                                convert_to_numpy=True, batch_size=64,
                                show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """编码查询：bge 中文模型加检索指令，m3e 直接编码。"""
        if self.add_instruction:
            texts = [f"{CONFIG.query_instruction}{t}" for t in texts]
        vec = self.model.encode(texts, normalize_embeddings=True,
                                convert_to_numpy=True, batch_size=64,
                                show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32)


class TfidfEmbedder(BaseEmbedder):
    """离线降级嵌入：字符级 2~4 gram TF-IDF（对中文数字/专名区分度好）。"""

    backend = "tfidf"

    def __init__(self) -> None:
        """初始化 TF-IDF 向量化器（fit 后维度才确定）。"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(analyzer="char_wb",
                                          ngram_range=(2, 4),
                                          max_features=30000)
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


def create_embedder(backend: Optional[str] = None,
                    doc_texts: Optional[List[str]] = None) -> BaseEmbedder:
    """嵌入器工厂：按配置返回 bge / m3e 稠密嵌入或 TF-IDF 离线降级嵌入。

    选择逻辑：
    1. 后端显式为 tfidf → 直接 TF-IDF；
    2. 后端为 bge/m3e 且模型在本地缓存 → 离线加载，禁止联网；
    3. 本地无缓存 → 自动降级 TF-IDF（记录日志，不抛异常）。

    :param backend: EMBEDDER_MODELS 的键，缺省取 CONFIG.embedding_backend
    :param doc_texts: 文档块文本，TF-IDF 降级时用于拟合
    :return: 已就绪的嵌入器实例
    """
    backend = backend or CONFIG.embedding_backend
    model_name = EMBEDDER_MODELS.get(backend, backend)

    if model_name == "tfidf":
        logger.info("按配置启用 TF-IDF 离线嵌入（不加载外部模型）")
        embedder = TfidfEmbedder()
        if doc_texts and any(t.strip() for t in doc_texts):
            embedder.fit(doc_texts)
        logger.info("TF-IDF 嵌入就绪，维度=%d", embedder.dim)
        return embedder

    if model_available_locally(model_name):
        try:
            logger.info("本地加载 %s 嵌入模型（离线，不联网下载）: %s",
                        backend, model_name)
            return STEmbedder(model_name, backend=backend,
                              add_instruction=(backend.startswith("bge")))
        except Exception as exc:  # 加载异常一律降级，保证服务可用
            logger.warning("%s 模型加载失败，降级 TF-IDF：%s", backend, exc)
    else:
        logger.info("未检测到 %s（%s）本地缓存，启用 TF-IDF 离线降级（禁止联网下载）",
                    backend, model_name)

    embedder = TfidfEmbedder()
    if doc_texts and any(t.strip() for t in doc_texts):
        embedder.fit(doc_texts)
    logger.info("TF-IDF 嵌入就绪，维度=%d", embedder.dim)
    return embedder
