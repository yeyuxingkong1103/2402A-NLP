# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：嵌入模型模块。封装 BGE-M3，提供 LangChain Embeddings 接口，
          用于把文档片段与用户问题映射到统一的向量空间。
"""
import numpy as np
from langchain_core.embeddings import Embeddings

import config
from src.utils import logger

_model_cache = {}


def _resolve_device() -> str:
    """自动选择运行设备：优先 CUDA，其次 CPU。"""
    if config.EMBEDDING_DEVICE:
        return config.EMBEDDING_DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 归一化，配合内积（IP）等价于余弦相似度。"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return vectors / norms


class BGEM3Embeddings(Embeddings):
    """基于 BGE-M3 的向量化实现（优先 FlagEmbedding，缺失时回退 sentence-transformers）。"""

    def __init__(self, model_path: str = None, batch_size: int = None, max_length: int = None):
        self.model_path = model_path or config.EMBEDDING_MODEL_PATH
        self.batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        self.max_length = max_length or config.EMBEDDING_MAX_LENGTH
        self.device = _resolve_device()
        self.backend = ""
        self._model = self._load_model()

    def _load_model(self):
        """加载模型，兼容两种后端。"""
        try:
            from FlagEmbedding import BGEM3FlagModel

            model = BGEM3FlagModel(self.model_path, use_fp16=self.device == "cuda", device=self.device)
            self.backend = "FlagEmbedding"
            logger.info("BGE-M3 加载完成（FlagEmbedding），device=%s", self.device)
            return model
        except Exception as exc:
            logger.warning("FlagEmbedding 加载失败（%s），回退 sentence-transformers", exc)

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_path, device=self.device)
        model.max_seq_length = self.max_length
        self.backend = "sentence-transformers"
        logger.info("BGE-M3 加载完成（sentence-transformers），device=%s", self.device)
        return model

    def _encode(self, texts: list) -> np.ndarray:
        if not texts:
            return np.zeros((0, config.MILVUS_DIM), dtype=np.float32)
        if self.backend == "FlagEmbedding":
            output = self._model.encode(
                texts, batch_size=self.batch_size, max_length=self.max_length
            )
            vectors = np.asarray(output["dense_vecs"], dtype=np.float32)
        else:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
        return _normalize(vectors)

    def embed_documents(self, texts: list) -> list:
        """向量化文档列表。"""
        return self._encode(list(texts)).tolist()

    def embed_query(self, text: str) -> list:
        """向量化单条查询。"""
        return self._encode([text])[0].tolist()


def get_embedding_model() -> BGEM3Embeddings:
    """获取嵌入模型单例，避免重复加载占用显存。"""
    if "embedding" not in _model_cache:
        _model_cache["embedding"] = BGEM3Embeddings()
    return _model_cache["embedding"]
