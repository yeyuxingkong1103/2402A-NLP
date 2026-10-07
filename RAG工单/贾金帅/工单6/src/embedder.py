"""
向量化模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

使用本机已下载的 bge-small-zh-v1.5（512 维，中文短文本检索表现稳定）。
模型在进程内**单例加载**：重复加载会把内存峰值推高一个量级，
本机 16GB 内存本来就紧张。

为什么不用在线 embedding API：本工单要求「响应时间不超过 3 秒」，
本地小模型单条编码实测 ~15ms（CPU），比一次网络往返更可控，且断网可用。
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL_PATH, WORK_ORDER_NO  # noqa: F401

_model = None
_model_lock = threading.Lock()


def get_model():
    """线程安全地取模型单例（首次调用才真正加载）。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                path = Path(EMBEDDING_MODEL_PATH)
                if not path.is_dir():
                    raise FileNotFoundError(
                        f"向量模型目录不存在：{path}。"
                        f"请修改 .env 里的 EMBEDDING_MODEL_PATH 指向本地 bge 模型。"
                    )
                _model = SentenceTransformer(str(path), device="cpu")
    return _model


def embed_texts(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
    """
    把文本编码成**已 L2 归一化**的向量矩阵。

    归一化后余弦相似度退化为点积，检索阶段直接一次矩阵乘法即可，
    比逐条算 cosine 快一个数量级。
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """单条查询编码，返回 (1, D) 的归一化向量。"""
    return embed_texts([text])


def cosine_scores(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """
    全量余弦打分。matrix 已归一化 → 直接点积。

    对 ~3000 块 × 512 维，本机实测 < 5ms，无需引入 FAISS/Milvus 等外部服务，
    少一层依赖就少一处故障点。
    """
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return matrix @ query_vec.reshape(-1).astype(np.float32)
