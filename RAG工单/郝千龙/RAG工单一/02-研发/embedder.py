# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【向量化组件 · embedder.py】将文本块嵌入为向量，Query 实时向量化复用同一单例
from typing import List
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

import config

# CPU 高并发服务：限制单请求 torch 运算线程，避免多请求线程超额订阅
# （22 核上 10 个并发 × 2 线程 ≈ 20，接近物理核数，并发吞吐反而更优）
os.environ.setdefault("OMP_NUM_THREADS", "2")
torch.set_num_threads(2)


class Embedder:
    """单例 Embedding 编码器"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def __init__(self, model_name: str = None, device: str = None):
        # 仅首次初始化时加载模型
        if self._model is None:
            model_name = model_name or config.EMBED_MODEL_NAME
            device = device or config.EMBED_DEVICE
            print(f"[INFO] 加载 Embedding 模型: {model_name} on {device}")
            self._model = SentenceTransformer(model_name, device=device)
            # 实际维度可能与配置不同，校准；兼容新旧版 sentence-transformers API
            if hasattr(self._model, "get_embedding_dimension"):
                dim = self._model.get_embedding_dimension()
            else:  # pragma: no cover
                dim = self._model.get_sentence_embedding_dimension()
            if dim != config.EMBED_DIM:
                print(f"[WARN] 模型实际维度 {dim} 与配置 {config.EMBED_DIM} 不一致，以实际为准")
            self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str], batch_size: int = None,
               normalize: bool = True) -> np.ndarray:
        """
        批量向量化，默认 L2 归一化（搭配 COSINE 度量）
        返回: np.ndarray, shape=(N, dim), dtype=float32
        """
        batch_size = batch_size or config.EMBED_BATCH_SIZE
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        ).astype(np.float32)
        return vecs

    def encode_one(self, text: str, normalize: bool = True) -> np.ndarray:
        """单条文本向量化"""
        return self.encode([text], normalize=normalize)[0]


if __name__ == "__main__":
    # 自测
    emb = Embedder()
    v = emb.encode_one("武汉兴图新科电子股份有限公司")
    print(f"维度: {v.shape}, 前5维: {v[:5]}")

# ====================================================================
# 技术备注：
# 1. Transformer：SentenceTransformer 基于 BERT/MPNet Transformer 主干，
#    使用 mean pooling 聚合 token 表示得到句向量。
# 2. RAG：Embedding 是检索召回的核心，决定 Top-K 质量。
# 3. bge-small-zh：北京智源研究院开源的中文检索优化模型，512 维，速度快。
# 4. L2 归一化后内积 = 余弦相似度，便于在 Milvus 用 IP 索引提速。
# 5. Fine-tuning：可使用 InfoNCE/对比学习对 bge 在招股说明书领域微调，
#    提升专业术语与表格描述的检索能力。
# ====================================================================
