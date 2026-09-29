# app/config/components.py
"""重资源单例：BGE-M3 / BGE-reranker / LLM。

全部**懒加载**——import 本模块不会立刻加载模型，
避免脚本或单元测试一启动就占用显存 / 拖慢启动。
"""
import threading
from typing import List

import torch

from app.config import settings
import logging

logger = logging.getLogger(__name__)

_embedder = None
_reranker = None
_lock = threading.Lock()


# ============================================================
# 1. BGE-M3：同时产出稠密向量与稀疏权重，支撑混合检索
# ============================================================
class BGEM3Embeddings:
    def __init__(self, model_path: str):
        from FlagEmbedding import BGEM3FlagModel
        self.use_fp16 = torch.cuda.is_available()
        logger.info("加载 BGE-M3: %s (fp16=%s)", model_path, self.use_fp16)
        self.model = BGEM3FlagModel(model_name_or_path=model_path,
                                    use_fp16=self.use_fp16)
        logger.info("BGE-M3 加载完成")

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """批量稠密向量"""
        if not texts:
            return []
        out = self.model.encode(texts, batch_size=batch_size, return_dense=True,
                                return_sparse=False, return_colbert_vecs=False)
        return out["dense_vecs"].tolist()

    def embed_query(self, text: str) -> List[float]:
        """单条查询稠密向量"""
        out = self.model.encode([text], return_dense=True,
                                return_sparse=False, return_colbert_vecs=False)
        return out["dense_vecs"][0].tolist()

    def compute_sparse(self, texts: List[str], batch_size: int = 32) -> List[dict]:
        """稀疏权重（lexical weights），供 Milvus 稀疏向量字段使用。"""
        if not texts:
            return []
        out = self.model.encode(texts, batch_size=batch_size, return_dense=False,
                                return_sparse=True, return_colbert_vecs=False)
        return out["lexical_weights"]

    def embed_both(self, texts: List[str], batch_size: int = 32):
        """一次前向同时拿稠密 + 稀疏，比调用两次快接近一倍。"""
        if not texts:
            return [], []
        out = self.model.encode(texts, batch_size=batch_size, return_dense=True,
                                return_sparse=True, return_colbert_vecs=False)
        return out["dense_vecs"].tolist(), out["lexical_weights"]


def get_embeddings() -> BGEM3Embeddings:
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                _embedder = BGEM3Embeddings(settings.EMBEDDING_MODEL_PATH)
    return _embedder


# ============================================================
# 2. 重排序模型
# ============================================================
def get_reranker():
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                from app.core.retrieve_service import RerankService
                _reranker = RerankService(
                    model_path=settings.RERANKER_MODEL_PATH,
                    use_fp16=torch.cuda.is_available())
    return _reranker


# ============================================================
# 3. 大模型
# ============================================================
def get_llm():
    from app.core.llm_service import LLMService
    return LLMService.instance()


# 文本切分已改由 app/core/chunk_service.py 的语义分块负责，
# 不再需要 RecursiveCharacterTextSplitter
