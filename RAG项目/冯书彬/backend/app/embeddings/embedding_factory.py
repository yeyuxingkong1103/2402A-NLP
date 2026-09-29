from functools import lru_cache

from backend.app.core.startup_checks import DependencyStatus
from backend.app.embeddings.base import EmbeddingClient
from backend.app.embeddings.bge_m3 import BgeM3EmbeddingClient
from backend.app.rerank.rerank_factory import get_rerank_client


@lru_cache(maxsize=1)
def get_embedding_client() -> EmbeddingClient:
    # 工厂集中创建客户端，并缓存模型实例，避免重复占用 GPU 显存。
    return BgeM3EmbeddingClient()


def check_model_readiness() -> list[DependencyStatus]:
    # readiness 明确加载两个本地 GPU 模型，不做 CPU 或量化降级。
    statuses: list[DependencyStatus] = []
    try:
        # 通过工厂加载 embedding 模型，复用缓存避免重复占用显存。
        get_embedding_client()
        statuses.append(DependencyStatus(name="bge_m3", ok=True, detail="cuda loaded"))
    except Exception as exc:
        # 只记录异常类型，避免日志和返回值携带模型路径或用户文本。
        statuses.append(DependencyStatus(name="bge_m3", ok=False, detail=type(exc).__name__))
    try:
        # 通过工厂加载 reranker 模型，验证本地 FP16 CUDA readiness。
        get_rerank_client()
        statuses.append(DependencyStatus(name="bge_reranker", ok=True, detail="cuda loaded"))
    except Exception as exc:
        # 失败信息保持脱敏；CUDA/显存问题由调用方据此报告 BLOCKED。
        statuses.append(DependencyStatus(name="bge_reranker", ok=False, detail=type(exc).__name__))
    return statuses
