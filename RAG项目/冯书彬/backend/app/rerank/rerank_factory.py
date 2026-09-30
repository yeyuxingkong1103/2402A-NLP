from functools import lru_cache

from backend.app.rerank.base import RerankClient
from backend.app.rerank.bge_reranker import BgeRerankClient


@lru_cache(maxsize=1)
def get_rerank_client() -> RerankClient:
    # 工厂集中创建客户端，并缓存模型实例，避免重复占用 GPU 显存。
    return BgeRerankClient()
