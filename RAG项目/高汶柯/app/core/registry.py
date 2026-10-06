"""资源注册：懒加载并复用的单例组件。"""
from __future__ import annotations

from functools import lru_cache

from app.core.embedder import Embedder
from app.core.llm import LLMClient
from app.core.reranker import Reranker


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient()


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
