# -*- coding: utf-8 -*-
"""向量化后端。

* ``offline`` —— 零依赖兜底（字符 n-gram 哈希 TF），用于离线模式与单测；
* ``bge_m3`` —— 重型本地实现（torch + sentence-transformers），需 ``requirements-full``；
* ``ollama`` —— **本阶段默认**：走 Ollama ``/api/embed``（bge-m3，1024 维），
  免密钥、免 torch，带批量、缓存、超时与退避重试。
"""
from .base import Embedder, build_embedder
from .cache import EmbedCache, MemoryEmbedCache, RedisEmbedCache, build_embed_cache
from .ollama_embedder import EmbeddingDimError, OllamaEmbedder, probe_dimension

__all__ = [
    "Embedder", "build_embedder",
    "OllamaEmbedder", "EmbeddingDimError", "probe_dimension",
    "EmbedCache", "RedisEmbedCache", "MemoryEmbedCache", "build_embed_cache",
]
