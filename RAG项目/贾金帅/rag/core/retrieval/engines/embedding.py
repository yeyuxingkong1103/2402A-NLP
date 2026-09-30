# -*- coding: utf-8 -*-
"""向量化函数：远端 BGE 服务优先，本地 SentenceTransformer 兜底。

自 ``vector_store.py`` 拆出。
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from src import config

logger = logging.getLogger(__name__)


EmbeddingFn = Callable[[list[str]], list[list[float]]]

def remote_embedding_fn(texts: list[str], endpoint: Optional[str] = None,
                       timeout: Optional[float] = None, post=None) -> list[list[float]]:
    """Call the project's BGE embedding service using its ``/embed`` API."""
    if not texts:
        return []
    endpoint = (endpoint or config.EMBEDDING_API_URL).rstrip("/") + "/embed"
    timeout = float(timeout if timeout is not None else config.EMBEDDING_API_TIMEOUT)
    if post is None:
        import requests

        post = requests.post
    response = post(
        endpoint,
        json={
            "texts": list(texts),
            "normalize_embeddings": True,
            "is_query": True,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise ValueError("embedding 服务返回的数据格式不正确")
    dimension = int(payload.get("dimension") or 0)
    if dimension != config.EMBEDDING_DIM:
        raise ValueError(
            f"embedding 维度不匹配: service={dimension}, expected={config.EMBEDDING_DIM}"
        )
    if any(not isinstance(vector, list) or len(vector) != dimension
           for vector in embeddings):
        raise ValueError("embedding 向量格式或长度有误")
    return [[float(value) for value in vector] for vector in embeddings]


def default_embedding_fn() -> EmbeddingFn:
    """Remote BGE first, with the local SentenceTransformer model as fallback."""
    local_fn = None
    local_error = None

    def _load_local() -> EmbeddingFn:
        nonlocal local_fn, local_error
        if local_fn is not None:
            return local_fn
        if local_error is not None:
            raise local_error
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        except (ImportError, OSError) as exc:
            local_error = ImportError(
                f"Embedding 模型加载失败: {exc}请确保已安装 sentence-transformers 库 "
                f"并检查模型路径{config.EMBEDDING_MODEL_NAME}是否正确。"
            )
            raise local_error from exc

        def _local_embed(texts: list[str]) -> list[list[float]]:
            vectors = model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

        local_fn = _local_embed
        return local_fn

    def _embed(texts: list[str]) -> list[list[float]]:
        if not (config.EMBEDDING_API_URL or "").strip():
            # 未配置远端 embedding 服务时直接使用本地模型，避免每次检索空等/误报。
            return _load_local()(texts)
        try:
            return remote_embedding_fn(texts)
        except Exception as exc:
            logger.warning("远程 embedding 服务调用失败: %s", exc)
            return _load_local()(texts)

    return _embed
