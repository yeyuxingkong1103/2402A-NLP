# -*- coding: utf-8 -*-
"""向量化客户端：调用本地 Ollama 的 bge-m3 模型批量编码。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

说明：为降低环境依赖（无需 faiss / sentence-transformers / torch），
本工单统一通过 Ollama HTTP 接口获取稠密向量，向量以 numpy 形式持久化，
检索时用矩阵乘法完成余弦相似度计算（切片规模约千级，毫秒级完成）。
"""
from __future__ import annotations

import time
from typing import List, Sequence

import numpy as np
import requests

from src import config


class Embedder:
    """Ollama 文本向量化客户端（带内存缓存与批量编码）。"""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or config.EMBEDDING_MODEL
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self._cache: dict[str, np.ndarray] = {}

    # -- 基础能力 ------------------------------------------------------------
    def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": list(texts)},
            timeout=config.EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def available(self) -> bool:
        """探测 Ollama 服务是否可用。"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # -- 对外接口 ------------------------------------------------------------
    def encode(self, texts: Sequence[str], batch_size: int | None = None,
               verbose: bool = False) -> np.ndarray:
        """批量编码，返回 (N, D) 的 float32 矩阵。"""
        bs = batch_size or config.EMBED_BATCH_SIZE
        out: List[np.ndarray] = []
        todo = list(texts)
        t0 = time.time()
        for i in range(0, len(todo), bs):
            batch = todo[i:i + bs]
            vecs = self._embed_batch(batch)
            out.extend(np.asarray(v, dtype=np.float32) for v in vecs)
            if verbose and (i // bs) % 10 == 0:
                print(f"  embedded {min(i + bs, len(todo))}/{len(todo)} "
                      f"({time.time() - t0:.1f}s)", flush=True)
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(out).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """编码单条文本（带缓存），返回 (D,) 向量。"""
        key = text.strip()
        if key in self._cache:
            return self._cache[key]
        vec = self.encode([key])[0]
        self._cache[key] = vec
        return vec


def normalize(mat: np.ndarray) -> np.ndarray:
    """按行 L2 归一化（余弦相似度 = 归一化后内积）。"""
    if mat.size == 0:
        return mat
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (mat / norm).astype(np.float32)