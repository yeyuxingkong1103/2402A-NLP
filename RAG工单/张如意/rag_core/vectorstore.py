# -*- coding: utf-8 -*-
"""
向量库封装
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

优先使用 ChromaDB 持久化向量库；若环境缺少 chromadb，则自动降级为
纯 NumPy 的本地向量库（功能等价，便于在无依赖环境下演示）。

对外统一接口：add / search / count / reset / list_collections
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .chunk import Chunk

CHROMA_DIR = config.INDEX_DIR / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

_BACKEND: str | None = None


def backend() -> str:
    """探测可用的向量库后端。"""
    global _BACKEND
    if _BACKEND is None:
        try:
            import chromadb  # noqa: F401
            _BACKEND = "chroma"
        except ImportError:
            _BACKEND = "numpy"
    return _BACKEND


# ---------------------------------------------------------------------------
# Chroma 后端
# ---------------------------------------------------------------------------
def _chroma_collection(name: str):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


# ---------------------------------------------------------------------------
# NumPy 降级后端
# ---------------------------------------------------------------------------
class _NumpyStore:
    """极简本地向量库：向量 + 元数据存盘，检索用矩阵点积。"""

    def __init__(self, path: Path):
        self.path = path
        self.vectors: np.ndarray | None = None
        self.docs: list[dict] = []
        self._load()

    def _load(self):
        if self.path.exists():
            with self.path.open("rb") as f:
                obj = pickle.load(f)
            self.vectors, self.docs = obj["vectors"], obj["docs"]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as f:
            pickle.dump({"vectors": self.vectors, "docs": self.docs}, f)

    def add(self, vectors: np.ndarray, docs: list[dict]):
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.docs.extend(docs)
        self.save()

    def search(self, qvec: np.ndarray, top_k: int) -> list[dict]:
        if self.vectors is None or len(self.docs) == 0:
            return []
        sims = self.vectors @ qvec.reshape(-1)
        idx = np.argsort(-sims)[:top_k]
        return [{**self.docs[i], "score": float(sims[i])} for i in idx]


# ---------------------------------------------------------------------------
# 统一门面
# ---------------------------------------------------------------------------
class VectorStore:
    """
    向量库门面对象。

    用法：
        vs = VectorStore("prospectus")
        vs.add_chunks(chunks)                   # 自动批量编码入库
        vs.search("军用领域收入", top_k=5)       # 返回 [{chunk_id,text,score,...}]
    """

    def __init__(self, name: str):
        self.name = name
        self._np: _NumpyStore | None = None
        if backend() == "numpy":
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            self._np = _NumpyStore(config.INDEX_DIR / f"{safe}.pkl")

    # -- 写入 ---------------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64,
                   show_progress: bool = True) -> int:
        """批量编码并写入向量库，返回写入条数。"""
        from . import embed

        chunks = [c for c in chunks if c.text.strip()]
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vecs = embed.encode(texts, batch_size=batch_size, show_progress=show_progress)
        metas = [
            {
                "chunk_id": c.chunk_id, "doc": c.doc, "page": c.page,
                "type": c.type, "section": c.section or "",
            }
            for c in chunks
        ]
        ids = [c.chunk_id for c in chunks]

        if backend() == "chroma":
            col = _chroma_collection(self.name)
            for i in range(0, len(chunks), batch_size):
                col.add(
                    embeddings=vecs[i:i + batch_size].tolist(),
                    documents=texts[i:i + batch_size],
                    metadatas=metas[i:i + batch_size],
                    ids=ids[i:i + batch_size],
                )
        else:
            self._np.add(vecs, [{**m, "text": t} for m, t in zip(metas, texts)])
        return len(chunks)

    # -- 检索 ---------------------------------------------------------------
    def search(self, query: str, top_k: int = config.TOP_K_RECALL,
               where: dict | None = None) -> list[dict]:
        """向量相似度检索，返回按相似度降序的文档列表。"""
        from . import embed
        qvec = embed.encode(query, is_query=True)

        if backend() == "chroma":
            col = _chroma_collection(self.name)
            if col.count() == 0:
                return []
            res = col.query(
                query_embeddings=[qvec.tolist()],
                n_results=min(top_k, col.count()),
                where=where or None,
            )
            out = []
            for i in range(len(res["ids"][0])):
                meta = res["metadatas"][0][i] or {}
                out.append({
                    **meta,
                    "text": res["documents"][0][i],
                    "score": 1.0 - res["distances"][0][i],   # cosine 距离 -> 相似度
                })
            return out
        return self._np.search(qvec, top_k)

    def count(self) -> int:
        if backend() == "chroma":
            return _chroma_collection(self.name).count()
        return len(self._np.docs) if self._np else 0

    def reset(self) -> None:
        """清空集合（重建索引前调用）。"""
        if backend() == "chroma":
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            try:
                client.delete_collection(self.name)
            except Exception:
                pass
        elif self._np and self._np.path.exists():
            self._np.path.unlink()
            self._np = _NumpyStore(self._np.path)


def list_collections() -> list[str]:
    if backend() == "chroma":
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return [c.name for c in client.list_collections()]
    return [p.stem for p in config.INDEX_DIR.glob("*.pkl")]
