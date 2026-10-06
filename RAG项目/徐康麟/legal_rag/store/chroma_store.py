# -*- coding: utf-8 -*-
"""ChromaDB 向量库实现（本地轻量，作为 Milvus 的过渡）。

chromadb 是可选重依赖：未安装时给出带修复命令的 RuntimeError。
"""
from __future__ import annotations

from pathlib import Path

from ..schemas import Chunk, SearchHit
from .base import VectorStore

DEFAULT_SUBDIR = "chroma"


class ChromaVectorStore(VectorStore):
    """ChromaDB 后端：本地文件持久化，作为 Milvus 的轻量替代。

    * 懒加载：构造时**不**导入 chromadb，第一次真正用到才 import ——
      未安装时抛的 ``RuntimeError`` 里带**可复制执行**的修复命令
      （装 `requirements-full.txt` 或改 ``VECTOR_STORE=memory``）；
    * 距离口径：collection 建的是 ``hnsw:space=cosine``，Chroma 返回**距离**，
      这里统一换算成相似度 ``1 - distance``，让上层与其它后端可比；
    * ⚠️ 本后端**未在本项目真机验证过**（见 `README.md` 已知限制：重依赖路线未验证），
      生产用 Milvus；它的存在是为了"换后端不改上层代码"。
    """

    name = "chroma"

    def __init__(self, persist_dir=None, collection: str = "legal_kb") -> None:
        self.persist_dir = Path(persist_dir) if persist_dir else Path(".")
        self.collection_name = collection
        self._client = None
        self._collection = None

    # ---------- 懒加载 ----------
    def _ensure(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "未安装 chromadb，无法使用 ChromaDB 向量库。\n"
                "请先安装重依赖：\n"
                "    .venv\\Scripts\\python.exe -m pip install -r requirements-full.txt\n"
                "或改用内存兜底：VECTOR_STORE=memory"
            ) from exc

        path = self.persist_dir / DEFAULT_SUBDIR
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    @staticmethod
    def _metadata(chunk: Chunk) -> dict:
        meta = chunk.to_metadata()
        meta["is_parent"] = bool(chunk.is_parent)
        meta["parent_id"] = chunk.parent_id or ""
        return {k: v for k, v in meta.items() if v is not None and not isinstance(v, (list, dict))}

    # ---------- VectorStore 接口 ----------
    def upsert(self, chunks: list[Chunk]) -> int:
        """按 id 覆盖写入。**没有向量的 chunk 会被丢掉**（Chroma 必须给 embedding）。"""
        payload = [c for c in chunks if c.vector]
        if not payload:
            return 0
        collection = self._ensure()
        collection.upsert(
            ids=[c.id for c in payload],
            embeddings=[list(c.vector) for c in payload],
            documents=[c.text for c in payload],
            metadatas=[self._metadata(c) for c in payload],
        )
        return len(payload)

    def search(self, vector: list[float], top_k: int = 10,
               where: dict | None = None) -> list[SearchHit]:
        """向量检索 top-k；空集合直接返回 ``[]``（Chroma 对空集合 query 会报错）。"""
        collection = self._ensure()
        if collection.count() == 0:
            return []
        result = collection.query(
            query_embeddings=[list(vector)],
            n_results=min(top_k, max(collection.count(), 1)),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for i, chunk_id in enumerate(ids):
            meta = dict(metadatas[i] or {})
            text = documents[i] if i < len(documents) else meta.get("text", "")
            meta.pop("text", None)
            chunk = Chunk(id=chunk_id, text=text, **{
                k: v for k, v in meta.items()
                if k in Chunk.__dataclass_fields__ and k not in ("id", "text")
            })
            # cosine 距离 -> 相似度
            score = 1.0 - float(distances[i]) if i < len(distances) else 0.0
            hits.append(SearchHit(chunk=chunk, score=score, vector_score=score))
        return hits

    def get(self, chunk_id: str) -> Chunk | None:
        """按 id 取单条；不存在返回 ``None``（不抛）。"""
        collection = self._ensure()
        result = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        ids = result.get("ids") or []
        if not ids:
            return None
        text = (result.get("documents") or [""])[0]
        meta = dict((result.get("metadatas") or [{}])[0] or {})
        meta.pop("text", None)
        return Chunk(id=ids[0], text=text, **{
            k: v for k, v in meta.items()
            if k in Chunk.__dataclass_fields__ and k not in ("id", "text")
        })

    def all_chunks(self, where: dict | None = None) -> list[Chunk]:
        """按 ``where`` 取全部 chunk（BM25 建索引用；注意 Chroma 单次返回有上限）。"""
        collection = self._ensure()
        result = collection.get(where=where or None, include=["documents", "metadatas"])
        out: list[Chunk] = []
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for i, chunk_id in enumerate(ids):
            meta = dict(metadatas[i] or {})
            text = documents[i] if i < len(documents) else meta.get("text", "")
            meta.pop("text", None)
            out.append(Chunk(id=chunk_id, text=text, **{
                k: v for k, v in meta.items()
                if k in Chunk.__dataclass_fields__ and k not in ("id", "text")
            }))
        return out

    def delete(self, ids: list[str]) -> int:
        """按 id 删除；返回请求删除的条数（Chroma 不回传实际删除数）。"""
        if not ids:
            return 0
        collection = self._ensure()
        collection.delete(ids=list(ids))
        return len(ids)

    def delete_by_role(self, role_id: str) -> int:
        """先查出该 role 的全部 id 再删（Chroma 的 delete 只认 id）。"""
        collection = self._ensure()
        existing = collection.get(where={"role_id": role_id}, include=[])
        ids = existing.get("ids") or []
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        """活条数（Chroma 没有墓碑概念，其 count 就是活条数）。"""
        return self._ensure().count()
