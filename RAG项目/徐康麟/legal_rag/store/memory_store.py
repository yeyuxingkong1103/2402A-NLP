# -*- coding: utf-8 -*-
"""内存向量库（零依赖兜底）。

全量暴力检索，规模在几千条 chunk 时完全够用；
可选 pickle 落盘，便于进程间复用索引。
"""
from __future__ import annotations

import pickle
from pathlib import Path

from ..schemas import Chunk, SearchHit
from ..utils import cosine_similarity
from .base import VectorStore

DEFAULT_FILENAME = "memory_store.pkl"


class MemoryVectorStore(VectorStore):
    """零依赖内存实现：dict 存全量 chunk，检索=遍历算余弦再排序。

    * **定位**：单测/离线链路，以及 Milvus 连不上时的**降级目标**（降级时
      工厂会补上 ``primary`` / ``fallback_reason`` / ``collection`` 三个标记，
      供 ``/health`` 如实报出"配置要求 milvus、实际用 memory"）；
    * **代价**：进程重启即失、多进程之间不可见 —— 只适合跑通链路；
    * ``persist_dir`` 非空时构造即尝试 ``_load()``、``persist()`` 时落盘 pickle。
    """

    name = "memory"

    def __init__(self, persist_dir=None) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._chunks: dict[str, Chunk] = {}
        if self.persist_dir is not None:
            self._load()

    # ---------- 内部 ----------
    @property
    def _pickle_path(self) -> Path | None:
        return None if self.persist_dir is None else self.persist_dir / DEFAULT_FILENAME

    def _load(self) -> None:
        path = self._pickle_path
        if path is None or not path.is_file():
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._chunks = {c.id: c for c in data if isinstance(c, Chunk)}
        except Exception:  # noqa: BLE001 - 索引损坏时按空库启动
            self._chunks = {}

    @staticmethod
    def _match(chunk: Chunk, where: dict | None) -> bool:
        if not where:
            return True
        for key, expected in where.items():
            actual = getattr(chunk, key, None)
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    # ---------- VectorStore 接口 ----------
    def upsert(self, chunks: list[Chunk]) -> int:
        """按 id 覆盖写入（同 id 重复 upsert 天然幂等）。"""
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        return len(chunks)

    def search(self, vector: list[float], top_k: int = 10,
               where: dict | None = None) -> list[SearchHit]:
        """暴力全量打分：余弦相似度降序取 top_k。

        没有向量的 chunk 直接跳过（否则余弦无意义）；``vector_score`` 与
        ``score`` 同值（内存库没有重排分数，两者本就同源）。
        """
        hits: list[SearchHit] = []
        for chunk in self._chunks.values():
            if not self._match(chunk, where):
                continue
            if not chunk.vector:
                continue
            score = cosine_similarity(vector, chunk.vector)
            hits.append(SearchHit(chunk=chunk, score=score, vector_score=score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def get(self, chunk_id: str) -> Chunk | None:
        """按 id 取单条；不存在返回 ``None``。"""
        return self._chunks.get(chunk_id)

    def all_chunks(self, where: dict | None = None) -> list[Chunk]:
        """按 ``where`` 过滤后返回全部 chunk（BM25 建索引用）。"""
        return [c for c in self._chunks.values() if self._match(c, where)]

    def delete(self, ids: list[str]) -> int:
        """按 id 删除；返回**真的删掉了**的条数（不存在的 id 不计入）。"""
        removed = 0
        for chunk_id in ids:
            if self._chunks.pop(chunk_id, None) is not None:
                removed += 1
        return removed

    def delete_by_role(self, role_id: str) -> int:
        """删掉该 role 分区的全部 chunk（先删后插的第一步）。"""
        targets = [c.id for c in self._chunks.values() if c.role_id == role_id]
        return self.delete(targets)

    def count(self) -> int:
        """活条数（内存库没有墓碑，就是 dict 长度）。"""
        return len(self._chunks)

    def persist(self) -> None:
        """把全量 chunk 落盘成 pickle；未配置 ``persist_dir`` 时是空操作。"""
        path = self._pickle_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(list(self._chunks.values()), f)
