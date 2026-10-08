# -*- coding: utf-8 -*-
"""SQLite 离线向量库（OfflineStore）。

用标准库 ``sqlite3`` 在**单个本地文件**里存储「文本分块 + 归一化向量 + 元数据」，
提供向量（余弦）、关键词（BM25）、混合（RRF）三路检索，无需任何服务器，
是 ``hybrid_retriever``（Milvus + BM25）的离线替代。

设计要点：

- 向量以 float32 字节序列存为 BLOB，入库时归一化，检索时点积即余弦相似度；
- 关键词路复用 ``hybrid_retriever.BM25Index``（jieba 分词，纯内存，检索时现建）；
- 混合路复用 ``weighted_rrf`` 融合，与在线版行为一致；
- 单文件持久化，支持 ``with OfflineStore(...) as db``，内部线程锁串行访问。

用法示例（详见 README）：

    from rag2 import OfflineStore

    db = OfflineStore("kb.sqlite")
    db.add(texts=["……"], vectors=[[0.1, 0.2, ...]], metadatas=[{"source": "a.pdf"}])
    hits = db.search("问题", top_k=3, embed_fn=embed, mode="hybrid")
    for h in hits:
        print(h.chunk_id, h.score, h.text)
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import struct
import threading
import uuid
from pathlib import Path
from typing import Any, Sequence

from .hybrid_retriever import Hit, BM25Index, weighted_rrf

logger = logging.getLogger("rag2.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT UNIQUE,
    doc_id TEXT,
    source TEXT,
    page INTEGER,
    text TEXT,
    meta TEXT,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


def _normalize(vec: Sequence[float]) -> list[float]:
    """向量 L2 归一化（零向量返回原样，避免除零）。

    归一化后每个向量的模长（长度）都为 1，此时两个向量的「点积」恰好等于它们的
    「余弦相似度」（值域 [-1,1]）。这样检索只比较方向（语义），不受文本长短影响。
    """
    v = [float(x) for x in vec]
    n = math.sqrt(sum(x * x for x in v)) or 1.0  # 模长 = sqrt(各分量平方和)，0 向量兜底为 1
    return [x / n for x in v]                     # 每个分量除以模长 → 模长变为 1


def _pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _row_to_hit(row: sqlite3.Row, score: float = 0.0, routes: dict | None = None) -> Hit:
    meta = {}
    try:
        meta = json.loads(row["meta"] or "{}")
    except Exception:  # noqa: BLE001 - 脏数据兜底
        meta = {}
    return Hit(
        chunk_id=row["chunk_id"],
        text=row["text"],
        score=score,
        doc_id=row["doc_id"],
        source=row["source"],
        page=int(row["page"] or 0),
        routes=dict(routes or {}),
        extra=meta,
    )


class OfflineStore:
    """SQLite 离线向量库（单文件、零服务器）。"""

    def __init__(self, db_path: str | Path, dim: int | None = None) -> None:
        """初始化（不立即建库，首次访问时才创建 .sqlite 文件）。

        参数：
            db_path: 数据库文件路径（.sqlite / .db）。
            dim:     向量维度（仅作记录，检索时按 BLOB 长度自动推断）。
        """
        self.db_path = Path(db_path)
        self.dim = int(dim) if dim else None
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 连接
    @property
    def conn(self) -> sqlite3.Connection:
        """懒打开连接并建表（幂等）。"""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            logger.info("打开离线库：%s", self.db_path)
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ------------------------------------------------------------------ 写入
    def add(
        self,
        texts: Sequence[str],
        vectors: Sequence[Sequence[float]] | None = None,
        metadatas: Sequence[dict] | None = None,
        ids: Sequence[str] | None = None,
    ) -> int:
        """写入分块与向量（chunk_id 相同时覆盖，幂等）。

        参数：
            texts:     分块文本列表。
            vectors:   与 texts 同序的向量列表；缺省抛错（离线库不内置向量化）。
            metadatas: 元数据列表，支持 doc_id/source/page 及任意自定义键。
            ids:       chunk_id 列表；缺省自动生成。

        返回：
            写入条数。
        """
        texts = [str(t) for t in texts]
        if not texts:
            return 0
        if vectors is None:
            raise ValueError("离线库不内置向量化，请传入 vectors（可用 rag2.OfflineRAG.embed 生成）")
        vectors = [_normalize(v) for v in vectors]
        n = len(texts)
        if len(vectors) != n:
            raise ValueError(f"vectors 数量({len(vectors)})与 texts 数量({n})不一致")
        metadatas = list(metadatas) if metadatas is not None else [{} for _ in range(n)]
        if len(metadatas) != n:
            raise ValueError(f"metadatas 数量({len(metadatas)})与 texts 数量({n})不一致")
        ids = [str(x) for x in ids] if ids is not None else [uuid.uuid4().hex for _ in range(n)]

        with self._lock:
            conn = self.conn
            for i in range(n):
                m = metadatas[i] or {}
                blob = _pack(vectors[i])
                conn.execute(
                    "INSERT OR REPLACE INTO chunks"
                    "(chunk_id, doc_id, source, page, text, meta, embedding)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        ids[i],
                        str(m.get("doc_id", ""))[:250],
                        str(m.get("source", ""))[:500],
                        int(m.get("page", 0) or 0),
                        texts[i],
                        json.dumps(m, ensure_ascii=False),
                        blob,
                    ),
                )
            conn.commit()
        logger.info("离线库写入 %d 条", n)
        return n

    # ------------------------------------------------------------------ 检索
    def _all_rows(self) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                "SELECT chunk_id, doc_id, source, page, text, meta, embedding FROM chunks"
            ).fetchall())

    def search_vector(self, vector: Sequence[float], top_k: int = 10) -> list[Hit]:
        """向量（余弦）检索。"""
        vector = _normalize(vector)
        rows = self._all_rows()
        scored: list[tuple[float, sqlite3.Row]] = []
        for r in rows:
            emb = _unpack(r["embedding"])
            # 查询向量与库中向量都已归一化，点积（逐分量相乘再求和）即余弦相似度，越大越相似
            sim = sum(a * b for a, b in zip(vector, emb))
            scored.append((sim, r))
        scored.sort(key=lambda t: -t[0])
        return [_row_to_hit(r, score=s) for s, r in scored[:top_k]]

    def search_keyword(self, query: str, top_k: int = 10) -> list[Hit]:
        """关键词（BM25）检索。"""
        rows = self._all_rows()
        texts = [r["text"] for r in rows]
        bm25 = BM25Index()
        bm25.build(texts)
        hits: list[Hit] = []
        for idx, score in bm25.search(query, top_k):
            hits.append(_row_to_hit(rows[idx], score=score))
        return hits

    def search(
        self,
        query: str | Sequence[float],
        top_k: int = 10,
        embed_fn: Any = None,
        mode: str = "hybrid",
        weights: dict[str, float] | None = None,
    ) -> list[Hit]:
        """离线检索主入口。

        参数：
            query:    查询文本（mode 含 vector 时需 embed_fn）或查询向量。
            top_k:    返回条数。
            embed_fn: 向量化函数 (texts) -> vectors；query 为文本时必需。
            mode:     vector / keyword / hybrid。
            weights:  hybrid 融合权重 {"vector": 1.0, "keyword": 1.0}。

        返回：
            按分数降序排列的 Hit 列表。
        """
        if mode not in ("vector", "keyword", "hybrid"):
            raise ValueError(f"不支持的检索模式：{mode}（可选 vector/keyword/hybrid）")
        top_k = int(top_k)
        pool = top_k * 3 if mode == "hybrid" else top_k

        vhits: list[Hit] = []
        khits: list[Hit] = []
        if mode in ("vector", "hybrid"):
            if isinstance(query, str):
                if embed_fn is None:
                    raise ValueError("query 为文本时需提供 embed_fn")
                query = embed_fn([query])[0]
            vhits = self.search_vector(list(query), pool)
        if mode in ("keyword", "hybrid"):
            khits = self.search_keyword(str(query) if not isinstance(query, str) else query, pool)

        if mode == "vector":
            return vhits[:top_k]
        if mode == "keyword":
            return khits[:top_k]

        rankings = {"vector": [h.chunk_id for h in vhits], "keyword": [h.chunk_id for h in khits]}
        scores = {"vector": {h.chunk_id: h.score for h in vhits},
                  "keyword": {h.chunk_id: h.score for h in khits}}
        fused = weighted_rrf(rankings, weights=weights or {"vector": 1.0, "keyword": 1.0}, scores=scores)
        payload = {h.chunk_id: h for h in vhits + khits}
        result: list[Hit] = []
        for item in fused[:top_k]:
            h = payload.get(item.key)
            if h is None:
                continue
            result.append(Hit(chunk_id=h.chunk_id, text=h.text, score=item.score,
                              doc_id=h.doc_id, source=h.source, page=h.page,
                              routes=item.routes, extra=h.extra))
        return result

    # ------------------------------------------------------------------ 维护
    def get(self, chunk_id: str) -> Hit | None:
        """按 chunk_id 取单条。"""
        with self._lock:
            row = self.conn.execute(
                "SELECT chunk_id, doc_id, source, page, text, meta, embedding FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        return _row_to_hit(row) if row else None

    def count(self) -> int:
        with self._lock:
            return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def delete(self, chunk_ids: Sequence[str] | str) -> int:
        """按 chunk_id 删除，返回删除条数。"""
        ids = [chunk_ids] if isinstance(chunk_ids, str) else list(chunk_ids)
        if not ids:
            return 0
        with self._lock:
            cur = self.conn.executemany("DELETE FROM chunks WHERE chunk_id = ?", [(i,) for i in ids])
            self.conn.commit()
            return int(cur.rowcount or 0)

    def clear(self) -> None:
        """清空全部数据（保留表结构）。"""
        with self._lock:
            self.conn.execute("DELETE FROM chunks")
            self.conn.commit()

    # ------------------------------------------------------------------ 上下文
    def __enter__(self) -> "OfflineStore":
        _ = self.conn
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - 调试打印
        return f"OfflineStore({self.db_path}, 条数={self.count() if self._conn else '未打开'})"
