# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""本地向量与 BM25 索引：轻量自实现，不依赖 工单7 的 AppConfig。

  * :func:`build_index`  —— 块表 → bge-m3 向量（1024 维）+ FAISS + BM25 倒排
  * :func:`load_index`   —— 读回，供 RAG 基线与 LightRAG 局部检索共用
  * :class:`VectorIndex` —— 向量检索（FAISS 缺失时自动降级为 numpy 全量点积）
  * :class:`BM25Index`   —— jieba 分词 + 标准 BM25（k1/b 可配）

所有阈值（top_k / 相似度下限 / k1 / b）来自配置。
"""
from __future__ import annotations

import json
import math
import pickle
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Cfg
from .corpus import Chunk
from .logging_util import get_logger

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+[\d,\.]*%?")


def tokenize(text: str, *, use_jieba: bool = True) -> list[str]:
    """中文分词：优先 jieba，缺失时退回「单字 + 英文词 + 数字」。"""
    if use_jieba:
        try:
            import jieba

            return [t for t in jieba.lcut(text) if t.strip()]
        except Exception:  # noqa: BLE001 - jieba 不可用不应阻断
            pass
    return _TOKEN_RE.findall(text)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行 L2 归一化；零向量保持零。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class BM25Index:
    """标准 BM25（Okapi）。"""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, use_jieba: bool = True) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.use_jieba = bool(use_jieba)
        self.ids: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freq: Counter[str] = Counter()
        self.idf: dict[str, float] = {}
        self.avg_len: float = 0.0
        self._tf: list[Counter[str]] = []

    def build(self, chunks: Sequence[Chunk], *, extra_text: dict[str, str] | None = None) -> "BM25Index":
        self.ids = [c.chunk_id for c in chunks]
        self.doc_tokens = []
        for chunk in chunks:
            text = chunk.content
            if extra_text and chunk.chunk_id in extra_text:
                text = text + "\n" + extra_text[chunk.chunk_id]
            self.doc_tokens.append(tokenize(text, use_jieba=self.use_jieba))
        self._tf = [Counter(toks) for toks in self.doc_tokens]
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        self.doc_freq = df
        total = len(self.doc_tokens)
        # BM25 idf（带 +1 平滑，避免负值）
        self.idf = {term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
        self.avg_len = (sum(len(t) for t in self.doc_tokens) / total) if total else 0.0
        return self

    def search(self, query: str, top_k: int = 12) -> list[tuple[str, float]]:
        qtokens = tokenize(query, use_jieba=self.use_jieba)
        if not qtokens or not self.ids:
            return []
        scores = np.zeros(len(self.ids), dtype=np.float64)
        for term in qtokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, tf in enumerate(self._tf):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                dl = len(self.doc_tokens[idx]) or 1
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                scores[idx] += idf * freq * (self.k1 + 1) / denom
        order = np.argsort(-scores)[:top_k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fh:
            pickle.dump({"ids": self.ids, "tf": self._tf, "idf": self.idf,
                         "avg_len": self.avg_len, "k1": self.k1, "b": self.b,
                         "use_jieba": self.use_jieba,
                         "doc_len": [len(t) for t in self.doc_tokens]}, fh)
        return out

    @classmethod
    def load(cls, path: Path | str) -> "BM25Index":
        with Path(path).open("rb") as fh:
            data = pickle.load(fh)
        obj = cls(k1=data["k1"], b=data["b"], use_jieba=data["use_jieba"])
        obj.ids = list(data["ids"])
        obj._tf = list(data["tf"])
        obj.idf = dict(data["idf"])
        obj.avg_len = float(data["avg_len"])
        obj.doc_tokens = [[] for _ in obj.ids]
        obj.doc_freq = Counter()
        return obj

    @property
    def size(self) -> int:
        return len(self.ids)


@dataclass
class VectorIndex:
    """向量索引：FAISS 加速，缺失则 numpy 全量点积。"""

    ids: list[str]
    vectors: np.ndarray
    dim: int
    model: str
    use_faiss: bool = False
    _faiss: Any = field(default=None, repr=False)

    def search(self, query_vec: Sequence[float], top_k: int = 12) -> list[tuple[str, float]]:
        if not self.ids or self.vectors.size == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        q = l2_normalize(q)
        k = min(int(top_k), len(self.ids))
        if self._faiss is not None:
            scores, idxs = self._faiss.search(q, k)
            return [(self.ids[int(i)], float(s)) for s, i in zip(scores[0], idxs[0]) if int(i) >= 0]
        sims = (self.vectors @ q[0])
        order = np.argsort(-sims)[:k]
        return [(self.ids[int(i)], float(sims[int(i)])) for i in order]

    def to_dict(self) -> dict[str, Any]:
        return {"ids": len(self.ids), "dim": self.dim, "model": self.model,
                "use_faiss": self.use_faiss, "vectors": [int(x) for x in self.vectors.shape]}


def build_index(chunks: Sequence[Chunk], *, cfg: Cfg, client: Any, logger: Any = None) -> dict[str, Any]:
    """构建并落盘 向量索引 + BM25。返回真实统计（含耗时与 token）。"""
    log = logger or get_logger("index", cfg)
    index_dir = cfg.path("index_dir")
    stats: dict[str, Any] = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    texts = [c.content for c in chunks]
    t0 = time.perf_counter()
    vectors = np.asarray(client.embed_texts(texts), dtype=np.float32)
    stats["embed_seconds"] = round(time.perf_counter() - t0, 3)
    stats["embed_calls"] = int(client.stats.calls)
    stats["embed_chunks"] = len(texts)
    stats["embed_dim"] = int(vectors.shape[1]) if vectors.size else 0
    vectors = l2_normalize(vectors)

    ids = [c.chunk_id for c in chunks]
    np.save(index_dir / "vectors.npy", vectors)
    (index_dir / "ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")

    use_faiss = False
    try:
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(index_dir / "faiss.index"))
        use_faiss = True
    except Exception as exc:  # noqa: BLE001 - FAISS 缺失时降级
        log.warning("FAISS 不可用，降级为 numpy 全量检索：%s: %s", type(exc).__name__, exc)
        (index_dir / "faiss.index").unlink(missing_ok=True)
    stats["use_faiss"] = use_faiss

    t1 = time.perf_counter()
    bm25 = BM25Index(k1=float(cfg.get("retrieval", {}).get("bm25_k1", 1.5)),
                     b=float(cfg.get("retrieval", {}).get("bm25_b", 0.75)))
    bm25.build(chunks)
    bm25.save(index_dir / "bm25.pkl")
    stats["bm25_seconds"] = round(time.perf_counter() - t1, 3)
    stats["bm25_docs"] = bm25.size
    stats["bm25_vocab"] = len(bm25.idf)
    stats["chunk_total"] = len(chunks)
    stats["model"] = str(cfg.llm.embed_model)
    stats["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    (index_dir / "index_manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def index_exists(cfg: Cfg) -> bool:
    d = Path(cfg.path_str("index_dir"))
    return all((d / name).exists() for name in ("vectors.npy", "ids.json", "bm25.pkl"))


def load_index(cfg: Cfg, *, logger: Any = None) -> tuple[VectorIndex, BM25Index]:
    """读回向量索引与 BM25。缺失时抛 FileNotFoundError（调用方决定是否重建）。"""
    log = logger or get_logger("index", cfg)
    d = Path(cfg.path_str("index_dir"))
    vectors = np.load(d / "vectors.npy")
    ids = json.loads((d / "ids.json").read_text(encoding="utf-8"))
    use_faiss = False
    faiss_index = None
    if (d / "faiss.index").exists():
        try:
            import faiss

            faiss_index = faiss.read_index(str(d / "faiss.index"))
            use_faiss = True
        except Exception as exc:  # noqa: BLE001
            log.warning("FAISS 读取失败，降级 numpy：%s: %s", type(exc).__name__, exc)
    vindex = VectorIndex(ids=ids, vectors=vectors, dim=int(vectors.shape[1]) if vectors.size else 0,
                         model=str(cfg.llm.embed_model), use_faiss=use_faiss, _faiss=faiss_index)
    bm25 = BM25Index.load(d / "bm25.pkl")
    return vindex, bm25
