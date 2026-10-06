"""BM25 关键词索引。

工单要求：向量检索之外同时建立 BM25 关键词索引，构成混合检索。

实现说明：
- 优先使用 ``rank_bm25``（若已安装）；
- 未安装时使用本文件内置的 **纯 Python Okapi BM25** 实现，行为与
  ``rank_bm25.BM25Okapi`` 对齐（k1=1.5, b=0.75），保证离线环境同样可用。
- 索引可保存/加载（JSON），与向量库一起放在 ``data/index/``。
"""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable

from app.core.config import get_settings
from app.core.logging_conf import logger, trace
from app.core.text_utils import tokenize
from app.models.schemas import Chunk

try:  # pragma: no cover - 依赖可用性分支
    from rank_bm25 import BM25Okapi as _RankBM25

    HAS_RANK_BM25 = True
except Exception:  # pragma: no cover
    _RankBM25 = None
    HAS_RANK_BM25 = False


class PureBM25:
    """纯 Python 的 Okapi BM25 实现（rank_bm25 的等价替代）。"""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_count = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_length = (sum(self.doc_lengths) / self.doc_count) if self.doc_count else 0.0
        self.doc_freqs: list[Counter[str]] = [Counter(doc) for doc in corpus]
        # 词 -> 包含该词的文档数
        df: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        # IDF 采用 BM25 常用平滑形式，保证非负
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def get_scores(self, query: Iterable[str]) -> list[float]:
        """返回查询对每个文档的 BM25 分数。"""
        scores = [0.0] * self.doc_count
        if not self.doc_count or not self.avg_length:
            return scores
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, freqs in enumerate(self.doc_freqs):
                freq = freqs.get(term, 0)
                if not freq:
                    continue
                norm = 1 - self.b + self.b * (self.doc_lengths[index] / self.avg_length)
                scores[index] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)
        return scores

    def get_top_n(self, query: Iterable[str], n: int = 10) -> list[tuple[int, float]]:
        """返回 (文档下标, 分数) 的 top-n，按分数降序。"""
        scores = self.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [(index, score) for index, score in ranked[:n] if score > 0]


class BM25Index:
    """BM25 索引封装：负责分词、建索引、查询与持久化。"""

    def __init__(self, use_jieba: bool = True) -> None:
        self.use_jieba = use_jieba
        self.chunk_ids: list[str] = []
        self.corpus_tokens: list[list[str]] = []
        self._backend: PureBM25 | None = None
        self._rank_backend = None

    # ------------------------------------------------------------------
    # 构建与查询
    # ------------------------------------------------------------------
    @trace
    def build(self, chunks: list[Chunk]) -> None:
        """用分块列表重建索引。"""
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.corpus_tokens = [tokenize(chunk.content, self.use_jieba) for chunk in chunks]
        if not self.corpus_tokens:
            self._backend = None
            self._rank_backend = None
            logger.warning("app.core.bm25_index", "空语料，BM25 索引未建立")
            return
        if HAS_RANK_BM25:
            self._rank_backend = _RankBM25(self.corpus_tokens)
            self._backend = None
        else:
            self._rank_backend = None
            self._backend = PureBM25(self.corpus_tokens)
        logger.info(
            "app.core.bm25_index",
            "BM25 索引构建完成",
            documents=len(self.chunk_ids),
            backend="rank_bm25" if HAS_RANK_BM25 else "pure_python",
        )

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    @trace
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """检索，返回 ``(chunk_id, 归一化分数)`` 列表。

        分数归一化到 0~1，便于与向量分数加权融合。
        """
        if not self.chunk_ids or (self._backend is None and self._rank_backend is None):
            return []
        tokens = tokenize(query, self.use_jieba)
        if not tokens:
            return []

        if self._rank_backend is not None:
            scores = [float(score) for score in self._rank_backend.get_scores(tokens)]
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
            pairs = [(index, score) for index, score in ranked[:top_k] if score > 0]
        else:
            pairs = self._backend.get_top_n(tokens, top_k)  # type: ignore[union-attr]

        if not pairs:
            return []
        top = max(score for _, score in pairs) or 1.0
        return [(self.chunk_ids[index], score / top) for index, score in pairs]

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, path: Path | str | None = None) -> Path:
        """保存索引到磁盘（pickle，含分词结果）。"""
        settings = get_settings()
        target = Path(path) if path else settings.paths.data_index / "bm25_index.pkl"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_ids": self.chunk_ids,
            "corpus_tokens": self.corpus_tokens,
            "use_jieba": self.use_jieba,
        }
        with open(target, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # 同时导出一份可读的元信息，便于人工核对
        meta = {
            "documents": len(self.chunk_ids),
            "backend": "rank_bm25" if HAS_RANK_BM25 else "pure_python",
            "use_jieba": self.use_jieba,
            "avg_tokens": (sum(len(t) for t in self.corpus_tokens) / len(self.corpus_tokens)) if self.corpus_tokens else 0,
        }
        (target.with_suffix(".meta.json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("app.core.bm25_index", "BM25 索引已保存", path=str(target), documents=len(self.chunk_ids))
        return target

    @classmethod
    def load(cls, path: Path | str | None = None) -> "BM25Index":
        """从磁盘加载索引；文件不存在时返回空索引（不抛异常）。"""
        settings = get_settings()
        target = Path(path) if path else settings.paths.data_index / "bm25_index.pkl"
        index = cls()
        if not target.exists():
            logger.warning("app.core.bm25_index", "BM25 索引文件不存在，返回空索引", path=str(target))
            return index
        with open(target, "rb") as handle:
            payload = pickle.load(handle)
        index.chunk_ids = payload.get("chunk_ids", [])
        index.corpus_tokens = payload.get("corpus_tokens", [])
        index.use_jieba = payload.get("use_jieba", True)
        if index.corpus_tokens:
            if HAS_RANK_BM25:
                index._rank_backend = _RankBM25(index.corpus_tokens)
            else:
                index._backend = PureBM25(index.corpus_tokens)
        logger.info("app.core.bm25_index", "BM25 索引加载完成", path=str(target), documents=len(index.chunk_ids))
        return index
