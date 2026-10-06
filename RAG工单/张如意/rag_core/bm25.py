# -*- coding: utf-8 -*-
"""
全文检索模块（倒排索引 + BM25）
工单编号：人工智能NLP-RAG-混合检索任务

工单要求：使用倒排索引技术实现全文检索，支持
  1. 布尔查询    —— "军用 AND 收入"、"军用 OR 民用"、"军用 NOT 民用"
  2. 短语匹配    —— 加引号 "军用领域" 精确短语
  3. 模糊匹配    —— 通配符 军用* / 编辑距离近似
  4. 多字段检索  —— 标题(section) / 正文(text) / 摘要 分别加权

中文分词优先用 jieba；环境无 jieba 时降级为「单字 + 二元组」切分。
"""
from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import config
from .chunk import Chunk

try:
    import jieba
    jieba.setLogLevel(20)
    _HAS_JIEBA = True
except ImportError:                                     # pragma: no cover
    _HAS_JIEBA = False


# ---------------------------------------------------------------------------
# 分词
# ---------------------------------------------------------------------------
_STOPWORDS = set(
    "的 了 是 在 和 与 或 及 有 为 对 上 中 下 个 我 你 他 她 它 这 那 请问 多少 哪些 哪个 什么 如何 怎么 什么".split()
)


def tokenize(text: str, for_query: bool = False) -> list[str]:
    """
    中英文混合分词。
    for_query=True 时用搜索引擎模式，粒度更细，提升召回。
    """
    if not text:
        return []
    if _HAS_JIEBA:
        toks = jieba.lcut_for_search(text) if for_query else jieba.lcut(text)
    else:
        # 降级：中文按单字+二元组，英文/数字按词
        toks = []
        for seg in re.findall(r"[A-Za-z0-9]+|[一-鿿]+", text):
            if seg.isascii():
                toks.append(seg.lower())
            else:
                toks.extend(list(seg))
                toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return [t.strip().lower() for t in toks if t.strip() and t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# BM25 检索器
# ---------------------------------------------------------------------------
class BM25Retriever:
    """
    Okapi BM25 全文检索器（自实现倒排索引，不依赖 Elasticsearch）。

    倒排索引结构： term -> {doc_idx: tf}
    另存 doc_len / avgdl 用于 BM25 长度归一化。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75,
                 field_weights: dict[str, float] | None = None):
        self.k1 = k1
        self.b = b
        # 多字段检索：正文权重最高，章节路径次之（工单06 要求）
        self.field_weights = field_weights or {"text": 1.0, "section": 1.6, "title": 2.0}

        self.doc_ids: list[str] = []
        self.docs: list[dict] = []
        self.inverted: dict[str, dict[int, float]] = defaultdict(dict)  # term -> {doc: 加权tf}
        self.doc_len: list[float] = []
        self.avgdl: float = 0.0

    # -- 建索引 -------------------------------------------------------------
    def build(self, chunks: list[Chunk]) -> int:
        self.doc_ids, self.docs = [], []
        self.inverted = defaultdict(dict)
        self.doc_len = []

        for c in chunks:
            if not c.text.strip():
                continue
            idx = len(self.doc_ids)
            self.doc_ids.append(c.chunk_id)
            self.docs.append(c.to_dict())

            tf: Counter[str] = Counter()
            total = 0.0
            for field, weight in self.field_weights.items():
                val = getattr(c, field, "") or ""
                if not val:
                    continue
                toks = tokenize(val)
                for t in toks:
                    tf[t] += weight
                total += len(toks) * weight

            for t, w in tf.items():
                self.inverted[t][idx] = w
            self.doc_len.append(total or 1.0)

        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0
        return len(self.doc_ids)

    # -- BM25 打分 ----------------------------------------------------------
    def _idf(self, term: str) -> float:
        n = len(self.inverted.get(term, {}))
        N = len(self.doc_ids)
        if N == 0:
            return 0.0
        return math.log(1 + (N - n + 0.5) / (n + 0.5))

    def _bm25(self, term: str, doc_idx: int) -> float:
        f = self.inverted.get(term, {}).get(doc_idx)
        if not f:
            return 0.0
        dl = self.doc_len[doc_idx]
        return self._idf(term) * f * (self.k1 + 1) / (
            f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        )

    def search(self, query: str, top_k: int = config.TOP_K_RECALL,
               boolean: str | None = None) -> list[dict]:
        """
        全文检索。

        Args:
            boolean: None=普通 BM25；"AND"/"OR"/"NOT" 走布尔查询分支
        """
        if boolean:
            return self._boolean_search(query, boolean, top_k)

        scores: dict[int, float] = defaultdict(float)
        for t in tokenize(query, for_query=True):
            for doc_idx in self.inverted.get(t, {}):
                scores[doc_idx] += self._bm25(t, doc_idx)

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [{**self.docs[i], "score": s, "source": "bm25"} for i, s in ranked]

    # -- 布尔查询 -----------------------------------------------------------
    def _boolean_search(self, query: str, op: str, top_k: int) -> list[dict]:
        terms = [t for t in tokenize(query, for_query=True)]
        if not terms:
            return []
        sets = [set(self.inverted.get(t, {})) for t in terms]

        if op.upper() == "AND":
            hit = set.intersection(*sets) if sets else set()
        elif op.upper() == "OR":
            hit = set.union(*sets) if sets else set()
        elif op.upper() == "NOT":
            base, excl = sets[0], set.union(*sets[1:]) if len(sets) > 1 else set()
            hit = base - excl
        else:
            hit = set.union(*sets) if sets else set()

        scored = sorted(
            ((i, sum(self._bm25(t, i) for t in terms)) for i in hit),
            key=lambda x: -x[1],
        )[:top_k]
        return [{**self.docs[i], "score": s, "source": f"bm25:{op}"} for i, s in scored]

    # -- 短语匹配 -----------------------------------------------------------
    def phrase_search(self, phrase: str, top_k: int = config.TOP_K_RECALL) -> list[dict]:
        """精确子串匹配（对招股书里的专有名词、数字很有效）。"""
        p = phrase.strip().strip('"“”')
        hits = [
            (i, d) for i, d in enumerate(self.docs)
            if p and p in d.get("text", "")
        ]
        out = []
        for i, d in hits[:top_k]:
            tf = d["text"].count(p)
            out.append({**d, "score": float(tf), "source": "phrase"})
        return sorted(out, key=lambda x: -x["score"])

    # -- 模糊匹配 -----------------------------------------------------------
    def fuzzy_search(self, pattern: str, top_k: int = config.TOP_K_RECALL,
                     max_dist: int = 1) -> list[dict]:
        """
        通配符 + 编辑距离模糊匹配。
        pattern 支持 `*`（如 军用*），无通配符时按编辑距离<=max_dist 近似。
        """
        if "*" in pattern:
            rx = re.compile("^" + re.escape(pattern).replace(r"\*", ".*") + "$")
            cands = [t for t in self.inverted if rx.match(t)]
        else:
            cands = [
                t for t in self.inverted
                if abs(len(t) - len(pattern)) <= max_dist
                and _edit_distance(t, pattern) <= max_dist
            ]

        scores: dict[int, float] = defaultdict(float)
        for t in cands:
            for doc_idx in self.inverted[t]:
                scores[doc_idx] += self._bm25(t, doc_idx)
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [{**self.docs[i], "score": s, "source": "fuzzy"} for i, s in ranked]

    # -- 持久化 -------------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = path or (config.INDEX_DIR / "bm25.pkl")
        with Path(path).open("wb") as f:
            pickle.dump({
                "k1": self.k1, "b": self.b, "field_weights": self.field_weights,
                "doc_ids": self.doc_ids, "docs": self.docs,
                "inverted": dict(self.inverted), "doc_len": self.doc_len,
                "avgdl": self.avgdl,
            }, f)
        return Path(path)

    @classmethod
    def load(cls, path: Path | None = None) -> "BM25Retriever":
        path = path or (config.INDEX_DIR / "bm25.pkl")
        with Path(path).open("rb") as f:
            o = pickle.load(f)
        r = cls(o["k1"], o["b"], o["field_weights"])
        r.doc_ids, r.docs = o["doc_ids"], o["docs"]
        r.inverted = defaultdict(dict, o["inverted"])
        r.doc_len, r.avgdl = o["doc_len"], o["avgdl"]
        return r


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein 距离（滚动数组，O(min(m,n)) 空间）。"""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
