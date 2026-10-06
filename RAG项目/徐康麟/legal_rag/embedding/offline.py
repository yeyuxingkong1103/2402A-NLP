# -*- coding: utf-8 -*-
"""零依赖兜底向量化：字符 n-gram + 词元的哈希 TF 向量。

不依赖 torch / FlagEmbedding / 网络，纯标准库实现。
用于在没有 GPU、没有模型权重的情况下把整条链路先跑通，
也作为单元测试与离线模式的默认后端。

特性：
  * 确定性：同样的文本永远得到同样的向量；
  * 中文友好：以「单字 + 相邻双字」为词元，无需分词器；
  * 固定维度：哈希到 dim 维，带符号以降低哈希碰撞的偏置；
  * 次线性 TF：1 + log(tf)，抑制高频词元；
  * L2 归一化：使余弦相似度等价于点积。
"""
from __future__ import annotations

import hashlib
import math
import re

from .base import Embedder

_TOKEN_RE = re.compile(r"[0-9A-Za-z_]+")


class OfflineEmbedder(Embedder):
    name = "offline-hash-tf"

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正整数")
        self.dim = dim

    # ---------- 词元化 ----------
    @staticmethod
    def _terms(text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        terms: list[str] = [w.lower() for w in _TOKEN_RE.findall(text)]
        chars = [c for c in text if not c.isspace()]
        terms.extend(chars)                                        # 单字
        terms.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))  # 双字
        return terms

    # ---------- 哈希 ----------
    def _bucket(self, term: str) -> tuple[int, float]:
        digest = hashlib.md5(term.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return index, sign

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        terms = self._terms(text)
        if not terms:
            return vector

        counts: dict[str, int] = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1

        for term, count in counts.items():
            index, sign = self._bucket(term)
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0.0:
            vector = [v / norm for v in vector]
        return vector

    # ---------- Embedder 接口 ----------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
