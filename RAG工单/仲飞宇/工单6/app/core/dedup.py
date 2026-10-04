# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""
去重：内容哈希（精确）+ SimHash（近似）。

【为什么必须做】
实测招股说明书1.pdf 里「重要供应商」相关内容重复出现在 **11 个页面**上，
「科技进步一等奖」重复 **9 页**，而且措辞还不完全一致
（p181 写「国防用户第一个」，p182 写「全军第一个」，两处表述冲突）。

不去重的后果不是"浪费空间"，而是**检索质量塌方**：
top-3 名额被同一段话的 3 个副本占满，LLM 拿到的上下文全是重复内容，
真正能回答问题的那个片段被挤出去；更糟的是副本之间措辞冲突时，
模型会无所适从或随机挑一个说法输出。

【为什么不用现成的 simhash 包】
只需要 64 位 SimHash + 海明距离，自己实现约 30 行，
省一个依赖，也便于按中文分词（jieba）定制 —— 直接按字符做 SimHash
对中文长文本的区分度太差。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import jieba

from app.config import settings

_HASH_BITS = 64


# ----------------------------------------------------------------------
# 精确哈希
# ----------------------------------------------------------------------
def content_hash(text: str) -> str:
    """归一化后的 SHA-256 前 32 位，用于精确去重与 Milvus 字段。"""
    norm = "".join(text.split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


# ----------------------------------------------------------------------
# SimHash
# ----------------------------------------------------------------------
def _token_weights(text: str) -> dict[str, int]:
    """
    分词并统计词频作为权重。

    【为什么权重用词频而不是 1】招股书里「发行人」「公司」这类高频词
    对区分度没有贡献，但在 64 维指纹里会被反复投票，反而拉近不相关文本的
    距离。用词频加权后，长文档的指纹会偏向其真正有信息量的词。
    """
    counts: dict[str, int] = {}
    for tok in jieba.cut(text):
        tok = tok.strip()
        if len(tok) < 2 and not tok.isdigit():
            continue          # 单字（的/了/和）噪声大，丢弃
        counts[tok] = counts.get(tok, 0) + 1
    return counts


def simhash(text: str, bits: int = _HASH_BITS) -> int:
    """计算 64 位 SimHash 指纹。"""
    weights = _token_weights(text)
    if not weights:
        return 0

    vector = [0] * bits
    for token, w in weights.items():
        h = int.from_bytes(
            hashlib.md5(token.encode("utf-8")).digest()[: bits // 8], "big"
        )
        for i in range(bits):
            vector[i] += w if (h >> i) & 1 else -w

    out = 0
    for i in range(bits):
        if vector[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    """海明距离。"""
    return bin(a ^ b).count("1")


# ----------------------------------------------------------------------
# 去重
# ----------------------------------------------------------------------
@dataclass
class DedupResult:
    kept: list
    dropped: list
    n_exact: int = 0
    n_near: int = 0

    @property
    def drop_rate(self) -> float:
        total = len(self.kept) + len(self.dropped)
        return len(self.dropped) / total if total else 0.0


class SimHashDeduper:
    """
    两级去重：
      1. 精确：归一化 SHA-256 相同 → 直接丢
      2. 近似：SimHash 海明距离 ≤ 阈值 → 丢后来的，保留先出现的

    【为什么保留先出现的】招股书的正文（第二节以后）比后面的重复附录
    更完整、页码更靠前，引用时更自然。
    """

    def __init__(self, max_distance: int | None = None) -> None:
        self.max_distance = (settings.simhash_max_distance
                             if max_distance is None else max_distance)
        self._exact: dict[str, int] = {}          # hash → kept 下标
        self._fps: list[tuple[int, int]] = []     # (simhash, kept 下标)

    def add(self, chunk) -> tuple[bool, str]:
        """返回 (是否保留, 判定理由)。"""
        h = content_hash(chunk.content)
        if h in self._exact:
            return False, "exact"

        fp = simhash(chunk.content)
        for other_fp, _other_idx in self._fps:
            if hamming(fp, other_fp) <= self.max_distance:
                return False, "near"

        self._exact[h] = len(self._fps)
        self._fps.append((fp, len(self._fps)))
        return True, "kept"


def dedup_chunks(chunks: Sequence, max_distance: int | None = None) -> DedupResult:
    """
    【注意：跳过表格】两张不同的表格可能结构高度相似（都是键值对、词也接近），
    SimHash 容易误判为重复 —— 而它们的数值恰恰是答案所在。
    实测 p21 的「发行人基本情况」和 p22 的「本次发行概况」就属于这种情况。
    因此表格只做**精确**去重，不做近似去重。
    """
    d = SimHashDeduper(max_distance=max_distance)
    kept: list = []
    dropped: list = []
    n_exact = n_near = 0

    for c in chunks:
        is_table = getattr(c, "chunk_type", "text") == "table"
        if is_table:
            h = content_hash(c.content)
            if h in d._exact:
                dropped.append(c)
                n_exact += 1
                continue
            d._exact[h] = len(kept)
            kept.append(c)
            continue

        ok, why = d.add(c)
        if ok:
            kept.append(c)
        else:
            dropped.append(c)
            if why == "exact":
                n_exact += 1
            else:
                n_near += 1

    return DedupResult(kept=kept, dropped=dropped, n_exact=n_exact, n_near=n_near)
