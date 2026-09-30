# -*- coding: utf-8 -*-
"""retrieval/bm25.py —— jieba 分词与 BM25 关键词打分。

在链路中的位置：
    关键词召回路的底层实现：retrieval/recall.py 的 keyword_hits 调它给全语料打分。

为什么要 BM25（而不是只靠向量）：
    向量模型对"语义相近"很强，但对"字面完全相同却语义不相似"的词反而弱 ——
    典型就是 "GB/T 44653-2024" 这类标准号和"净化处理装置"这类专有名词。
    用户明确输入标准号时期望的是字面命中条款，BM25 正好补上这一路。
"""
from __future__ import annotations

import math
import re

import jieba

from .config import STOP_WORDS, TOKEN_RE

def tokenize(text: str) -> list[str]:
    """把文本切成 BM25 用的关键词。

    参数：
        text: 待分词的查询或文档文本
    返回：
        归一化为小写的关键词列表。

    过滤掉的四类：
        1. 停用词（的/是/在…）——没有区分度
        2. 不含字母数字汉字的碎片（纯标点）
        3. 单字且非字母数字 —— 中文单字歧义太大，做检索键只会引入噪声
        4. 纯数字（页码、年份、条款号）—— 这类数字在几乎每页都出现，命中它们没有信息量
    """
    tokens = []
    for part in jieba.lcut(text or ""):
        part = part.strip()
        if not part or part in STOP_WORDS or not TOKEN_RE.fullmatch(part):
            continue
        if len(part) < 2 and not part.isalnum():
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", part):
            continue
        tokens.append(part.lower())
    return tokens

class BM25:
    """用于补足标准编号、术语等精确字面匹配。

    为什么在向量检索之外还要 BM25：
        向量模型对"语义相近"很强，但对"字面完全相同但语义不相似"的词反而弱 ——
        典型的就是 "GB/T 44653-2024" 这种标准号、以及"净化处理装置"这种专有名词。
        用户明确输入标准号时，他期望的就是字面命中那些条款，而不是语义近邻。
        BM25 正好补上这一路：它只看词是否原样出现。

    打分考虑了四件事（这也是 BM25 相对朴素词频统计的优势）：
        词是否出现在该文档、在该文档出现多少次、该词在整个语料中有多稀有、文档有多长。
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        """建立 BM25 索引。

        参数：
            texts: 全部候选文档（这里就是知识库里所有 chunk 的原文）
            k1: 词频饱和系数，1.5 是经典默认值 —— 一个词出现 10 次不该比出现 5 次强一倍，
                所以词频的贡献要饱和
            b: 文档长度归一化系数，0.75 是经典默认值 —— 长文档天然容易命中更多词，
                要用平均长度做惩罚，否则长 chunk 永远占优
        """
        self.k1, self.b = k1, b
        self.docs = [tokenize(text) for text in texts]
        self.total = len(self.docs)
        self.average_length = sum(map(len, self.docs)) / max(self.total, 1)  # max 防 0 除
        # 文档频率：每个词出现在多少篇文档里（用 set(tokens) 去重，同一篇里出现多次只算一次）
        self.document_frequency: dict[str, int] = {}
        for tokens in self.docs:
            for token in set(tokens):
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1

    def score(self, query: str, index: int) -> float:
        """算某个文档相对查询的 BM25 分。

        参数：
            query: 用户查询（已改写后的检索词串）
            index: 文档在语料中的下标
        返回：
            BM25 分数，越高越相关。

        公式（标准 BM25，未做改动）：
            score = Σ idf(t) * tf(t) * (k1 + 1) / (tf(t) + k1 * (1 - b + b * len/avg_len))
            idf(t) = log((N - df + 0.5) / (df + 0.5) + 1)
        加 1 是为了让 idf 恒为正：df > N/2 的常见词也不会变成负贡献。
        """
        query_tokens = tokenize(query)
        document = self.docs[index]
        frequency: dict[str, int] = {}
        for token in document:
            frequency[token] = frequency.get(token, 0) + 1
        length = len(document)
        score = 0.0
        for token in query_tokens:
            count = frequency.get(token, 0)
            if not count:  # 该文档没出现过这个词，跳过（BM25 无贡献）
                continue
            df = self.document_frequency.get(token, 0)
            idf = math.log((self.total - df + 0.5) / (df + 0.5) + 1)
            denominator = count + self.k1 * (1 - self.b + self.b * length / max(self.average_length, 1))
            score += idf * count * (self.k1 + 1) / denominator
        return score
