"""BM25 稀疏检索（混合检索的稀疏路 / 多路召回之一）

在客户端用纯 Python 实现 Okapi BM25，作为 Milvus 向量检索之外的第二路召回。
选用客户端 BM25 而非 Milvus 内建 BM25 Function 的原因：
  1. Milvus 默认分词器对中文按空格切分，召回差，而这里可以做中文字符二元组分词；
  2. 避免改动 Milvus 集合 schema / 触发重建，保证现有知识库「跑通」不受影响；
  3. 数据量（本项目 KB 规模）完全在内存可承受范围内。
"""
import re
import math
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

from src.utils.logger import logger


# 中文字符区间（含扩展 A 区常用字）
_CJK_RE = re.compile(r'[一-鿿]')
_WORD_RE = re.compile(r'[a-z0-9]+')


class BM25Index:
    """内存 BM25 索引，支持中英混合分词"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.df: Counter = Counter()      # term -> document frequency
        self.total_docs: int = 0

    # ---- 分词 ----
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """中英混合分词：英文/数字按词，中文按字符二元组（末尾补单字）。"""
        text = (text or '').lower()
        tokens: List[str] = []
        for m in re.finditer(r'[a-z0-9]+|[一-鿿]+', text):
            tok = m.group()
            if _WORD_RE.fullmatch(tok):
                tokens.append(tok)
            else:
                # 中文：字符二元组 + 单个字兜底，保证单字查询也能命中
                if len(tok) == 1:
                    tokens.append(tok)
                else:
                    for i in range(len(tok) - 1):
                        tokens.append(tok[i:i + 2])
                    tokens.append(tok[-1])
        return tokens

    # ---- 构建 ----
    def build(self, docs: List[Dict[str, Any]]):
        """docs: [{chunk_id, text, source, page_number, ...}]"""
        self.docs = docs
        self.doc_tokens = [self.tokenize(d.get('text', '')) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.total_docs = len(docs)
        self.avgdl = (sum(self.doc_len) / self.total_docs) if self.total_docs else 0.0

        df: Counter = Counter()
        for toks in self.doc_tokens:
            for term in set(toks):
                df[term] += 1
        self.df = df
        logger.info(f"BM25 索引构建完成：{self.total_docs} 篇文档")

    @property
    def is_built(self) -> bool:
        return self.total_docs > 0

    # ---- 检索 ----
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """返回 [{chunk_id, text, source, page_number, bm25_score, source_type='bm25'}]"""
        if not self.is_built:
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        q_tf = Counter(q_tokens)
        scores: List[float] = []
        for i, toks in enumerate(self.doc_tokens):
            tf = Counter(toks)
            dl = self.doc_len[i]
            score = 0.0
            for term, qtf in q_tf.items():
                f = tf.get(term, 0)
                if f == 0:
                    continue
                df = self.df.get(term, 0)
                # 平滑 IDF
                idf = math.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (f * (self.k1 + 1)) / denom * qtf
            scores.append(score)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            doc = dict(self.docs[i])
            doc['bm25_score'] = scores[i]
            doc['source_type'] = 'bm25'
            results.append(doc)
        return results
