# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
检索优化模块：
  - TfidfRetriever：基线（TF-IDF 余弦相似度）
  - BM25Retriever ：优化（BM25 评分，对中文长文档召回更稳）
"""
import math
import numpy as np

try:
    import jieba
except ImportError:
    jieba = None


def tokenize(text: str):
    text = text.lower()
    if jieba is not None:
        return [t.strip() for t in jieba.cut(text) if t.strip()]
    chars = [c for c in text if not c.isspace()]
    return chars + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


class TfidfRetriever:
    """基线检索器（与工单01一致）。"""

    def __init__(self):
        self.docs = []
        self.vocab = {}
        self.idf = None
        self.vecs = None

    def build_index(self, docs):
        self.docs = list(docs)
        tokenized = [tokenize(d) for d in self.docs]
        df = {}
        for toks in tokenized:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.vocab = {t: i for i, t in enumerate(sorted(df.keys()))}
        n = len(self.docs)
        self.idf = np.zeros(len(self.vocab))
        for t, i in self.vocab.items():
            self.idf[i] = math.log((1 + n) / (1 + df[t])) + 1.0
        self.vecs = np.zeros((n, len(self.vocab)))
        for di, toks in enumerate(tokenized):
            tf = {}
            for t in toks:
                if t in self.vocab:
                    tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                self.vecs[di, self.vocab[t]] = (1 + math.log(c)) * self.idf[self.vocab[t]]
        norms = np.linalg.norm(self.vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vecs /= norms

    def search(self, query, top_k=3):
        toks = tokenize(query)
        q = np.zeros(len(self.vocab))
        tf = {}
        for t in toks:
            if t in self.vocab:
                tf[t] = tf.get(t, 0) + 1
        for t, c in tf.items():
            q[self.vocab[t]] = (1 + math.log(c)) * self.idf[self.vocab[t]]
        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm
        scores = self.vecs @ q
        order = np.argsort(-scores)
        return [(self.docs[i], float(scores[i])) for i in order if scores[i] > 0][:top_k]


class BM25Retriever:
    """优化检索器：BM25 评分。"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_tokens = []
        self.df = {}
        self.avg_len = 0
        self.n = 0

    def build_index(self, docs):
        self.docs = list(docs)
        self.doc_tokens = [tokenize(d) for d in self.docs]
        self.n = len(self.docs)
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.avg_len = sum(len(t) for t in self.doc_tokens) / max(1, self.n)

    def _idf(self, term):
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query, top_k=3):
        qt = tokenize(query)
        scores = []
        for i, toks in enumerate(self.doc_tokens):
            dl = len(toks)
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for term in set(qt):
                if term not in self.df:
                    continue
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf(term)
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avg_len))
                score += idf * f * (self.k1 + 1) / denom
            scores.append((i, score))
        scores.sort(key=lambda x: -x[1])
        return [(self.docs[i], float(s)) for i, s in scores if s > 0][:top_k]
