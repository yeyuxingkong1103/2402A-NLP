# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
检索模块：BM25 检索器（表格文本与正文文本统一检索）。
"""
import math

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


class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_tokens = []
        self.df = {}
        self.n = 0
        self.avg_len = 0

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
        scored = []
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
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avg_len)))
            scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        return [(self.docs[i], float(s)) for i, s in scored if s > 0][:top_k]
