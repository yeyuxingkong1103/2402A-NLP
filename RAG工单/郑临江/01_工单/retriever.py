# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
向量检索模块：使用 TF-IDF 将文档块与问题向量化，通过余弦相似度召回最相关块。
（轻量实现，无需下载嵌入模型，离线可运行；如需更强语义可替换为 sentence-transformers。）
"""
import math
import numpy as np

try:
    import jieba
except ImportError:
    jieba = None


def tokenize(text: str):
    """中文分词；jieba 不可用时退化为字符二元组。"""
    text = text.lower()
    if jieba is not None:
        toks = [t.strip() for t in jieba.cut(text) if t.strip()]
    else:
        # 退化为单字 + 二元组，保证中文检索可用
        chars = [c for c in text if not c.isspace()]
        toks = chars + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return toks


class TfidfRetriever:
    """基于 TF-IDF 与余弦相似度的轻量向量检索器。"""

    def __init__(self):
        self.docs = []
        self.vocab = {}
        self.idf = None
        self.doc_vectors = None

    def build_index(self, docs):
        """输入文档块列表，构建词表与文档向量。"""
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

        self.doc_vectors = np.zeros((n, len(self.vocab)))
        for di, toks in enumerate(tokenized):
            tf = {}
            for t in toks:
                if t in self.vocab:
                    tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                self.doc_vectors[di, self.vocab[t]] = (1 + math.log(c)) * self.idf[self.vocab[t]]
        norms = np.linalg.norm(self.doc_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_vectors /= norms

    def _query_vector(self, query):
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
        return q

    def search(self, query, top_k=3):
        """返回 [(文本块, 相似度得分)]，按得分降序。"""
        if self.doc_vectors is None or len(self.docs) == 0:
            return []
        q = self._query_vector(query)
        scores = self.doc_vectors @ q
        order = np.argsort(-scores)
        results = []
        for i in order:
            if scores[i] <= 0:
                continue
            results.append((self.docs[i], float(scores[i])))
            if len(results) >= top_k:
                break
        return results
