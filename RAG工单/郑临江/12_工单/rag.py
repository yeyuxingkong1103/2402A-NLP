# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-LightRAG优化
传统 RAG 基线：TF-IDF 向量检索 + LLM 生成，用于与 LightRAG 对比。
"""
import math
from collections import Counter

from llm import call_llm

try:
    import jieba
except Exception:
    jieba = None


def tokenize(text):
    text = text.lower()
    if jieba is not None:
        return [t.strip() for t in jieba.cut(text) if t.strip()]
    chars = [c for c in text if not c.isspace()]
    return chars + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


class RAG:
    def __init__(self, docs):
        self.docs = docs
        self._index()

    def _index(self):
        self.df = Counter()
        self.tf = []
        for d in self.docs:
            toks = tokenize(d["text"])
            self.tf.append(Counter(toks))
            for t in set(toks):
                self.df[t] += 1
        n = max(len(self.docs), 1)
        self.idf = {t: math.log((n + 1) / (f + 1)) + 1 for t, f in self.df.items()}

    def _vec(self, toks):
        return {t: self.idf[t] for t in toks if t in self.idf}

    def retrieve(self, query, top_k=5):
        qv = self._vec(tokenize(query))
        scores = []
        for i, tf in enumerate(self.tf):
            s = sum(w * tf.get(t, 0) * self.idf[t] for t, w in qv.items())
            scores.append((s, i))
        scores.sort(reverse=True)
        return [self.docs[i] for _, i in scores[:top_k]]

    def answer(self, query, top_k=5):
        docs = self.retrieve(query, top_k)
        ctx = "\n".join(d["text"] for d in docs)
        ans = call_llm(f"根据以下上下文回答问题：\n{ctx}\n\n问题：{query}")
        return {
            "contexts": [d["text"] for d in docs],
            "answer": ans or "（离线模式）检索到的相关内容见上下文。",
            "docs": docs,
        }
