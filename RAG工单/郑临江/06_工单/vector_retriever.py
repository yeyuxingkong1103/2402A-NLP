# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
向量检索模块：召回 + 重排。
  召回：向量余弦相似度召回 Top-K；
  重排：提供 3 种重排算法——
    1. TfidfReranker           基于 TF-IDF 相似度重排
    2. LLMReranker            基于 LLM 的相关性打分重排（离线降级为规则）
    3. AdaptiveFeedbackReranker 基于用户反馈的自适应重排
"""
import math
import numpy as np
from embedder import tokenize, TfidfEmbedder


class VectorRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.docs = []
        self.doc_vectors = None

    def build(self, docs):
        self.docs = list(docs)
        self.doc_vectors = self.embedder.encode(docs)

    def recall(self, query, top_k=10):
        q = self.embedder.encode([query])[0]
        scores = self.doc_vectors @ q
        order = np.argsort(-scores)
        return [(self.docs[i], float(scores[i])) for i in order if scores[i] > 0][:top_k]


class TfidfReranker:
    """重排算法1：TF-IDF 相似度重排。"""

    def __init__(self, docs):
        self.tfidf = TfidfEmbedder()
        self.tfidf.fit(docs)
        self.vecs = self.tfidf.encode(docs)

    def rerank(self, query, candidates, top_k=3):
        q = self.tfidf.encode([query])[0]
        scored = []
        for doc in candidates:
            # 在已建索引中找该文档的向量；简化：直接现编码
            v = self.tfidf.encode([doc])[0]
            scored.append((doc, float(v @ q)))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


class LLMReranker:
    """
    重排算法2：基于 LLM 的重排器。
    配置 API 时用 LLM 打分；离线降级为关键词命中打分。
    """

    def __init__(self, llm=None):
        self.llm = llm

    def rerank(self, query, candidates, top_k=3):
        if self.llm and self.llm.api_key:
            return self._llm_rerank(query, candidates, top_k)
        return self._keyword_rerank(query, candidates, top_k)

    def _keyword_rerank(self, query, candidates, top_k):
        qt = set(tokenize(query))
        scored = []
        for doc in candidates:
            dt = set(tokenize(doc))
            overlap = len(qt & dt) / max(1, len(qt))
            scored.append((doc, overlap))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def _llm_rerank(self, query, candidates, top_k):
        prompt = "请对以下候选段落与问题的相关性从0到1打分（只输出数字，用换行分隔）：\n"
        prompt += f"问题：{query}\n"
        for i, c in enumerate(candidates):
            prompt += f"[{i}] {c[:80]}\n"
        out = self.llm.generate(prompt)
        nums = [float(x) for x in out.replace("\n", " ").split() if x.replace(".", "").isdigit()]
        scored = list(zip(candidates, nums[:len(candidates)]))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


class AdaptiveFeedbackReranker:
    """
    重排算法3：基于用户反馈的自适应重排。
    根据历史“点击/采纳”反馈，对相关文档加分。
    """

    def __init__(self):
        self.feedback = {}  # doc文本哈希 -> 累计反馈分

    def record_feedback(self, doc, positive=True):
        key = hash(doc)
        self.feedback[key] = self.feedback.get(key, 0) + (1.0 if positive else -1.0)

    def rerank(self, query, candidates, top_k=3):
        qt = set(tokenize(query))
        scored = []
        for doc in candidates:
            base = len(qt & set(tokenize(doc))) / max(1, len(qt))
            fb = self.feedback.get(hash(doc), 0.0)
            scored.append((doc, base + 0.1 * fb))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
