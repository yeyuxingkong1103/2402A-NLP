# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
重排算法集合（≥3 种，用于向量检索的"召回+重排"）：
  1. LLMReranker       —— 基于 LLM 的相关性重排（调用大模型打分/选序）
  2. TFIDFReranker     —— 基于 TF-IDF 向量余弦相似度的统计重排
  3. FeedbackReranker  —— 基于用户反馈的自适应重排（👍/👎 累积得分，在线调整排序）
三种重排器统一接口：rerank(query, candidates, top_k) -> [(chunk, score), ...]
"""
import os
import re
import json
import math
from collections import defaultdict

from config import TOP_K

try:
    from reranker import LLMReranker
except Exception:                                     # 保证 LLM 不可用时仍可导入
    LLMReranker = None

FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "feedback.json")


# ============ 1. TF-IDF 重排器 ============
class TFIDFReranker:
    """对候选集重算 TF-IDF 余弦相似度并重排（统计型重排，无需大模型）"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    @staticmethod
    def _tok(text):
        text = text.lower()
        toks = re.findall(r"[a-z0-9]+", text)
        for seg in re.findall(r"[一-鿿]+", text):
            toks += [seg[i:i + 2] for i in range(len(seg) - 1)] or [seg]
        return toks

    def rerank(self, query, candidates, top_k=TOP_K):
        docs = [self._tok(c["text"] if isinstance(c, dict) else c) for c, _ in candidates]
        q = self._tok(query)
        N = max(len(docs), 1)
        df = defaultdict(int)
        for d in docs:
            for t in set(d):
                df[t] += 1
        avgdl = sum(len(d) for d in docs) / N or 1.0

        def idf(t):
            return math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))

        def vec_score(d, qset):
            dl = len(d) or 1
            num = 0.0
            for t in qset:
                tf = d.count(t)
                if tf == 0:
                    continue
                num += idf(t) * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / avgdl))
            return num / (dl ** 0.5)

        qset = set(q)
        scored = [(c, s, vec_score(d, qset)) for (c, s), d in zip(candidates, docs)]
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(c, s) for c, s, _ in scored[:top_k]]


# ============ 2. 基于用户反馈的自适应重排器 ============
class FeedbackReranker:
    """
    自适应重排：记录用户对检索结果的 👍/👎 反馈（按块指纹聚合），
    排序时将反馈得分作为加权项叠加到原检索分数上；
    反馈越多，权重自适应增大（上限 alpha_max），实现"越用越准"。
    """

    def __init__(self, base_reranker=None, alpha=0.3, alpha_max=0.8, feedback_file=FEEDBACK_FILE):
        self.base = base_reranker
        self.alpha = alpha
        self.alpha_max = alpha_max
        self.file = feedback_file
        self.feedback = self._load()

    # ---- 反馈持久化 ----
    def _load(self):
        if os.path.exists(self.file):
            try:
                with open(self.file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(self.feedback, f, ensure_ascii=False, indent=2)

    @staticmethod
    def key_of(chunk):
        """块指纹：文档+页码+文本前64字，保证跨会话稳定"""
        if isinstance(chunk, str):
            return chunk[:64]
        return f"{chunk.get('doc','')}|{chunk.get('page','')}|{chunk.get('text','')[:64]}"

    def record(self, chunk, useful=True):
        """记录一次用户反馈，useful=True 为 👍，False 为 👎"""
        k = self.key_of(chunk)
        rec = self.feedback.setdefault(k, {"up": 0, "down": 0})
        rec["up" if useful else "down"] += 1
        self._save()

    def stats(self):
        up = sum(v["up"] for v in self.feedback.values())
        down = sum(v["down"] for v in self.feedback.values())
        return {"feedback_chunks": len(self.feedback), "up": up, "down": down}

    # ---- 自适应权重：反馈总量越大，alpha 越大 ----
    def _alpha_now(self):
        n = sum(v["up"] + v["down"] for v in self.feedback.values())
        return min(self.alpha_max, self.alpha + 0.02 * n)

    def _boost(self, chunk):
        rec = self.feedback.get(self.key_of(chunk))
        if not rec:
            return 0.0
        u, d = rec["up"], rec["down"]
        return (u - d) / (u + d + 1.0)      # 归一化到 (-1, 1)

    def rerank(self, query, candidates, top_k=TOP_K):
        # 先做基础重排（若配置了 LLM/TF-IDF 则串联）
        base = self.base.rerank(query, candidates, top_k=len(candidates)) if self.base else candidates
        vals = [s for _, s in base] or [0.0]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        alpha = self._alpha_now()
        out = [(c, (s - lo) / span + alpha * self._boost(c)) for c, s in base]
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_k]


# ============ 3. 统一工厂 ============
RERANKER_NAMES = ["llm", "tfidf", "feedback", "none"]


def build_reranker(name, feedback_alpha=0.3):
    """按名称构建重排器；feedback 重排器默认串联 TF-IDF 作为基础重排"""
    name = (name or "none").lower()
    if name == "llm" and LLMReranker is not None:
        return LLMReranker()
    if name == "tfidf":
        return TFIDFReranker()
    if name == "feedback":
        return FeedbackReranker(base_reranker=TFIDFReranker(), alpha=feedback_alpha)
    return None
