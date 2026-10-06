# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-混合检索任务
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务
# 模块：rerank —— 重排器（至少 3 种，可插拔）
# 说明：为工单 06《混合检索任务》提供「重排」能力。三种实现：
#   1) LLMReranker        —— 基于 LLM 的重排器：让 qwen2:7b 给候选片段打 0-10 相关性分
#   2) TFIDFReranker      —— 基于 TF-IDF 的重排器：候选集内 tf-idf 余弦相似度
#   3) FeedbackReranker   —— 基于用户反馈的自适应重排器：按历史 👍/👎 自适应加权
#   另含 NoReranker（不重排）。统一接口 rerank(query, hits, top_k) -> hits（改写 score_rerank 字段）

import math
import os
import json
import re
from collections import Counter, defaultdict

from bm25 import tokenize
import llm


class BaseReranker:
    name = "none"

    def rerank(self, query, hits, top_k=None):
        return hits[:top_k] if top_k else hits

    # 统一出口：把重排分写回 hit["rerank_score"]，按分排序
    @staticmethod
    def _finalize(hits, scores, top_k):
        out = []
        for h, s in zip(hits, scores):
            h = dict(h)
            h["rerank_score"] = round(float(s), 4)
            out.append(h)
        out.sort(key=lambda x: -x["rerank_score"])
        return out[:top_k] if top_k else out


class NoReranker(BaseReranker):
    name = "none"


class TFIDFReranker(BaseReranker):
    """候选集内 TF-IDF 余弦重排（不依赖外部模型，纯统计）。
    alpha=1.0 时纯 TF-IDF 排序；默认 0.5 与原始召回分混合（实测纯 TF-IDF 会丢好结果）。"""

    name = "tfidf"

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def rerank(self, query, hits, top_k=None):
        if not hits:
            return []
        q = Counter(tokenize(query))
        toks = [Counter(tokenize(h.get("text", ""))) for h in hits]
        df = Counter()
        for t in toks:
            for w in t:
                df[w] += 1
        n = len(hits)

        def vec(t):
            v = {}
            for w, f in t.items():
                v[w] = (1 + math.log(f)) * math.log((n + 1) / (df[w] + 1)) + 1e-9
            return v

        def cos(a, b):
            va, vb = vec(a), vec(b)
            dot = sum(va.get(w, 0.0) * vb.get(w, 0.0) for w in vb)
            na = math.sqrt(sum(x * x for x in va.values())) or 1.0
            nb = math.sqrt(sum(x * x for x in vb.values())) or 1.0
            return dot / (na * nb)

        qv = vec(q)
        raw = []
        for h, t in zip(hits, toks):
            tv = vec(t)
            dot = sum(qv.get(w, 0.0) * tv.get(w, 0.0) for w in tv)
            na = math.sqrt(sum(x * x for x in qv.values())) or 1.0
            nb = math.sqrt(sum(x * x for x in tv.values())) or 1.0
            raw.append(dot / (na * nb))
        mx = max(raw) or 1.0
        bases = [float(h.get("rrf", 0.0)) or float(h.get("dense_sim", 0.0))
                 or float(h.get("score", 0.0)) or 0.0 for h in hits]
        bmx = max(bases) or 1.0
        a = self.alpha
        scores = [a * (r / mx) + (1 - a) * (b / bmx) for r, b in zip(raw, bases)]
        return self._finalize(hits, scores, top_k)


class LLMReranker(BaseReranker):
    """基于 LLM 的重排器：逐条让生成模型打 0-10 相关性分（失败自动回退原序）。"""

    name = "llm"
    _SYS = ("你是检索结果重排助手。判断【片段】是否包含回答【问题】所需的信息，"
            "只输出一个 0-10 的整数（0=完全无关，10=直接给出答案），不要解释。")

    def __init__(self, model=None, max_chars=600, batch_note=True):
        self.model = model
        self.max_chars = max_chars

    def _score_one(self, query, text):
        msg = [{"role": "system", "content": self._SYS},
               {"role": "user", "content": "【问题】%s\n【片段】%s" % (query, text[:self.max_chars])}]
        try:
            out = llm.chat(msg, model=self.model, temperature=0.0)
        except Exception:  # noqa: BLE001
            return None
        m = re.search(r"\d+(?:\.\d+)?", out or "")
        return float(m.group()) / 10.0 if m else None

    def rerank(self, query, hits, top_k=None):
        if not hits:
            return []
        scores = []
        for h in hits:
            s = self._score_one(query, h.get("text", ""))
            if s is None:      # 模型不可用 -> 回退到原始分（rrf / dense_sim）
                s = float(h.get("rrf", 0.0)) or float(h.get("dense_sim", 0.0))
            scores.append(s)
        return self._finalize(hits, scores, top_k)


class FeedbackReranker(BaseReranker):
    """基于用户反馈的自适应重排器。

    从 data/feedback.jsonl 读取历史反馈（payload 可带 hits/chunk_ids/page 信息），
    统计「块/页」级的好评(+)与差评(-)，对新候选做自适应加权：score += rate * boost。
    """

    name = "feedback"

    def __init__(self, feedback_path=None, rate=0.15):
        from config import DATA_DIR
        self.path = feedback_path or os.path.join(DATA_DIR, "feedback.jsonl")
        self.rate = rate
        self.chunk_boost = defaultdict(float)
        self.page_boost = defaultdict(float)
        self.reload()

    def reload(self):
        self.chunk_boost, self.page_boost = defaultdict(float), defaultdict(float)
        if not os.path.exists(self.path):
            return
        for line in open(self.path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            vote = rec.get("vote") or rec.get("feedback") or ""
            w = 1.0 if vote in (1, "1", "up", "👍", True) else (-1.0 if vote in (-1, "-1", "down", "👎", False) else 0.0)
            if not w:
                continue
            for cid in (rec.get("chunk_ids") or []):
                self.chunk_boost[cid] += w
            for pg in (rec.get("pages") or []):
                self.page_boost[pg] += w
            for h in (rec.get("hits") or []):
                if isinstance(h, dict):
                    if h.get("id") is not None:
                        self.chunk_boost[h["id"]] += w
                    if h.get("page") is not None:
                        self.page_boost[h["page"]] += w

    def rerank(self, query, hits, top_k=None):
        if not hits:
            return []
        maxb = max([abs(v) for v in list(self.chunk_boost.values()) + list(self.page_boost.values())] or [1.0]) or 1.0
        scores = []
        for h in hits:
            base = (float(h.get("rrf", 0.0)) or float(h.get("dense_sim", 0.0))
                    or float(h.get("score", 0.0)) or 1.0)   # 无基础分时取 1.0，保证反馈可独立排序
            b = self.chunk_boost.get(h.get("id"), 0.0) + self.page_boost.get(h.get("page"), 0.0)
            scores.append(base * (1.0 + self.rate * b / maxb))
        return self._finalize(hits, scores, top_k)


_RERANKERS = {"none": NoReranker, "tfidf": TFIDFReranker, "llm": LLMReranker, "feedback": FeedbackReranker}


def get_reranker(name="none", **kw):
    cls = _RERANKERS.get((name or "none").lower())
    if cls is None:
        raise KeyError("未知重排器：%s（可选 %s）" % (name, list(_RERANKERS)))
    return cls(**kw)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from clean import clean_blocks          # noqa
    from chunk import build_chunks          # noqa
    from parse import load_blocks           # noqa
    ch = build_chunks(clean_blocks(load_blocks()))
    q = "武汉兴图新科电子股份有限公司的注册资本是多少？"
    cand = [c for c in ch if c["page"] in (52, 22, 129, 343, 30)][:6]
    for name in ("tfidf", "llm", "feedback"):
        r = get_reranker(name)
        out = r.rerank(q, cand, top_k=3)
        print("%-9s -> %s" % (name, [(h["page"], h.get("rerank_score")) for h in out]))
