# -*- coding: utf-8 -*-
"""
重排模块（3 种重排算法）
工单编号：人工智能NLP-RAG-混合检索任务

工单要求「提供至少 3 种重排算法」，本模块实现：
  1. LLMReranker         —— 基于 LLM 的列表级重排（相关性打分 + 理由）
  2. TFIDFReranker       —— 基于 TF-IDF / BM25 词权重的轻量重排
  3. AdaptiveReranker    —— 基于用户反馈的自适应重排（在线学习）

三者可串联（级联重排）：先用便宜的 TF-IDF 粗排，再用 LLM 精排 Top-N，
从而把「重排带来的 3 秒响应时间约束」和「精度」同时满足。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol

import numpy as np

from . import config, llm
from .bm25 import tokenize

FEEDBACK_FILE = config.INDEX_DIR / "rerank_feedback.json"


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, docs: list[dict], top_k: int) -> list[dict]: ...


# ---------------------------------------------------------------------------
# 1. 基于 LLM 的重排器
# ---------------------------------------------------------------------------
_LLM_RERANK_SYS = """你是一个检索结果重排专家。给定用户问题和若干候选文档片段，
请判断每个片段对回答该问题的价值，输出 0~10 的相关性分数。

评分标准：
10 = 片段直接包含问题所需的全部关键信息（如具体数字、名称、结论）
7-9 = 片段包含问题所需的大部分信息，或需与其他片段合并才能完整回答
4-6 = 片段涉及相关主题，但缺少回答问题所需的关键细节
1-3 = 片段仅主题相关，无法用于回答
0   = 完全不相关

只输出 JSON，格式：{"scores": [{"id": <片段序号>, "score": <分数>, "reason": "<不超过20字的理由>"}]}"""


class LLMReranker:
    """
    LLM 列表级重排：把 query + 候选片段一起喂给模型，让模型直接输出相关度。

    相比交叉编码器（cross-encoder），无需本地部署重排模型；
    缺点是每次调用有网络延迟，因此实践中只对粗排 Top-N 使用。
    """

    name = "llm"

    def __init__(self, model: str | None = None, max_doc_chars: int = 500):
        self.model = model
        self.max_doc_chars = max_doc_chars

    def rerank(self, query: str, docs: list[dict], top_k: int = config.TOP_K_RERANK,
               with_reason: bool = False) -> list[dict]:
        if not docs:
            return []

        listing = "\n\n".join(
            f"[{i}] {d.get('text', '')[:self.max_doc_chars]}"
            for i, d in enumerate(docs)
        )
        user = f"用户问题：{query}\n\n候选片段：\n{listing}\n\n请为每个片段打分。"

        try:
            resp = llm.chat_json(
                [{"role": "system", "content": _LLM_RERANK_SYS},
                 {"role": "user", "content": user}],
                model=self.model, temperature=0.0, max_tokens=1500, tag="rerank",
            )
            scores = {int(s["id"]): float(s["score"])
                      for s in resp.get("scores", []) if "id" in s}
            reasons = {int(s["id"]): s.get("reason", "")
                       for s in resp.get("scores", []) if "id" in s}
        except Exception as e:                      # 网络异常时不影响主流程
            print(f"  [warn] LLM 重排失败，回退原顺序：{e}")
            scores, reasons = {}, {}

        out = []
        for i, d in enumerate(docs):
            nd = dict(d)
            # 归一化：LLM 分数 0-10 映射到 0-1，与原检索分数做加权
            s = scores.get(i, 5.0) / 10.0
            base = float(d.get("score", 0.0))
            nd["rerank_score"] = s
            nd["final_score"] = 0.8 * s + 0.2 * base
            if with_reason:
                nd["rerank_reason"] = reasons.get(i, "")
            nd["reranker"] = self.name
            out.append(nd)

        out.sort(key=lambda x: -x["final_score"])
        return out[:top_k]


# ---------------------------------------------------------------------------
# 2. 基于 TF-IDF 的重排器
# ---------------------------------------------------------------------------
class TFIDFReranker:
    """
    TF-IDF 重排：零成本、无网络依赖，作为级联重排的第一级。

    在 BM25 基础上做了两点针对性改进：
      1. 数字加权   —— 招股书问题大量涉及金额/比例/年份，数字命中应显著加分
      2. 实体加权   —— 问题中的公司名、专有名词命中加权
    """

    name = "tfidf"

    def __init__(self, number_boost: float = 2.0, entity_boost: float = 1.8):
        self.number_boost = number_boost
        self.entity_boost = entity_boost
        self._df: Counter[str] = Counter()
        self._n_docs = 1

    def fit(self, docs: list[dict]) -> "TFIDFReranker":
        """在全量语料上统计文档频率，用于计算 IDF。"""
        self._df = Counter()
        for d in docs:
            self._df.update(set(tokenize(d.get("text", ""))))
        self._n_docs = max(len(docs), 1)
        return self

    def _idf(self, term: str) -> float:
        return math.log(1 + self._n_docs / (1 + self._df.get(term, 0)))

    def rerank(self, query: str, docs: list[dict],
               top_k: int = config.TOP_K_RERANK) -> list[dict]:
        if not docs:
            return []

        q_tokens = tokenize(query, for_query=True)
        q_numbers = set(re.findall(r"\d[\d,.]*%?", query))
        q_entities = set(re.findall(r"[一-鿿]{2,}(?:公司|股份|集团|银行)", query))
        qt = Counter(q_tokens)

        out = []
        for d in docs:
            text = d.get("text", "")
            d_tokens = Counter(tokenize(text))
            d_len = max(sum(d_tokens.values()), 1)

            # 基础 TF-IDF 余弦式打分
            score = 0.0
            for t, q_tf in qt.items():
                if t in d_tokens:
                    score += (q_tf / len(qt)) * (d_tokens[t] / d_len) * self._idf(t) ** 2
            score = math.sqrt(score) if score > 0 else 0.0

            # 数字命中加权（招股书问答的关键技巧）
            if q_numbers:
                hit = sum(1 for n in q_numbers if n in text)
                score *= 1 + (self.number_boost - 1) * hit / len(q_numbers)

            # 实体命中加权
            if q_entities:
                hit = sum(1 for e in q_entities if e in text)
                score *= 1 + (self.entity_boost - 1) * hit / len(q_entities)

            nd = dict(d)
            nd["rerank_score"] = score
            nd["final_score"] = score
            nd["reranker"] = self.name
            out.append(nd)

        out.sort(key=lambda x: -x["final_score"])
        return out[:top_k]


# ---------------------------------------------------------------------------
# 3. 基于用户反馈的自适应重排器
# ---------------------------------------------------------------------------
class AdaptiveReranker:
    """
    自适应重排：把用户对答案的「点赞/点踩」沉淀成特征权重，
    下次遇到相似问题时自动调整排序，实现越用越准。

    在线学习策略（简单但有效）：
      · 维护 term -> 反馈分（命中该词且被点赞则 +，点踩则 -）
      · 维护 doc 类型先验（表格/图像类片段在数字类问题上的权重）
      · 用 sigmoid 把反馈分压缩到有界区间，避免被少量极端反馈带偏
    """

    name = "adaptive"

    def __init__(self):
        self.term_feedback: dict[str, float] = defaultdict(float)
        self.type_feedback: dict[str, float] = defaultdict(float)
        self.n_up, self.n_down = 0, 0
        self._load()

    # -- 反馈写入 -----------------------------------------------------------
    def record(self, query: str, doc: dict, helpful: bool) -> None:
        """记录一次用户反馈。helpful=True 表示该片段对回答有帮助。"""
        delta = 1.0 if helpful else -1.0
        if helpful:
            self.n_up += 1
        else:
            self.n_down += 1

        for t in set(tokenize(query)) | set(tokenize(doc.get("text", ""))):
            self.term_feedback[t] += delta
        self.type_feedback[doc.get("type", "text")] += delta
        self._save()

    # -- 重排 ---------------------------------------------------------------
    def rerank(self, query: str, docs: list[dict],
               top_k: int = config.TOP_K_RERANK) -> list[dict]:
        if not docs:
            return []

        q_tokens = set(tokenize(query, for_query=True))
        out = []
        for d in docs:
            prior = 0.0
            for t in q_tokens:
                prior += self.term_feedback.get(t, 0.0)
            prior += self.type_feedback.get(d.get("type", "text"), 0.0)

            # sigmoid 压缩，避免反馈量级主导排序
            adapted = 1.0 / (1.0 + math.exp(-prior / 5.0))
            base = float(d.get("score", 0.0))
            base_norm = base if 0 <= base <= 1 else 1.0 / (1.0 + math.exp(-base))

            nd = dict(d)
            nd["rerank_score"] = adapted
            nd["final_score"] = 0.7 * base_norm + 0.3 * adapted
            nd["reranker"] = self.name
            out.append(nd)

        out.sort(key=lambda x: -x["final_score"])
        return out[:top_k]

    # -- 持久化 -------------------------------------------------------------
    def _save(self) -> None:
        FEEDBACK_FILE.write_text(json.dumps({
            "term_feedback": dict(self.term_feedback),
            "type_feedback": dict(self.type_feedback),
            "n_up": self.n_up, "n_down": self.n_down,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    def _load(self) -> None:
        if not FEEDBACK_FILE.exists():
            return
        try:
            o = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
            self.term_feedback = defaultdict(float, o.get("term_feedback", {}))
            self.type_feedback = defaultdict(float, o.get("type_feedback", {}))
            self.n_up, self.n_down = o.get("n_up", 0), o.get("n_down", 0)
        except Exception:
            pass

    def stats(self) -> dict:
        return {"点赞": self.n_up, "点踩": self.n_down,
                "已学习词条": len(self.term_feedback)}


# ---------------------------------------------------------------------------
# 级联重排
# ---------------------------------------------------------------------------
class CascadeReranker:
    """
    级联重排：TF-IDF 粗排 Top-M → LLM 精排 Top-K。
    兼顾精度与 3 秒响应时间约束（LLM 只处理少量候选）。
    """

    name = "cascade"

    def __init__(self, coarse_k: int = 10):
        self.tfidf = TFIDFReranker()
        self.llm = LLMReranker()
        self.coarse_k = coarse_k

    def fit(self, docs: list[dict]) -> "CascadeReranker":
        self.tfidf.fit(docs)
        return self

    def rerank(self, query: str, docs: list[dict],
               top_k: int = config.TOP_K_RERANK) -> list[dict]:
        coarse = self.tfidf.rerank(query, docs, top_k=min(self.coarse_k, len(docs)))
        fine = self.llm.rerank(query, coarse, top_k=top_k, with_reason=True)
        for d in fine:
            d["reranker"] = self.name
        return fine


RERANKERS = {
    "none": None,
    "llm": LLMReranker,
    "tfidf": TFIDFReranker,
    "adaptive": AdaptiveReranker,
    "cascade": CascadeReranker,
}


def get_reranker(name: str) -> Reranker | None:
    if name == "none":
        return None
    if name not in RERANKERS:
        raise ValueError(f"未知重排器：{name}，可选 {list(RERANKERS)}")
    return RERANKERS[name]()
