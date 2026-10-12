# -*- coding: utf-8 -*-
"""
rerankers.py — 三种重排算法
工单编号：人工智能NLP-RAG-混合检索任务

对召回结果做二次排序，提供至少 3 种重排算法：
  1. 基于 LLM 的重排器（LLMReranker）
  2. 基于 TF-IDF 的重排器（TfidfReranker）
  3. 基于用户反馈的自适应重排器（AdaptiveReranker）
"""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

from rag_core import tokenize, BaseLLM


def _score_via_llm(llm: BaseLLM, query: str, text: str) -> float:
    prompt = (
        "请判断以下文本片段与问题的相关性，仅输出 0 到 1 之间的一位小数。\n"
        f"问题：{query}\n文本片段：{text[:400]}\n相关性分数："
    )
    try:
        m = re.search(r"(\d+\.?\d*)", llm.generate(prompt))
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


class LLMReranker:
    """基于 LLM 的重排器：模型对每个候选片段打分。"""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for h in hits:
            h["score"] = _score_via_llm(self.llm, query, h["text"])
        return sorted(hits, key=lambda x: -x["score"])


class TfidfReranker:
    """基于 TF-IDF 的重排器：词面重合度重排。"""

    def rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        q_tokens = set(tokenize(query))
        for h in hits:
            s_tokens = set(tokenize(h["text"]))
            if not s_tokens:
                continue
            h["score"] = len(q_tokens & s_tokens) / max(len(q_tokens), 1)
        return sorted(hits, key=lambda x: -x["score"])


class AdaptiveReranker:
    """基于用户反馈的自适应重排器。

    根据历史点击/正反馈统计，对「常被用户认可」的文档片段加权，
    实现个性化、自适应的重排。反馈通过 `record_feedback(index, label)` 累积。
    """

    def __init__(self):
        self.feedback: Dict[int, float] = {}  # index -> 累计加权

    def record_feedback(self, index: int, label: int):
        """label: 1 表示有用，-1 表示无用。"""
        self.feedback[index] = self.feedback.get(index, 0.0) + label

    def rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        q_tokens = set(tokenize(query))
        for h in hits:
            base = 0.0
            s_tokens = set(tokenize(h["text"]))
            if s_tokens:
                base = len(q_tokens & s_tokens) / max(len(q_tokens), 1)
            # 叠加历史反馈加权（软性，幅度受限）
            fb = self.feedback.get(h.get("index", -1), 0.0)
            h["score"] = base + 0.1 * fb
        return sorted(hits, key=lambda x: -x["score"])


def get_reranker(name: str, llm: Optional[BaseLLM] = None):
    """工厂：按名称返回重排器。"""
    name = (name or "").lower()
    if name in ("llm", "大模型"):
        assert llm is not None
        return LLMReranker(llm)
    if name in ("adaptive", "反馈", "自适应"):
        return AdaptiveReranker()
    return TfidfReranker()  # 默认 tf-idf
