# -*- coding: utf-8 -*-
"""多信号重排：把混合检索候选按加权线性融合重新排序。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

信号构成：
    向量相似度 / BM25 / 关键词覆盖率 / 数值年份匹配 / 实体匹配 /
    表格块加成 / 图形块加成 / 文档路由加成 / 多轮话题实体加成 /
    【本工单新增】同义词覆盖加成（跨表述命中，如「军用领域」问题命中「国防客户」句）/
    【本工单新增】数值序列信号（「…分别是多少」类问题，受关键词/同义词覆盖门控）
"""
from __future__ import annotations

import re
from typing import List

from src import config
from src.knowledge_base import KnowledgeBase
from src.retriever import Retrieved

_KIND_BONUS = {"table": config.W_TABLE, "figure": config.W_FIGURE}

# 数值序列信号：「分别是多少」类问题，答案常以「（合计）分别为 A、B、C 和 D」给出
_SERIES_RE = re.compile(r"(合计分别为|分别为|依次为|分别是)")
_SERIES_TYPES = {"amount", "ratio", "number", "table"}


def _norm(vals: List[float]) -> List[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def _coverage(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    hit = sum(1 for k in keywords if k and k in text)
    return hit / len(keywords)


def _series_match(text: str, answer_type: str) -> float:
    """数值序列表述命中（针对「分别是多少」类问题）。"""
    if answer_type in _SERIES_TYPES and _SERIES_RE.search(text or ""):
        return 1.0
    return 0.0


def _numeric_match(text: str, numbers: List[str]) -> float:
    if not numbers:
        return 0.0
    hit = sum(1 for n in numbers if n in text)
    return hit / len(numbers)


def rerank(kb: KnowledgeBase, candidates: List[Retrieved], analysis,
           topic_entities: List[str] | None = None,
           top_k: int | None = None) -> List[Retrieved]:
    """对候选做加权线性融合重排。"""
    top_k = top_k or config.TOP_K
    if not candidates:
        return []

    v = _norm([c.vector_score for c in candidates])
    b = _norm([c.bm25_score for c in candidates])
    keywords = list(getattr(analysis, "keywords", []) or [])
    synonyms = list(getattr(analysis, "synonyms", []) or [])
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", getattr(analysis, "core", "") or "")
    entities = list(getattr(analysis, "entities", []) or [])
    doc_hint = getattr(analysis, "doc_hint", "") or ""
    topics = topic_entities or []
    atype = getattr(analysis, "answer_type", "") or ""

    scored: List[Retrieved] = []
    for idx, c in enumerate(candidates):
        chunk = kb.chunks[c.chunk_id]
        text = chunk.text
        # 数值序列信号受关键词/同义词覆盖门控：仅当该片段确实命中查询主题时才加成，
        # 避免「分别为」这类高频表述把无关片段顶上来。
        series_gate = max(_coverage(text, keywords), _coverage(text, synonyms))
        s = (config.W_VECTOR * v[idx]
             + config.W_BM25 * b[idx]
             + config.W_KEYWORD * _coverage(text, keywords)
             + config.W_SYN * _coverage(text, synonyms)
             + config.W_NUMERIC * _numeric_match(text, numbers)
             + config.W_SERIES * _series_match(text, atype) * series_gate
             + config.W_ENTITY * _coverage(text, entities)
             + _KIND_BONUS.get(chunk.kind, 0.0)
             + config.W_DOC * (1.0 if doc_hint and chunk.source == doc_hint else 0.0)
             + config.W_TOPIC * _coverage(text, topics))
        scored.append(Retrieved(chunk_id=c.chunk_id, score=s,
                                vector_score=c.vector_score,
                                bm25_score=c.bm25_score, rrf=c.rrf))
    scored.sort(key=lambda r: -r.score)
    return scored[:top_k]