# -*- coding: utf-8 -*-
"""重排模块（优化版）
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

对应优化方案中的【优化点 3-3：重排优化】。基线仅用 RRF 融合排名，未利用
查询与片段的“内容级”匹配信号。本模块在 RRF 召回候选之上，做加权线性重排：

    score = w1·向量语义相似度 + w2·BM25 词法分 + w3·查询关键词覆盖率
            + w4·数值/年份匹配度 + w5·实体匹配度

其中：
- 关键词覆盖率：查询实词在片段中的出现比例（解决“答非所问”的语义漂移）；
- 数值匹配度：比例 / 金额 / 数量型问题，优先含数字、单位、年份的片段
  （金融问答的关键数据多藏于含数值的句子或表格行）；
- 实体匹配度：公司名、标准名、工程名等专有名词的命中情况；
- 表格块加成：金额 / 比例型问题对表格块给予轻微加成。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from src import config
from src.query_understanding import QueryAnalysis

_NUM_RE = re.compile(r"\d[\d,，.]*\s*(?:万元|亿元|元|%|％|股|人|项|家|个)?")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}\s*年")


@dataclass
class ScoredDoc:
    """重排后的候选。"""

    doc: Document
    score: float
    detail: dict


def _norm_scores(values: List[float]) -> List[float]:
    """把一组分数做 min-max 归一化到 [0,1]。"""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _keyword_coverage(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    t = text.replace(" ", "")
    hit = sum(1 for k in keywords if k.replace(" ", "") in t)
    return hit / len(keywords)


def _numeric_match(text: str, atype: str, years: List[str]) -> float:
    """数值型问题的数值线索匹配度。"""
    score = 0.0
    if atype in ("ratio", "amount", "number"):
        nums = _NUM_RE.findall(text)
        if nums:
            score += 0.6
        if atype == "ratio" and re.search(r"\d+(?:\.\d+)?\s*[%％]", text):
            score += 0.4
        elif atype == "amount" and re.search(r"\d[\d,，.]*\s*(?:万元|亿元|元)", text):
            score += 0.4
    if years:
        y_hit = sum(1 for y in years if y.replace(" ", "") in text.replace(" ", ""))
        score += 0.2 * (y_hit / len(years))
    return min(score, 1.0)


def _entity_match(text: str, entities: List[str]) -> float:
    if not entities:
        return 0.0
    t = text.replace(" ", "")
    hit = sum(1 for e in entities if e.replace(" ", "") in t)
    return hit / len(entities)


class Reranker:
    """加权线性重排器。"""

    def __init__(
        self,
        w_vector: float = config.W_VECTOR,
        w_bm25: float = config.W_BM25,
        w_keyword: float = config.W_KEYWORD,
        w_numeric: float = config.W_NUMERIC,
        w_entity: float = config.W_ENTITY,
    ):
        self.w_vector = w_vector
        self.w_bm25 = w_bm25
        self.w_keyword = w_keyword
        self.w_numeric = w_numeric
        self.w_entity = w_entity

    def rerank(self, analysis: QueryAnalysis,
               candidates: List[tuple[Document, float, float]],
               top_n: int = config.RERANK_TOP_N) -> List[ScoredDoc]:
        """对候选重排。

        Args:
            analysis: Query 理解结果。
            candidates: [(文档, 向量相似度, BM25 原始分), ...]
            top_n: 保留数量。

        Returns:
            重排后的 ScoredDoc 列表。
        """
        if not candidates:
            return []

        vec_norm = _norm_scores([c[1] for c in candidates])
        bm_norm = _norm_scores([c[2] for c in candidates])

        scored: List[ScoredDoc] = []
        for i, (doc, v, b) in enumerate(candidates):
            text = doc.page_content
            kw = _keyword_coverage(text, analysis.keywords)
            num = _numeric_match(text, analysis.answer_type, analysis.years)
            ent = _entity_match(text, analysis.entities)

            s = (self.w_vector * vec_norm[i] + self.w_bm25 * bm_norm[i]
                 + self.w_keyword * kw + self.w_numeric * num
                 + self.w_entity * ent)
            if doc.metadata.get("ctype") == "table" and analysis.answer_type in ("amount", "ratio", "number"):
                s += 0.03
            scored.append(ScoredDoc(doc, s, {
                "vector": round(vec_norm[i], 4), "bm25": round(bm_norm[i], 4),
                "keyword": round(kw, 4), "numeric": round(num, 4),
                "entity": round(ent, 4),
            }))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]