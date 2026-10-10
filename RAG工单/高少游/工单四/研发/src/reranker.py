# -*- coding: utf-8 -*-
"""重排模块（图像内容解析及检索优化版）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

在 02 工单「加权线性重排」基础上，新增【表格感知】信号（本工单核心）：

    score = w1·向量语义相似度 + w2·BM25 词法分 + w3·查询关键词覆盖率
            + w4·数值/年份匹配度 + w5·实体匹配度 + w6·表格块加成

其中「表格块加成」：
- 当 Query 被判定为表格类问题（answer_type == "table"）时，对 `table` /
  `table_kv` 块给予更高加成，因为这类问题的答案几乎必然落在表格中；
- 键值对块（table_kv）相比 Markdown 块更贴近“字段—取值”的问法，加成略高；
- 非表格类问题仅对表格块给予微弱加成，避免干扰正文型问题的排序。
"""
from __future__ import annotations

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
    if atype in ("ratio", "amount", "number", "table"):
        if _NUM_RE.findall(text):
            score += 0.6
        if atype == "ratio" and re.search(r"\d+(?:\.\d+)?\s*[%％]", text):
            score += 0.4
        elif atype in ("amount", "table") and re.search(r"\d[\d,，.]*\s*(?:万元|亿元|元)", text):
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


def _table_bonus(ctype: str, atype: str) -> float:
    """表格块加成（本工单新增）。

    Returns:
        0~1 的加成系数：表格类问题对 table/table_kv 块强加成；其他问题弱加成。
    """
    is_table_q = atype == "table"
    if ctype == "table_kv":
        return 1.0 if is_table_q else 0.3
    if ctype == "table":
        return 0.85 if is_table_q else 0.25
    return 0.0


def _figure_bonus(ctype: str, atype: str) -> float:
    """图形块加成（本工单新增）。

    图形语义块（ctype="figure"）承载组织结构图 / 统计图的结构化语义。当问题被判为
    图形类（answer_type == "figure"）时给予强加成，确保「组织结构层级」「图内涨跌幅」
    类问题的答案块能排到前列；其他问题给予弱加成，避免干扰正文型问题。
    """
    if ctype != "figure":
        return 0.0
    return 1.0 if atype == "figure" else 0.2


class Reranker:
    """加权线性重排器（含表格感知加成）。"""

    def __init__(
        self,
        w_vector: float = config.W_VECTOR,
        w_bm25: float = config.W_BM25,
        w_keyword: float = config.W_KEYWORD,
        w_numeric: float = config.W_NUMERIC,
        w_entity: float = config.W_ENTITY,
        w_table: float = config.W_TABLE,
        w_figure: float = config.W_FIGURE,
        w_clip: float = config.W_CLIP,
        w_doc: float = config.W_DOC,
    ):
        self.w_vector = w_vector
        self.w_bm25 = w_bm25
        self.w_keyword = w_keyword
        self.w_numeric = w_numeric
        self.w_entity = w_entity
        self.w_table = w_table
        self.w_figure = w_figure
        self.w_clip = w_clip
        self.w_doc = w_doc

    def rerank(self, analysis: QueryAnalysis,
               candidates: List[tuple[Document, float, float]],
               top_n: int = config.RERANK_TOP_N,
               clip_map: dict | None = None) -> List[ScoredDoc]:
        """对候选重排。

        Args:
            analysis: Query 理解结果。
            candidates: [(文档, 向量相似度, BM25 原始分), ...]
            top_n: 保留数量。
            clip_map: 可选的 {块内容: CLIP 跨模态相似度}（「以文搜图」命中的图形块）。

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
            ctype = doc.metadata.get("ctype", "text")
            kw = _keyword_coverage(text, analysis.keywords)
            num = _numeric_match(text, analysis.answer_type, analysis.years)
            ent = _entity_match(text, analysis.entities)
            tbl = _table_bonus(ctype, analysis.answer_type)
            fig = _figure_bonus(ctype, analysis.answer_type)
            clip = float((clip_map or {}).get(text, 0.0))
            # 文档级路由加成：块来源与问题所指公司一致时加分（多文档消歧）
            doc_ok = 1.0 if (analysis.doc_hint
                             and doc.metadata.get("source") == analysis.doc_hint) else 0.0

            s = (self.w_vector * vec_norm[i] + self.w_bm25 * bm_norm[i]
                 + self.w_keyword * kw + self.w_numeric * num
                 + self.w_entity * ent + self.w_table * tbl + self.w_figure * fig
                 + self.w_clip * clip + self.w_doc * doc_ok)
            scored.append(ScoredDoc(doc, s, {
                "vector": round(vec_norm[i], 4), "bm25": round(bm_norm[i], 4),
                "keyword": round(kw, 4), "numeric": round(num, 4),
                "entity": round(ent, 4), "table": round(tbl, 4),
                "figure": round(fig, 4), "clip": round(clip, 4),
                "doc": round(doc_ok, 4),
            }))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]