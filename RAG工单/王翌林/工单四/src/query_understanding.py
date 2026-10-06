# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-基于PDF文档的问答系统
src/query_understanding.py — 查询理解
"""
import re
from typing import Dict, List, Optional
import jieba

_STOPWORDS = set("的 了 是 在 和 与 或 也 都 就 又 及 等 这 那 这个 那个 什么 怎么 为什么 如何 哪 请问 报告 报告期 期间 年度 年 月 日 季度 公司 股份 有限 招股 说明书 发行 上市 股票".split())

_INTENT_PATTERNS = [
    ("summary", [r"摘要", r"概述", r"介绍.*公司", r"主要业务"]),
    ("comparison", [r"比较", r"对比", r"区别", r"差异"]),
    ("statistics", [r"多少", r"多大", r"比例", r"占比", r"金额", r"数据", r"统计", r"百分比"]),
    ("definition", [r"什么是", r"什么意思", r"定义"]),
    ("risk", [r"风险", r"不确定性"]),
    ("finance", [r"营收", r"收入", r"利润", r"净利润", r"资产", r"负债", r"现金流"]),
    ("military", [r"军", r"军方", r"军用", r"国防", r"装备"]),
]

def detect_intent(query: str) -> List[str]:
    intents = []
    for name, patterns in _INTENT_PATTERNS:
        for p in patterns:
            if re.search(p, query): intents.append(name); break
    return intents if intents else ["qa"]

def extract_keywords(query: str, top_k: int = 10) -> List[str]:
    tokens = list(jieba.cut(query))
    filtered = []
    for t in tokens:
        t = t.strip()
        if not t or t in _STOPWORDS or len(t) == 1: continue
        if re.match(r"^[\d,.]+$", t): filtered.append(t); continue
        if t not in filtered: filtered.append(t)
    return filtered[:top_k]

def rewrite_query(query: str, context: Optional[str] = None) -> str:
    q = query.strip()
    if context:
        q = q.replace("该公司", "武汉兴图新科电子股份有限公司").replace("其", "武汉兴图新科电子股份有限公司")
    return q

def decompose_query(query: str) -> List[str]:
    parts = re.split(r"[；;，,]|(?:同时|并且|另外|以及)", query)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) >= 3]
    return parts if len(parts) > 1 else [query]

def analyze(query: str, context: Optional[str] = None) -> Dict:
    return {"original_query": query, "rewritten_query": rewrite_query(query, context),
            "intents": detect_intent(query), "keywords": extract_keywords(query),
            "sub_queries": decompose_query(query)}
