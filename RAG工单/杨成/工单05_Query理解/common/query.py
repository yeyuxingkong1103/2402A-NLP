"""Query 理解、改写与扩展工具。"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable


@dataclass(frozen=True)
class QueryAnalysis:
    """查询意图与实体分析结果。"""

    intent: str
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    original: str = ""


_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_PERCENT_RE = re.compile(r"(?:\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*百分之)")
_STOCK_RE = re.compile(r"(?<!\d)(?:[036]\d{5}|[A-Z]{2,5}\.?[A-Z]?)(?!\d)", re.I)
_COMPANY_RE = re.compile(r"[一-鿿]{2,}")
_COMPANY_SUFFIX_RE = re.compile(r"^(.{2,}?(?:银行|证券|保险|集团|股份|公司|科技|能源|汽车|医药))")
_CONNECTOR_RE = re.compile(r"和|与|及")
_STOPWORDS = {
    "请问", "多少", "是多少", "什么", "如何", "怎么", "为何", "为什么", "哪", "哪个", "哪家", "哪家公司",
    "的", "了", "呢", "吗", "一下", "其中", "分别", "营业收入", "净利润", "营收", "收入", "增长", "变化",
}


def _company_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for fragment in re.findall(r"[一-鿿]{2,}", text):
        for value in _CONNECTOR_RE.split(fragment):
            value = re.sub(r"(?:19|20)\d{2}", "", value).lstrip("年")
            suffix_match = _COMPANY_SUFFIX_RE.match(value)
            if suffix_match:
                value = suffix_match.group(1)
            for stopword in sorted(_STOPWORDS, key=len, reverse=True):
                position = value.find(stopword)
                if position >= 0:
                    value = value[:position]
            value = value.strip()
            if len(value) >= 2 and value not in candidates:
                candidates.append(value)
    return candidates


def analyze_query(query: str) -> QueryAnalysis:
    """识别查询意图，并提取年份、比例、股票代码和公司候选。"""
    if not isinstance(query, str):
        raise TypeError("query 必须是字符串")
    text = query.strip()
    if not text:
        raise ValueError("query 不能为空")
    if re.search(r"比较|相比|对比|高于|低于|哪个更|分别", text):
        intent = "comparison"
    elif re.search(r"趋势|变化|增长|下降|同比|环比|历年|近年来", text):
        intent = "trend"
    elif re.search(r"定义|含义|是什么|何谓|指什么", text):
        intent = "definition"
    else:
        intent = "factual"

    entities: list[str] = []
    for pattern in (_YEAR_RE, _PERCENT_RE, _STOCK_RE):
        for match in pattern.findall(text):
            value = match.strip().lstrip("年")
            if value and value not in entities:
                entities.append(value)
    for value in _company_candidates(text):
        if value not in entities:
            entities.append(value)
    keywords = [word for word in re.findall(r"[一-鿿]{2,}|[A-Za-z]{2,}|\d+", text)
                if word not in _STOPWORDS]
    return QueryAnalysis(intent=intent, entities=entities, keywords=keywords, original=text)


def rewrite_query(query: str, recent_turns: Iterable[tuple[str, str]] = (), *, llm=None) -> str:
    """结合最近对话主题改写指代不清的查询；可选 LLM 失败时回退规则。"""
    text = query.strip()
    if not text:
        raise ValueError("query 不能为空")
    turns = list(recent_turns)
    if llm is not None:
        try:
            rewritten = llm(text, turns)
            if isinstance(rewritten, str) and rewritten.strip():
                return rewritten.strip()
        except Exception:
            pass
    if not turns:
        return text
    previous_question, previous_answer = turns[-1]
    topic = previous_question.strip() or previous_answer.strip()
    if not topic:
        return text
    return f"{topic} {text}".strip()


def expand_query(query: str) -> list[str]:
    """生成原问题、去标点版本与关键词组合，去重并保持顺序。"""
    text = query.strip()
    if not text:
        raise ValueError("query 不能为空")
    plain = re.sub(r"[，。！？；：、“”‘’（）()\[\]{}<>《》,!?;:\"']", " ", text)
    plain = re.sub(r"\s+", " ", plain).strip()
    analysis = analyze_query(text)
    variants = [text, plain]
    if analysis.keywords:
        variants.append(" ".join(analysis.keywords))
        if analysis.entities:
            variants.append(" ".join(analysis.entities + analysis.keywords))
    return list(dict.fromkeys(value for value in variants if value))
