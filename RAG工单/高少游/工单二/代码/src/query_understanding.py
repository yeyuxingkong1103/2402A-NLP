# -*- coding: utf-8 -*-
"""Query 理解模块（优化版）
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

对应优化方案中的【优化点 3-1：查询侧优化】。基线把整句问题直接丢给检索，
存在两个问题：① 高频公司名/套话（“武汉兴图新科电子股份有限公司”“根据……
招股意向书”）占据语义权重，稀释真正关键的实词；② 数字型、比例型问题的
答案类型未被识别，无法做针对性重排。

本模块提供确定性的 Query 理解（不依赖大模型，毫秒级完成）：

1. 噪声剥离：剔除公司全称、文件名、套话，得到“核心问句”；
2. 关键词抽取：基于 jieba 抽取实词（去停用词），并保留年份 / 数字 / 单位；
3. 答案类型识别：数值型 / 比例型 / 金额型 / 实体型 / 列举型 / 是非型；
4. 查询扩展：生成“核心问句 + 关键词组合”的多路查询，提升召回。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import jieba

# 查询中的套话 / 噪声片段（检索时剔除，避免稀释语义）
_NOISE_PATTERNS = [
    r"武汉兴图新科电子股份有限公司", r"兴图新科",
    r"根据[^，。？]*招股意向书[，,]?", r"根据[^，。？]*招股说明书[，,]?",
    r"招股意向书", r"招股说明书", r"报告期内[，,]?", r"请(问|告知)",
    r"分别是多少", r"分别是多少？", r"是多少", r"有哪些", r"是什么",
    r"涉及哪些", r"主要包括哪些", r"占", r"的", r"了", r"在",
]

_STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "是", "在", "有", "为", "对", "从", "到",
    "多少", "哪些", "哪个", "什么", "分别", "根据", "关于", "以及", "主要", "包括",
    "涉及", "公司", "报告", "期内", "期间", "本次", "发行", "请问", "如何", "怎么",
    "相关", "情况", "进行", "已经", "成为", "使用", "计划", "万元", "亿元", "其中",
}

# 答案类型判定线索
_NUM_UNIT = re.compile(r"(万元|亿元|元|%|％|股|人|项|家|个|年|月|日)")
_YEAR_RE = re.compile(r"(19|20)\d{2}\s*年")


@dataclass
class QueryAnalysis:
    """Query 理解结果。"""

    question: str
    core: str = ""                                  # 剥离噪声后的核心问句
    keywords: List[str] = field(default_factory=list)   # 检索关键词
    entities: List[str] = field(default_factory=list)   # 专有名词/实体
    years: List[str] = field(default_factory=list)      # 年份
    answer_type: str = "fact"                       # 数值/比例/金额/实体/列举/是非/事实
    expand_queries: List[str] = field(default_factory=list)  # 多路检索查询

    @property
    def retrieval_queries(self) -> List[str]:
        qs = [self.core or self.question]
        qs.extend(self.expand_queries)
        # 去重保序
        seen, out = set(), []
        for q in qs:
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out


def _strip_noise(question: str) -> str:
    text = question or ""
    for pat in _NOISE_PATTERNS:
        text = re.sub(pat, "", text)
    text = re.sub(r"[，,。？?！!、\s]+", " ", text).strip()
    return text


def _keywords(question: str) -> List[str]:
    toks = []
    for w in jieba.cut(question or ""):
        w = w.strip()
        if not w or w in _STOPWORDS:
            continue
        if len(w) == 1 and not w.isdigit() and not re.match(r"[A-Za-z]", w):
            continue
        toks.append(w)
    # 去重保序
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _entities(question: str) -> List[str]:
    """抽取公司名、标准名、工程名、领域名等实体。"""
    ents = re.findall(r"[《“\"]([^》”\"]{2,30})[》”\"]", question or "")
    ents += re.findall(r"[\u4e00-\u9fa5A-Za-z]{2,}(?:领域|行业|系统|工程|标准|技术规范|平台|市场)", question or "")
    return [e for e in dict.fromkeys(ents) if e and e not in _STOPWORDS]


def _answer_type(question: str) -> str:
    q = question or ""
    if re.search(r"(比重|占比|比例|百分比)", q):
        return "ratio"
    if re.search(r"(金额|万元|亿元|募集资金|收入|利润|注册资本|资本)", q):
        return "amount"
    if re.search(r"(多少|几个|几项|数量|人数|年份)", q):
        return "number"
    if re.search(r"(哪些|主要包括|包括哪些|涉及哪些|哪些行业|哪些企业)", q):
        return "list"
    if re.search(r"(是否|是不是|有没有)", q):
        return "boolean"
    if re.search(r"(谁|法定代表人|董事长|总经理|负责人)", q):
        return "entity"
    return "fact"


class QueryUnderstanding:
    """确定性 Query 理解（毫秒级，无需大模型）。"""

    def analyze(self, question: str) -> QueryAnalysis:
        if not question or not question.strip():
            return QueryAnalysis(question=question or "")

        core = _strip_noise(question)
        kws = _keywords(core) or _keywords(question)
        ents = _entities(question)
        years = _YEAR_RE.findall(question) and [m for m in re.findall(r"(?:19|20)\d{2}\s*年", question)]
        atype = _answer_type(question)

        # 多路查询：核心问句 + 关键词组合 + 实体增强
        expand: List[str] = []
        if kws:
            expand.append(" ".join(kws[:8]))
        if ents:
            expand.append(" ".join(ents[:5]) + " " + " ".join(kws[:5]))
        if atype in ("ratio", "amount", "number") and kws:
            expand.append(" ".join(kws[:6]) + " 万元 %")

        return QueryAnalysis(
            question=question, core=core, keywords=kws, entities=ents,
            years=years or [], answer_type=atype, expand_queries=expand,
        )


if __name__ == "__main__":
    qu = QueryUnderstanding()
    for q in [
        "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "武汉兴图新科电子股份有限公司注册资本是多少？",
        "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
    ]:
        a = qu.analyze(q)
        print(a.answer_type, "|", a.core)
        print("   kw:", a.keywords)
        print("   qs:", a.retrieval_queries)