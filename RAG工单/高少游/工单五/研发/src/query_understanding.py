# -*- coding: utf-8 -*-
"""单轮 Query 理解：噪声剥离、关键词、实体、同义词扩展、答案类型与文档路由。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

在 02~04 工单基础上做「Query 理解优化」：
    1. 噪声剥离覆盖两份招股说明书的公司套话；
    2. 实体抽取覆盖公司名 / 工程名 / 标准名 / 领域名；
    3. 答案类型判定扩展「数值 / 比例 / 列举 / 实体 / 表格 / 图形」等；
    4. 文档级路由（公司名 → 源文件名），避免多文档串味；
    5. **同义词扩展**（本工单新增）：招股说明书中同一事实存在多种表述
       （「军用领域」=「国防客户」/「军方市场」，「收入」=「销售额」），
       仅靠字面匹配会漏召回，故用同义词表构造扩展查询，实现跨表述召回。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import jieba

from src import config

_NOISE_PATTERNS = [
    r"武汉力源信息技术股份有限公司", r"力源信息",
    r"武汉兴图新科电子股份有限公司", r"兴图新科",
    r"根据[^，。？]*招股意向书[，,]?", r"根据[^，。？]*招股说明书[，,]?",
    r"招股意向书", r"招股说明书", r"报告期内[，,]?", r"请(问|告知)",
    r"分别是多少", r"是多少", r"有哪些", r"是什么", r"涉及哪些", r"主要包括哪些",
    r"的", r"了",
]

_STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "是", "在", "有", "为", "对", "从", "到",
    "多少", "哪些", "哪个", "什么", "分别", "根据", "关于", "以及", "主要", "包括",
    "涉及", "公司", "报告", "期内", "期间", "本次", "发行", "请问", "如何", "怎么",
    "相关", "情况", "进行", "已经", "成为", "使用", "计划", "万元", "亿元", "其中",
}

_TABLE_HINTS = re.compile(
    r"(发行股数|发行后总股本|募集资金|拟投资|投资项目|持股比例|关联方|注册资本|"
    r"法定代表人|收入|占比|比重|比例|金额|项目名称|总投资|股东|限售|组织结构)"
)
_FIGURE_HINTS = re.compile(
    r"(组织结构图|组织架构图|结构图|架构图|流程图|示意图|框架图|框图|"
    r"增长图|趋势图|饼图|柱状图|条形图|折线图|柱形图|图中|图可以|由图)"
)


@dataclass
class QueryAnalysis:
    """Query 理解结果。"""
    question: str
    core: str = ""
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    years: List[str] = field(default_factory=list)
    answer_type: str = "fact"
    expand_queries: List[str] = field(default_factory=list)
    doc_hint: str = ""
    synonyms: List[str] = field(default_factory=list)   # 同义词（用于检索扩展与重排）

    @property
    def retrieval_queries(self) -> List[str]:
        qs = [self.core or self.question]
        qs.extend(self.expand_queries)
        seen, out = set(), []
        for q in qs:
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out


def strip_noise(question: str) -> str:
    text = question or ""
    for pat in _NOISE_PATTERNS:
        text = re.sub(pat, "", text)
    return re.sub(r"[，,。？?！!、\s]+", " ", text).strip()


def keywords_of(question: str) -> List[str]:
    toks = []
    for w in jieba.cut(question or ""):
        w = w.strip()
        if not w or w in _STOPWORDS:
            continue
        if len(w) == 1 and not w.isdigit() and not re.match(r"[A-Za-z]", w):
            continue
        toks.append(w)
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def entities_of(question: str) -> List[str]:
    ents = re.findall(r"[《“\"]([^》”\"]{2,30})[》”\"]", question or "")
    ents += re.findall(
        r"[\u4e00-\u9fa5A-Za-z]{2,}(?:领域|行业|系统|工程|标准|技术规范|平台|市场|公司|部|处)",
        question or "")
    return [e for e in dict.fromkeys(ents) if e and e not in _STOPWORDS]


def synonyms_of(question: str) -> List[str]:
    """按问题中出现的业务词，查同义词表，返回跨表述扩展词。"""
    out: List[str] = []
    q = question or ""
    for term, syns in config.SYNONYMS.items():
        if term in q:
            for s in syns:
                if s not in out:
                    out.append(s)
    return out


def matched_alias(question: str) -> tuple[str, str]:
    """返回（源文件名, 公司别名），取最长匹配避免歧义。"""
    q = question or ""
    best_src, best_alias = "", ""
    for source, aliases in config.DOC_COMPANY.items():
        for alias in aliases:
            if alias in q and len(alias) > len(best_alias):
                best_src, best_alias = source, alias
    return best_src, best_alias


def doc_hint_of(question: str) -> str:
    return matched_alias(question)[0]


def answer_type_of(question: str) -> str:
    q = question or ""
    if _FIGURE_HINTS.search(q):
        return "figure"
    if _TABLE_HINTS.search(q):
        return "table"
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
    """确定性单轮 Query 理解（毫秒级）。"""

    def analyze(self, question: str) -> QueryAnalysis:
        if not question or not question.strip():
            return QueryAnalysis(question=question or "")
        core = strip_noise(question)
        kws = keywords_of(core) or keywords_of(question)
        ents = entities_of(question)
        _, alias = matched_alias(question)
        if alias and alias not in ents:
            ents = [alias] + ents
        years = re.findall(r"(?:19|20)\d{2}\s*年", question)
        atype = answer_type_of(question)
        syns = synonyms_of(question)

        expand: List[str] = []
        if kws:
            expand.append(" ".join(kws[:8]))
        if ents:
            expand.append(" ".join(ents[:5]) + " " + " ".join(kws[:5]))
        # 同义词扩展：把跨表述词并入检索查询（如「国防客户 销售额」召回军用收入句）
        if syns:
            expand.append(" ".join(syns[:6]) + " " + " ".join(kws[:4]))
        if atype in ("ratio", "amount", "number", "table") and kws:
            expand.append(" ".join(kws[:6]) + " 万元 %")
        if atype == "figure" and kws:
            expand.append(" ".join(kws[:6]) + " 组织结构图 构成 销售处")
        return QueryAnalysis(question=question, core=core, keywords=kws, entities=ents,
                             years=years, answer_type=atype, expand_queries=expand,
                             doc_hint=doc_hint_of(question), synonyms=syns)


if __name__ == "__main__":
    qu = QueryUnderstanding()
    for q in ["报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
              "这个公司的法定代表人是谁？",
              "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？"]:
        a = qu.analyze(q)
        print(a.answer_type, "|", a.core, "|", a.doc_hint, "| 同义词:", a.synonyms)