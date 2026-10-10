# -*- coding: utf-8 -*-
"""Query 理解模块（图像内容解析及检索优化版）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

在 02 工单基础上，新增「表格类问题」的识别与查询构造：

1. 答案类型识别新增 `table` 类型：问题询问“发行股数 / 募集资金投向 / 关联方
   持股比例 / 分年度收入及占比”等，答案通常在表格中，触发表格块加成；
2. 噪声剥离扩展至多文档（同时剔除“武汉力源信息技术股份有限公司”“力源信息”
   与“武汉兴图新科电子股份有限公司”“兴图新科”等公司套话）；
3. 查询扩展新增「字段化查询」：把问题拆为“字段词 + 数值单位”，如
   “发行股数 万股 比例 %”，用于精准命中表格的键值对块。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import jieba

from src import config

# 查询中的套话 / 噪声片段（检索时剔除，避免稀释语义）——覆盖两份招股说明书
_NOISE_PATTERNS = [
    r"武汉力源信息技术股份有限公司", r"力源信息",
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

# 表格类问题的强线索：命中即判定答案在表格中
_TABLE_HINTS = re.compile(
    r"(发行股数|发行后总股本|募集资金|拟投资|投资项目|持股比例|关联方|注册资本|"
    r"法定代表人|收入|占比|比重|比例|金额|项目名称|总投资|股东|限售)"
)

# 【本工单新增】图形类问题的强线索：命中即判定答案在图形（图像）中
# 如“组织结构图”“IC 市场应用结构与增长图”“图中可以看出”等。
_FIGURE_HINTS = re.compile(
    r"(组织结构图|组织架构图|结构图|架构图|流程图|示意图|框架图|框图|"
    r"增长图|趋势图|饼图|柱状图|条形图|折线图|柱形图|"
    r"图中|图可以|由图|如下图|上图|下图)"
)


@dataclass
class QueryAnalysis:
    """Query 理解结果。"""

    question: str
    core: str = ""                                  # 剥离噪声后的核心问句
    keywords: List[str] = field(default_factory=list)   # 检索关键词
    entities: List[str] = field(default_factory=list)   # 专有名词/实体
    years: List[str] = field(default_factory=list)      # 年份
    answer_type: str = "fact"                       # table/数值/比例/金额/实体/列举/是非/事实
    expand_queries: List[str] = field(default_factory=list)  # 多路检索查询
    doc_hint: str = ""                              # 文档级路由提示（源文件名）

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


def _doc_hint(question: str) -> str:
    """文档级路由：根据问题中的公司名，判定应检索哪份招股说明书。

    知识库收录多份同构招股说明书（都含“发行股数/募集资金/关联方”表格），
    必须用公司名做文档级消歧，否则会跨文档串味。返回源文件名（无法判定时为空）。
    """
    return _matched_alias(question)[0]


def _matched_alias(question: str) -> tuple[str, str]:
    """返回问题中命中的（源文件名, 公司别名），未命中返回 ("", "")。

    取最长别名，避免“力源”误匹配到“力源贸易”等更长实体所在文档。
    """
    q = question or ""
    best_src, best_alias = "", ""
    for source, aliases in config.DOC_COMPANY.items():
        for alias in aliases:
            if alias in q and len(alias) > len(best_alias):
                best_src, best_alias = source, alias
    return best_src, best_alias


def _answer_type(question: str) -> str:
    q = question or ""
    # 图形类问题优先判定（本工单新增）：问句指向某张图形时，答案在图内语义中
    if _FIGURE_HINTS.search(q):
        return "figure"
    # 表格类问题优先判定（本工单新增）
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
    """确定性 Query 理解（毫秒级，无需大模型）。"""

    def analyze(self, question: str) -> QueryAnalysis:
        if not question or not question.strip():
            return QueryAnalysis(question=question or "")

        core = _strip_noise(question)
        kws = _keywords(core) or _keywords(question)
        ents = _entities(question)
        # 把命中的公司名加入实体，用于区分母公司与子公司等同类表格
        # （如“注册资本”问题需排除子公司的注册资本表）。
        _, alias = _matched_alias(question)
        if alias and alias not in ents:
            ents = [alias] + ents
        years = [m for m in re.findall(r"(?:19|20)\d{2}\s*年", question)]
        atype = _answer_type(question)

        # 多路查询：核心问句 + 关键词组合 + 实体增强 + 字段化查询
        expand: List[str] = []
        if kws:
            expand.append(" ".join(kws[:8]))
        if ents:
            expand.append(" ".join(ents[:5]) + " " + " ".join(kws[:5]))
        if atype in ("ratio", "amount", "number", "table") and kws:
            expand.append(" ".join(kws[:6]) + " 万元 %")
        # 字段化查询：命中表格类问题时，构造“字段 单位”形式，贴近键值对块
        if atype == "table" and kws:
            expand.append(" ".join(kws[:4]) + " 项目名称 金额 比例")
        # 图形化查询：命中图形类问题时，补充“图形语义”关键词，贴近图像语义块
        if atype == "figure" and kws:
            expand.append(" ".join(kws[:6]) + " 组织结构图 构成")
            expand.append(" ".join(kws[:6]) + " 由图可知 增长率 占比")

        return QueryAnalysis(
            question=question, core=core, keywords=kws, entities=ents,
            years=years or [], answer_type=atype, expand_queries=expand,
            doc_hint=_doc_hint(question),
        )


if __name__ == "__main__":
    qu = QueryUnderstanding()
    for q in [
        "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
        "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？",
        "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    ]:
        a = qu.analyze(q)
        print(a.answer_type, "|", a.core)
        print("   kw:", a.keywords)
        print("   qs:", a.retrieval_queries)