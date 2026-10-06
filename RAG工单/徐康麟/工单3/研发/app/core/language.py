# -*- coding: utf-8 -*-
"""工单3 语言判定与引用标签（设计/接口设计.md §3.15 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

判定规则（实测口径）：
    * 中文字符占比 ≥ 0.15 → ``zh``；
    * 拉丁字母占比 ≥ 0.6 且中文占比 < 0.05 → ``en``；
    * 其余 → ``mixed``。
引用标签：中文用 ``[页码: N]``，英文用 ``[Page: N]``；两者都允许 ``[文件名: 页码]`` 全称形式（§3.17）。
"""

from __future__ import annotations

import re

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """判定文本语言：``zh`` / ``en`` / ``mixed``。"""
    body = str(text or "")
    if not body.strip():
        return "zh"
    cjk = len(_CJK.findall(body))
    latin = len(_LATIN.findall(body))
    total = max(len(body.strip()), 1)
    cjk_ratio = cjk / total
    latin_ratio = latin / total
    if cjk_ratio >= 0.15:
        return "zh"
    if latin_ratio >= 0.6 and cjk_ratio < 0.05:
        return "en"
    return "mixed"


def is_english_question(text: str) -> bool:
    """是否为英文提问（决定作答语言与引用标签）。"""
    return detect_language(text) == "en"


def answer_language(question: str) -> str:
    """目标作答语言：英文提问 → ``en``；其余（含中英混合）→ ``zh``。"""
    return "en" if is_english_question(question) else "zh"


# ---------------------------------------------------------------------------
# t21（§24）：英文问句 → 中文语料词汇的**通用映射表**（跨语言主题落地用）
# ---------------------------------------------------------------------------
# 用途：中文语料里没有英文词，英文问句的「主题」无法用词面覆盖率衡量。
# 本表把**招股书领域的通用英文问法**映射到语料里真实存在的**中文词汇**（不是答案、也不是某条
# 具体问句的关键词），供可答性闸门做「问题主题是否落在证据里」的跨语言判定。
# 匹配：英文侧按短语做**不区分大小写的包含匹配**（长短语优先），中文侧按语料原文包含匹配。
EN_ZH_TOPIC_TERMS: dict[str, tuple[str, ...]] = {
    "registered capital": ("注册资本", "股本"),
    "legal representative": ("法定代表人",),
    "actual controller": ("实际控制人",),
    "controlling shareholder": ("控股股东",),
    "board of directors": ("董事",),
    "supervisory board": ("监事",),
    "senior management": ("高级管理人员",),
    "number of employees": ("员工", "职工"),
    "date of establishment": ("成立日期", "成立时间"),
    "registered address": ("注册地址", "注册地"),
    "registered office": ("注册地址", "注册地"),
    "stock code": ("股票代码", "证券代码"),
    "main business": ("主营业务",),
    "business scope": ("经营范围",),
    "operating revenue": ("营业收入",),
    "net profit": ("净利润",),
    "gross margin": ("毛利率",),
    "asset liability ratio": ("资产负债率",),
    "research and development": ("研发",),
    "core technology": ("核心技术",),
    "patents": ("专利",),
    "software copyright": ("软件著作权",),
    "raised funds": ("募集资金",),
    "investment projects": ("投资项目", "募投项目"),
    "working capital": ("流动资金",),
    "related party": ("关联方",),
    "related transactions": ("关联交易",),
    "guarantees": ("担保",),
    "litigation": ("诉讼",),
    "arbitration": ("仲裁",),
    "subsidiaries": ("子公司",),
    "customers": ("客户",),
    "suppliers": ("供应商",),
    "competition": ("竞争",),
    "market share": ("市场占有率", "市场份额"),
    "industry chain": ("产业链",),
    "upstream": ("上游",),
    "downstream": ("下游",),
    "major shareholder": ("股东",),
    "share issue": ("发行", "股份"),
    "initial public offering": ("首次公开发行", "发行"),
    "issuer": ("发行人",),
    "lock-up period": ("锁定期",),
    "dividend": ("分红", "股利"),
    "orders": ("订单",),
    "capacity": ("产能",),
    "qualification": ("资质", "认证"),
    "awards": ("奖项", "荣誉", "科技进步奖"),
    "risk factors": ("风险",),
    "fundraising purpose": ("募集资金用途", "募集资金"),
    "supplementary working capital": ("补充流动资金",),
    "important supplier": ("重要供应商",),
    "supplier": ("供应商",),
    "products": ("产品",),
    "technology standard": ("技术标准", "规范"),
    "shares": ("股份", "发行", "发行股数", "发行数量"),
    "field": ("领域",),
}


def match_corpus_topic_terms(question: str) -> list[tuple[str, tuple[str, ...]]]:
    """把英文问句映射到语料词汇：返回 ``[(英文短语, 候选中文词...), ...]``（长短语优先、去重保序）。

    只做**领域通用词汇**级映射（不翻译、不针对任何具体问句写死答案）。
    """
    text = str(question or "").lower()
    matched: list[tuple[str, tuple[str, ...]]] = []
    for phrase in sorted(EN_ZH_TOPIC_TERMS, key=len, reverse=True):
        if phrase in text:
            matched.append((phrase, EN_ZH_TOPIC_TERMS[phrase]))
    return matched


def citation_label(language: str) -> tuple[str, str]:
    """引用标签对：``zh`` → ``("页码", "页")``；``en`` → ``("Page", "Page")``。"""
    if str(language).lower().startswith("en"):
        return ("Page", "Page")
    return ("页码", "页")


# ---------------------------------------------------------------------------
# t22（§25）：英文作答支撑（提示词语言块 / 英文字段标签 / 拉丁占比 / 英文框架句）
# ---------------------------------------------------------------------------
# 提示词语言块：**zh 分支的文本必须与改造前 `qa_prompt.txt` 规则 6 的渲染结果逐字符相同**
# （该文件是中英共用文件；只有 en 分支才换成强约束英文指令）。
LANGUAGE_RULE_ZH = "`zh` 为 `en` 时用英文作答（引用写 `[Page: N]`），否则中文作答"
LANGUAGE_RULE_EN = ("**Must answer in English.** Write the answer as English sentences; keep ONLY proper nouns "
                    "(company/person names), numbers, units and any verbatim Chinese source fragment in Chinese; "
                    "never copy a whole Chinese sentence as the answer; citations use `[Page: N]`.")

# 英文字段标签（招股书常见字段；用于把「字段+取值」渲染成英文框架句）。键为中文字段词（与
# `query_understanding.FIELD_KEYWORDS` 同源词），值是通用英文标签——不针对任何具体问题写死答案。
EN_FIELD_LABELS: dict[str, str] = {
    "注册资本": "registered capital", "股本总额": "total share capital", "法定代表人": "legal representative",
    "注册地址": "registered address", "成立日期": "date of establishment", "发行股数": "number of shares issued",
    "发行数量": "number of shares issued", "持股比例": "shareholding percentage", "营业收入": "operating revenue",
    "收入": "revenue", "募集资金": "raised funds", "补充流动资金": "supplementary working capital",
    "关联方": "related parties", "控股股东": "controlling shareholder", "实际控制人": "actual controller",
    "员工人数": "number of employees", "主营业务": "main business", "重要供应商": "important supplier",
    "供应商": "supplier", "客户": "customer", "上游": "upstream industry", "下游": "downstream industry",
    "技术标准": "technical standard", "科技进步奖": "National Science and Technology Progress Award",
    "证券代码": "stock code", "股票代码": "stock code",
}


# 答案里**必要的**中文成分：逐字原文片段 `（verbatim source: 「…」）` / `(verbatim source: "…")`
_VERBATIM_SOURCE_PATTERN = re.compile(r"[（(]\s*verbatim source\s*[:：]\s*[「\"“][^」\"”]*[」\"”]\s*[）)]")


def english_ratio(text: str, *, keep_verbatim: bool = False) -> float:
    """文本里拉丁字母占比（t22 英文语言闸门的度量）。

    验收判据为「正文以英文为主（**中文仅限专名、金额单位、引用的原文片段等必要之处**）」，
    因此默认**排除**为避免编造而保留的逐字原文片段 `（verbatim source: 「…」）`；
    `keep_verbatim=True` 时按包含片段与引用行的全文本计算（更严口径，供对照）。
    """
    body = str(text or "").strip()
    if not keep_verbatim:
        body = _VERBATIM_SOURCE_PATTERN.sub("", body).strip()
    if not body:
        return 0.0
    letters = sum(1 for char in body if ("a" <= char.lower() <= "z"))
    return round(letters / len(body), 4)


def english_frame(field_zh: str, value_zh: str) -> str:
    """把「字段 + 取值」渲染成英文框架句，取值逐字保留（专名/金额/单位按原文）。

    找不到字段标签时退化为 ``The value is …``（仍是英文句，且不新增事实）。
    """
    label = EN_FIELD_LABELS.get(str(field_zh).strip())
    value = str(value_zh).strip()
    return f"The {label} is {value}." if label else f"The value is {value}."
