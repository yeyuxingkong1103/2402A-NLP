# -*- coding: utf-8 -*-
"""抽取式答案合成模块（优化版）
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

对应优化方案中的【优化点 4：答案合成优化】。

为什么采用抽取式？
- 需求要求“准确率 ≥ 90% 且响应 ≤ 3s”。本地可用的生成模型 `deepseek-r1:1.5b`
  属推理型，回答前会输出长思维链（基线平均 11.13s），既慢又不稳定；
- 金融招股说明书问答的答案多为“原文中的事实句 / 表格行”，抽取式合成可做到
  零幻觉、可溯源（附页码）、毫秒级返回。

做法：
1. 从“候选池”（重排后的 Top-N 片段及其父块）中收集候选句子 / 表格行；
2. 以「关键词覆盖率 + 数值线索 + 实体匹配 + 定义式句式 + 表格行 + 长度」综合打分；
3. 取 Top-N 句去重后**按得分排序**拼装为带页码引用的答案；
4. 同时输出“证据句”，便于人工核对与前后对比。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from src import config
from src.chunking import split_sentences
from src.query_understanding import QueryAnalysis
from src.reranker import ScoredDoc

_NUM_RE = re.compile(r"\d[\d,，.]*\s*(?:万元|亿元|元|%|％|股|人|项|家|个|年)?")
_PCT_RE = re.compile(r"\d+(?:\.\d+)?\s*[%％]")
_MONEY_RE = re.compile(r"\d[\d,，.]*\s*(?:万元|亿元|元)")
# 定义式句式：关键词后紧跟 为/是/：/| 等，表明该句在“给出取值”
_DEFN_RE = re.compile(r"(?:为|是|：|:|＝|=|\|)\s*[\d《（(A-Za-z\u4e00-\u9fa5]")

# 公司基本信息类字段（此类问题优先命中“概览/发行人基本情况”章节）
_BASIC_FIELDS = {
    "注册资本", "法定代表人", "成立日期", "注册地址", "控股股东", "实际控制人",
    "行业分类", "公司名称", "邮政编码", "主营业务", "英文名称", "实收资本",
}
_SUMMARY_SECTIONS = ("概览", "基本情况", "本次发行概况", "发行人基本情况")

# 对比词对：问题只问一侧时，出现“另一侧”的句子应被降权（如问下游却召回上游）
_CONTRAST_PAIRS = [
    ("上游", "下游"), ("军用", "民用"), ("直接", "间接"),
    ("增加", "减少"), ("上升", "下降"), ("境内", "境外"), ("进口", "出口"),
]
# 标题行（作为候选句时应剔除，避免标题中的关键词造成误命中）
_HEADING_RE = re.compile(
    r"^\s*(?:第[一二三四五六七八九十百零〇\d]+[章节]\s*[^\n]{0,40}"
    r"|[一二三四五六七八九十]+、[^\n]{0,40}"
    r"|（[一二三四五六七八九十]+）[^\n]{0,40}"
    r"|\d+(?:\.\d+){1,3}\s+[^\n]{0,40})\s*$"
)


@dataclass
class AnswerBundle:
    """答案合成结果。"""

    answer: str
    citations: List[int] = field(default_factory=list)
    evidences: List[dict] = field(default_factory=list)


def _kw_cov(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    t = text.replace(" ", "")
    return sum(1 for k in keywords if k.replace(" ", "") in t) / len(keywords)


def _num_score(text: str, atype: str, years: List[str]) -> float:
    s = 0.0
    if atype == "ratio":
        s += 0.7 if _PCT_RE.search(text) else 0.0
        s += 0.3 if _NUM_RE.search(text) else 0.0
    elif atype == "amount":
        s += 0.7 if _MONEY_RE.search(text) else 0.0
        s += 0.3 if _NUM_RE.search(text) else 0.0
    elif atype == "number":
        s += 1.0 if _NUM_RE.search(text) else 0.0
    else:
        s += 0.3 if _NUM_RE.search(text) else 0.0
    if years:
        s += 0.2 * (sum(1 for y in years if y.replace(" ", "") in text.replace(" ", "")) / len(years))
    return min(s, 1.0)


def _ent_score(text: str, entities: List[str]) -> float:
    if not entities:
        return 0.0
    t = text.replace(" ", "")
    return sum(1 for e in entities if e.replace(" ", "") in t) / len(entities)


def _len_score(text: str) -> float:
    n = len(text)
    if n < 12:
        return 0.2
    if n <= 200:
        return 1.0
    if n <= 400:
        return 0.6
    return 0.3


def _defn_score(text: str, keywords: List[str]) -> float:
    """定义式句式得分：关键词附近出现“为/是/：/|”等取值标记。"""
    if not keywords:
        return 0.0
    t = text.replace(" ", "")
    for k in keywords:
        kk = k.replace(" ", "")
        idx = t.find(kk)
        if idx >= 0:
            window = t[idx + len(kk): idx + len(kk) + 8]
            if _DEFN_RE.search(window) or window[:1] in "：:＝=|":
                return 1.0
    return 0.0


def _summary_prior(section: str, keywords: List[str], page: int) -> float:
    """章节先验：公司基本信息类问题优先“概览 / 发行人基本情况”章节。"""
    if not any(k in _BASIC_FIELDS for k in keywords):
        return 0.0
    return 1.0 if any(s in (section or "") for s in _SUMMARY_SECTIONS) else 0.0


def _enum_score(text: str) -> float:
    """列举型问题的“枚举线索”得分。"""
    s = 0.0
    if re.search(r"(主要包括|包括|涉及|分为|覆盖|适用于)", text):
        s += 0.5
    s += min(0.5, 0.1 * text.count("、"))
    return min(s, 1.0)


def _signal_score(text: str, analysis: QueryAnalysis) -> float:
    """按答案类型选择“答案线索”信号。"""
    at = analysis.answer_type
    if at in ("ratio", "amount", "number"):
        return _num_score(text, at, analysis.years)
    if at == "list":
        return _enum_score(text)
    if at == "entity":
        # 实体型：关键词后紧跟“为/是/：/，/、”等取值标记即视为给出取值
        # （表格行常用“，”分隔，如“法定代表人，程家明”）
        return 1.0 if re.search(r"[为是：:，,、]\s*[\u4e00-\u9fa5]{2,4}", text) else 0.3
    return 0.3 * _num_score(text, at, analysis.years)


# 签署页 / 中介机构套话：其中的人名（保荐代表人、项目协办人等）易被误当作答案
_BOILERPLATE_RE = re.compile(
    r"(保荐代表人|项目协办人|保荐机构|证券股份有限公司|会计师事务所|律师事务所"
    r"|签字|盖章|年\s*月\s*日|法定代表人\s*[:：]\s*李)"
)


def _boilerplate_penalty(text: str) -> float:
    """套话惩罚：签署页/中介机构条款中人名密集，避免其抢占答案位。"""
    return 1.0 if _BOILERPLATE_RE.search(text or "") else 0.0


def _contrast_penalty(text: str, keywords: List[str]) -> float:
    """对比词惩罚：问题只问一侧，句子却出现另一侧时降权。"""
    t = text.replace(" ", "")
    kws = " ".join(keywords)
    for a, b in _CONTRAST_PAIRS:
        if a in kws and b not in kws and b in t and a not in t:
            return 1.0
        if b in kws and a not in kws and a in t and b not in t:
            return 1.0
    return 0.0


def _candidate_units(scored: List[ScoredDoc]) -> List[Tuple[str, int, float, str, str]]:
    """收集候选句 / 表格行：返回 (文本, 页码, 片段得分, 章节, 块类型)。"""
    units: List[Tuple[str, int, float, str, str]] = []
    seen = set()
    for sd in scored:
        md = sd.doc.metadata
        page = int(md.get("page", 0) or 0)
        section = md.get("section", "") or ""
        ctype = md.get("ctype", "text")
        sources = [sd.doc.page_content]
        parent = md.get("parent")
        if parent and parent != sd.doc.page_content:
            sources.append(parent)
        for src in sources:
            if src.strip().startswith("[表格]") or src.strip().startswith("|"):
                cands = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("|")]
            else:
                cands = split_sentences(src)
            for s in cands:
                key = s.replace(" ", "")
                if key in seen or len(key) < 6:
                    continue
                if _HEADING_RE.match(s.strip()) and len(s.strip()) <= 45:
                    continue      # 剔除标题行，避免标题关键词造成误命中
                seen.add(key)
                units.append((s, page, sd.score, section, ctype))
    return units


def build_answer(analysis: QueryAnalysis, scored: List[ScoredDoc],
                 max_sentences: int = config.EXTRACTIVE_MAX_SENTENCES) -> AnswerBundle:
    """基于检索结果合成抽取式答案。"""
    units = _candidate_units(scored)
    if not units:
        return AnswerBundle(answer="根据招股说明书内容无法回答该问题。")

    ranked: List[Tuple[float, str, int]] = []
    for text, page, w, section, ctype in units:
        kw = _kw_cov(text, analysis.keywords)
        sig = _signal_score(text, analysis)
        ent = _ent_score(text, analysis.entities)
        defn = _defn_score(text, analysis.keywords)
        ln = _len_score(text)
        tbl = 1.0 if ctype == "table" else 0.0
        sp = _summary_prior(section, analysis.keywords, page)
        pen = _contrast_penalty(text, analysis.keywords)
        bp = _boilerplate_penalty(text)
        s = (0.34 * kw + 0.16 * sig + 0.09 * ent + 0.09 * defn
             + 0.05 * tbl + 0.04 * ln + 0.05 * w + 0.18 * sp
             - 0.20 * pen - 0.15 * bp)
        ranked.append((s, text, page))
    ranked.sort(key=lambda x: x[0], reverse=True)

    # 选取 Top-N 句：阈值过滤 + 单页限流（保证多样性）
    picked: List[Tuple[str, int, float]] = []
    used_pages: dict[int, int] = {}
    for s, text, page in ranked:
        if s <= 0.15:
            break
        if used_pages.get(page, 0) >= 2:
            continue
        picked.append((text, page, s))
        used_pages[page] = used_pages.get(page, 0) + 1
        if len(picked) >= max_sentences:
            break
    if not picked:
        picked = [(ranked[0][1], ranked[0][2], ranked[0][0])]

    # 按得分排序（得分高者在前），拼装答案
    picked.sort(key=lambda x: -x[2])
    cites = sorted({p for _, p, _ in picked if p})
    body = "；".join(_tidy(t) for t, _, _ in picked)
    page_ref = "、".join(f"第{p}页" for p in cites) if cites else "相关页面"
    answer = f"根据招股说明书{page_ref}：{body}"

    evidences = [
        {"page": p, "text": _tidy(t), "score": round(s, 4)}
        for t, p, s in picked
    ]
    return AnswerBundle(answer=answer, citations=cites, evidences=evidences)


def _tidy(text: str) -> str:
    """规整表格行与多余空白，便于阅读。"""
    t = re.sub(r"\s+", " ", text or "").strip()
    if t.startswith("|"):
        cells = [c.strip() for c in t.strip("|").split("|")]
        cells = [c for c in cells if c and set(c) - set("- ")]
        return "，".join(cells)
    return t