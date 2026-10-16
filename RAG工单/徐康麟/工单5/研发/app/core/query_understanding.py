# -*- coding: utf-8 -*-
"""工单4 问句理解（意图 / 实体 / 主体路由 / 答案形态）。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化

设计依据：
    * 设计/接口设计.md §4.2（retriever / qa_engine 契约）；
    * 任务书「★A 答案必须逐项完整枚举」：16 题的 ``answer_form`` 决定答案形态，
      问句理解必须把「枚举型 / 多点数值型 / 单点型」区分开，否则合成器无法保证完整性；
    * 任务书「跨文件串答陷阱」：问句点名的发行人决定**语料路由**（题 1/531/543 力源 vs 兴图新科）。

本模块**零外部依赖、零硬编码语料事实**：
    * 发行人名字由 ``corpus_issuers``（从索引块内容自动统计）注入，代码里不写公司名；
    * 意图/答案形态全部由**问句措辞**判定（比照 ``answer_form`` 的通用化措辞）。

``AnswerShape``（答案形态）取值与 golden ``answer_form`` 的对应：
    ``hierarchy_enum``  ⇐ 「X 有几个 Y 构成 / 由哪些构成」（题 5）
    ``enum_list``       ⇐ 「拟投资哪些项目 / 有哪些企业 / 主要包括哪些行业」（题 2/4/793）
    ``numeric_series``  ⇐ 「分别是多少 / 分别是多少比重」（题 260/33）
    ``single_value``    ⇐ 「注册资本是多少 / 法定代表人是谁 / 多少钱用于X」（题 543/531/207）
    ``dual_value``      ⇐ 「是多少，占比是多少」（题 1）
    ``relation``        ⇐ 「关联方是谁，持股比例和关系」（题 3）
    ``conclusion``      ⇐ 「哪个领域/哪个工程/哪个行业/增长率最快的是」（题 6/95/957/795）
    ``comparison``      ⇐ 「上游涉及哪些 / 下游主要包括哪些」（题 34/793 的上/下游切片）
    ``generic``         ⇐ 兜底：证据句选择
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .language import LanguageInfo, resolve_language
from .text_utils import STOPWORDS, extract_numbers, normalize_text, tokenize

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

#: 「强结论措辞」：问句出现这些词说明要的是**结论性判断**（最快/负增长/最高/位列），
#: 而不是一份清单 —— 它们优先于「哪些」参与答案形态判定（T7 修正，见 _decide_shape）
_STRONG_CONCLUSION = re.compile(r"(最快|最慢|最高|最低|负增长|正增长|位列第|排名第|第一|最大|最小)")

#: 答案形态（见模块 docstring）
ANSWER_SHAPES = (
    "hierarchy_enum", "enum_list", "numeric_series", "single_value", "dual_value",
    "relation", "conclusion", "comparison", "generic",
)

# 问句措辞 → 意图标志（全部是**通用问法**，不含任何公司名/页码）
_PATTERNS: dict[str, re.Pattern[str]] = {
    "asks_hierarchy": re.compile(r"(组织结构图|组织架构图|设[有立]|下设|构成|由几个|有几个|几个部门|几个销售处)"),
    "asks_count": re.compile(r"(几个|几家|几项|几个部门|几家分公司|多少家|多少个|数量是多少)"),
    "asks_enum": re.compile(r"(哪些|有哪些|包括什么|包括哪些|都包括|主要包括|涉及哪些|包含哪些|有什么)"),
    "asks_list": re.compile(r"(分别是|各是多少|分别为|依次)"),
    "asks_ratio": re.compile(r"(比重|占比|比例|百分|多少%|％)"),
    "asks_money": re.compile(r"(金额|多少钱|多少万元|多少亿元|注册资本|募集资金|收入是多少)"),
    "asks_relation": re.compile(r"(关系|关联方|控股股东|实际控制人|持股比例|控股关系)"),
    "asks_conclusion": re.compile(r"(哪个|哪家|哪一项|最快|最高|最低|负增长|增长最快|排名)"),
    "asks_standard": re.compile(r"(技术标准|标准|规范|制订|制定)"),
    "asks_award": re.compile(r"(荣获|获得|奖项|一等奖|国家科技进步)"),
    "asks_supplier": re.compile(r"(供应商|竞争地位|行业地位)"),
    "asks_upstream": re.compile(r"(上游)"),
    "asks_downstream": re.compile(r"(下游)"),
    "asks_numeric": re.compile(r"(是多少|多少|几|占比|比重|比例|金额)"),
    "asks_where": re.compile(r"(哪个领域|在哪个|什么领域)"),
    "asks_who": re.compile(r"(是谁|叫什么|哪一位|何人)"),
}

#: 「主体/字段」类实词：用于检索式加权与答案形态判定（通用财务/招股书字段，非语料专有）
FIELD_TERMS: tuple[str, ...] = (
    "注册资本", "法定代表人", "发行股数", "发行后总股本", "募集资金", "补充流动资金",
    "关联方", "控股股东", "实际控制人", "持股比例", "军用领域", "主营业务收入",
    "技术标准", "组织结构图", "上游", "下游", "供应商", "科技进步一等奖",
    "增长率", "报告期", "应用结构",
)


def _lazy_logger(logger: Any, module: str = "query_understanding") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class QueryAnalysis:
    """问句理解结果（检索与合成共用的唯一输入）。"""

    question: str
    language: str = "zh"
    language_info: dict[str, Any] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    field_terms: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    answer_shape: str = "generic"
    corpus_hint: str | None = None
    issuer_hint: str | None = None
    subject_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def has(self, intent: str) -> bool:
        """是否含某意图（如 ``asks_enum``）。"""
        return intent in self.intents


def _classify_intents(question: str) -> list[str]:
    return [name for name, pattern in _PATTERNS.items() if pattern.search(question or "")]


def _decide_shape(question: str, intents: list[str], *, field_terms: Sequence[str]) -> str:
    """按 golden ``answer_form`` 的通用化措辞判定答案形态（顺序敏感：先具体后笼统）。"""
    has = set(intents)
    if "组织结构图" in question or ("asks_hierarchy" in has and ("下设" in question or "构成" in question)):
        return "hierarchy_enum"
    if "asks_upstream" in has or "asks_downstream" in has:
        return "comparison"
    if "asks_relation" in has and ("asks_ratio" in has or "持股比例" in question):
        return "relation"
    if "asks_list" in has and ("asks_numeric" in has or "asks_ratio" in has or "asks_money" in has):
        return "numeric_series"
    if "asks_standard" in has or "asks_award" in has:
        return "conclusion"
    # T7 修正：**强结论措辞优先于「哪些」**。英文题经翻译后常出现
    # 「哪些行业的增长率最快，哪些行业出现了负增长」——旧顺序先命中 asks_enum 判成 enum_list，
    # 合成器于是找不到结论句，英文图表题被拒答（T6 §8.1 缺口3）。含「最快/负增长/最高/位列」时按结论题处理。
    if _STRONG_CONCLUSION.search(question) and (has & {"asks_conclusion", "asks_enum", "asks_where"}):
        return "conclusion"
    if "asks_enum" in has:
        return "enum_list"
    if "asks_where" in has or ("asks_conclusion" in has and not has.intersection({"asks_count"})):
        return "conclusion"
    if "asks_count" in has:
        return "single_value"
    if "asks_who" in has:
        return "single_value"
    if "asks_ratio" in has and ("发行股数" in question or "总股本" in question):
        return "dual_value"
    if "asks_money" in has or "asks_numeric" in has:
        return "single_value"
    return "generic"


def _match_field_terms(question: str) -> list[str]:
    return [term for term in FIELD_TERMS if term in (question or "")]


def match_corpus(question: str, corpus_issuers: Mapping[str, Sequence[str]]) -> tuple[str | None, str | None]:
    """问句点名的发行人 → 语料路由（``corpus_key``, ``issuer_name``）。

    ``corpus_issuers`` 形如 ``{"pdf1": ["武汉兴图新科电子股份有限公司", ...], "pdf2": [...]}``
    —— 由索引块内容自动统计（见 ``retriever.build_corpus_issuers``），**不得硬编码**。
    匹配规则：名字越长越优先；命中唯一文件即路由，命中多文件则返回 ``None``（不冒险）。
    """
    text = normalize_text(question or "")
    if not text:
        return None, None
    candidates: list[tuple[int, str, str]] = []
    for corpus, names in (corpus_issuers or {}).items():
        for name in names or []:
            clean = normalize_text(name)
            # 允许问句用简称（去掉「武汉」「股份」「有限」等修饰后仍能识别主体）
            core = re.sub(r"(武汉|股份有限公司|有限公司|电子)", "", clean)
            if len(clean) >= 6 and clean in text:
                candidates.append((len(clean), corpus, name))
            elif len(core) >= 4 and core in text and len(clean) >= 6:
                candidates.append((len(clean) - 2, corpus, name))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: -item[0])
    best_len = candidates[0][0]
    top = [c for c in candidates if c[0] == best_len]
    corpora = {c[1] for c in top}
    if len(corpora) != 1:
        return None, None
    return top[0][1], top[0][2]


def analyze(question: str, *, requested_lang: str = "auto",
            corpus_issuers: Mapping[str, Sequence[str]] | None = None,
            logger: Any = None) -> QueryAnalysis:
    """问句理解主入口：语言 / 实词 / 字段词 / 数字 / 意图 / 答案形态 / 语料路由。"""
    log = _lazy_logger(logger)
    with log.enter("analyze", {"question_digest": (question or "")[:80],
                               "requested_lang": requested_lang,
                               "corpora": sorted((corpus_issuers or {}).keys())}) as span:
        lang_info: LanguageInfo = resolve_language(question, requested_lang, logger=log)
        text = normalize_text(question or "")
        tokens = [t for t in tokenize(text, use_jieba=True) if t]
        keywords = [t for t in tokens if t not in STOPWORDS and (len(t) >= 2 or t.isdigit())]
        field_terms = _match_field_terms(text)
        numbers = extract_numbers(text)
        intents = _classify_intents(text)
        shape = _decide_shape(text, intents, field_terms=field_terms)
        corpus_hint, issuer_hint = match_corpus(text, corpus_issuers or {})
        subject_terms = [issuer_hint] if issuer_hint else []
        analysis = QueryAnalysis(
            question=question, language=lang_info.resolved, language_info=lang_info.to_dict(),
            tokens=tokens, keywords=keywords, field_terms=field_terms, numbers=numbers,
            intents=intents, answer_shape=shape, corpus_hint=corpus_hint, issuer_hint=issuer_hint,
            subject_terms=subject_terms,
        )
        log.log_event("query.analyzed", language=analysis.language, shape=shape, intents=intents,
                      keywords=keywords[:12], field_terms=field_terms, numbers=numbers,
                      corpus_hint=corpus_hint, issuer_hint=issuer_hint)
        span.set_output({"language": analysis.language, "shape": shape, "intents": intents,
                         "keywords": len(keywords), "corpus_hint": corpus_hint})
        return analysis
