# -*- coding: utf-8 -*-
"""工单4 问句理解（意图 / 实体 / 主体路由 / 答案形态）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

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

WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"

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


def routing_aliases(corpus_issuers: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """构造**语料路由专用**别名表：只保留「全局唯一」的名字。

    工单7（人工智能NLP-RAG-功能测试及评估）实测坑：
        ``corpus_issuers`` 里既有文件名派生的简称（"招商银行"），也有从正文统计来的全称。
        正文统计不可避免地把**别家公司的名字**带进来 —— 例如
        「招商证券年报」正文里会提到母公司「招商银行股份有限公司」，
        「平安银行年报」里会提到集团「中国平安保险集团股份有限公司」。
        这些「跨语料重名」会让 :func:`match_corpus` 同时命中多份语料 → 按规则返回 None
        → **语料过滤整体失效**（实测：9 份语料里 5 份问句拿不到 corpus_hint）。

    规则：一个名字若出现在 ≥2 份语料的候选表里，就认为它不能用于路由（有歧义），直接剔除。
    余下的名字必然唯一指向某一份语料。

    排序：**保留各语料候选表的原有次序，不按长度重排**。原因（实测）：
        候选表由 ``build_corpus_issuers`` 构造时已把「文件名派生别名」排在前面
        （文件名是语料自身的元数据，最能代表这份语料）；
        若在此按长度重排，「中国人寿富兰克林资产管理有限公司」会压过「中国人寿」，
        导致路由得到一个语义正确但名称冗余的"发行人"。
    """
    owners: dict[str, set[str]] = {}
    for corpus, names in (corpus_issuers or {}).items():
        for name in names or []:
            key = normalize_text(name)
            if len(key) < 2:
                continue
            owners.setdefault(key, set()).add(str(corpus))
    out: dict[str, list[str]] = {}
    for corpus, names in (corpus_issuers or {}).items():
        keep: list[str] = []
        for name in names or []:
            key = normalize_text(name)
            if len(key) < 2:
                continue
            if owners.get(key) == {str(corpus)}:
                keep.append(name)
        out[str(corpus)] = keep
    return out


# ---------------------------------------------------------------------------
# 工单7（人工智能NLP-RAG-功能测试及评估）：英文题面的语料路由
# ---------------------------------------------------------------------------
# 任务书要求「支持中英文」。语料是中文年报，正文里没有英文公司名，
# 所以**中文别名表对英文问句必然失效**（实测：英文题 Q13「Postal Savings Bank of China…」
# 拿不到 corpus_hint → 退化为全语料检索）。
# 这里按**股票代码**（语料文件名的第 3 个字段，如 ``601658``）挂一小张英文别名表：
# 它是「语料元数据 → 通用英文名」的映射，不是把语料答案写进代码；
# 未登记代码的语料只是拿不到英文路由（仍可中文路由 + 全语料检索），不影响正确性。
CROSSLINGUAL_ALIASES: dict[str, tuple[str, ...]] = {
    "000001": ("Ping An Bank",),
    "601318": ("Ping An Insurance", "Ping An Insurance Group", "Ping An"),
    "600036": ("China Merchants Bank",),
    "601658": ("Postal Savings Bank of China", "Postal Savings Bank"),
    "600030": ("CITIC Securities",),
    "601628": ("China Life Insurance", "China Life"),
    "601601": ("China Pacific Insurance", "China Pacific"),
    "600999": ("China Merchants Securities",),
    "601211": ("Guotai Junan",),
}
_STOCK_CODE_RE = re.compile(r"^\d{6}$")


def crosslingual_aliases(file_name: str) -> list[str]:
    """从文件名里的 6 位股票代码取英文别名（无命中返回空表）。"""
    stem = str(file_name).rsplit(".", 1)[0]
    for part in (p.strip() for p in stem.split("__")):
        if _STOCK_CODE_RE.match(part):
            return list(CROSSLINGUAL_ALIASES.get(part, ()))
    return []


def match_corpus(question: str, corpus_issuers: Mapping[str, Sequence[str]]) -> tuple[str | None, str | None]:
    """问句点名的发行人 → 语料路由（``corpus_key``, ``issuer_name``）。

    ``corpus_issuers`` 形如 ``{"pdf1": ["武汉兴图新科电子股份有限公司", ...], "pdf2": [...]}``
    —— 由索引块内容自动统计（见 ``retriever.build_corpus_issuers``），**不得硬编码**。
    匹配规则：名字越长越优先；命中唯一文件即路由，命中多文件则返回 ``None``（不冒险）。
    工单7：入参先经 :func:`routing_aliases` 剔除「跨语料重名」，避免正文噪声让路由恒返回 None。
    """
    text = normalize_text(question or "")
    if not text:
        return None, None
    corpus_issuers = routing_aliases(corpus_issuers)
    candidates: list[tuple[int, str, str]] = []
    for corpus, names in (corpus_issuers or {}).items():
        for name in names or []:
            clean = normalize_text(name)
            # 英文别名走小写比较（normalize_text 不处理拉丁大小写）
            if re.search(r"[A-Za-z]", name or ""):
                low_name = str(name).lower()
                if len(low_name) >= 4 and low_name in str(question or "").lower():
                    candidates.append((len(low_name), corpus, name))
                continue
            # 允许问句用简称（去掉「武汉」「股份」「有限」等修饰后仍能识别主体）
            core = re.sub(r"(武汉|股份有限公司|有限公司|电子)", "", clean)
            if not core:
                continue
            # 命中判定：名字本身、或它的核心词（去公司后缀）出现在问句里。
            # 工单7 放宽点：不再要求 len(clean) >= 6 —— 本工单语料的常用简称正好是 4 字
            # （「平安银行」/「招商银行」/「中国太保」/「邮储银行」/「中信证券」…），
            # 旧阈值把它们全部挡在门外，语料路由对 9 份语料里 5 份失效。
            # 短名带来的噪声风险由两道机制兜住：①入参已过 routing_aliases 的唯一性过滤；
            # ②命中长度取 max(len(clean), len(core))，仍保持「越长越优先」的相对次序。
            hit_len = max(len(clean), len(core))
            # 工单7：核心词至少 4 个汉字才参与路由。中文公司简称的通用长度就是 4
            # （平安银行/招商银行/中国太保/邮储银行/中信证券/中国平安…）；
            # 允许更短的核心词会命中任意问句的词尾残片（实测「管理」把 Q03/Q04 路由到错误语料）。
            if len(core) >= 4 and core in text:
                candidates.append((hit_len, corpus, name))
            elif len(clean) >= 6 and clean in text:
                candidates.append((len(clean), corpus, name))
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
        # 说明：英文题面的跨语言路由不需要在这里另做一遍 ——
        # ``retriever.build_corpus_issuers`` 已把「股票代码 → 通用英文名」
        # （见 CROSSLINGUAL_ALIASES）并入同一张候选表，match_corpus 对中英文问句一视同仁。
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
