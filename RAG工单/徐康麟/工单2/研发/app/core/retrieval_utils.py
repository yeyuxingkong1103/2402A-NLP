"""检索辅助：目标数值覆盖、领域关键词命中、释义页/碎片降权（相对化加权的实现基础）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 检索优化（对应 设计/优化方案设计.md §3.3）

**为什么需要"相对化"加权**（设计取证 A.3，实测）：
全库 64.7% 的块含 ≥6 位数字字符、24.3% 含 ``%``、322 块带 ``客户`` 标签。
因此基线那种"含数字就乘 1.3、含关键词就连乘"的无差别加权 ≈ 全库整体抬升，等于没加权。
本模块改为：

- **目标数值覆盖度**：先判断问题需要哪几类数值（金额 / 百分比 / 计数），
  再看候选块覆盖了其中几类，``coverage ∈ [0,1]``；
- **关键词按命中归一**：``min(命中权重和 / 提问权重和, 1)``，避免连乘爆炸；
- **释义页降权**：``is_boilerplate`` / 前 30 页公司名反复出现 / 前 30 页短块，三者取最小值，下限 0.30。
"""

from __future__ import annotations

import re

from app.core.config import get_settings
from app.models.schemas import Chunk

#: 金额类信号词
AMOUNT_MARKERS = ("元", "万元", "亿元", "金额", "注册资本", "募集资金", "收入", "利润")
#: 百分比类信号词
PERCENT_MARKERS = ("%", "％", "占比", "比重", "比例", "百分之")
#: 计数/枚举类信号词（"分别是多少""有几个"）
#: 注：「个」曾导致「哪个技术标准」这类非数值问题被判为需要计数（Q95 实测），
#: 因此单字「个」不纳入；「项」保留（"哪几项/多少项"确属计数）。
COUNT_MARKERS = ("分别", "几个", "多少", "数量", "项", "几家", "几次")

#: 公司全称（释义页判定用）
COMPANY_NAME = "武汉兴图新科电子股份有限公司"

AMOUNT_PATTERN = re.compile(r"\d[\d,]*\.?\d*\s*(?:万元|亿元|万元人民币|元)")
PERCENT_PATTERN = re.compile(r"\d[\d,]*\.?\d*\s*(?:%|％)")
NUMBER_PATTERN = re.compile(r"\d")
#: 碎片判定：以半句衔接符开头
FRAGMENT_PREFIXES = ("、", "，", ",", "；", ";", "。", "）", ")", "”", "”")
SENTENCE_HEAD = re.compile(r"^[（(【\[]?[一二三四五六七八九十0-9]*[、.)）]?")


def required_numeric_types(question: str) -> set[str]:
    """判断问题需要哪些类型的数值（无需求时返回空集合）。"""
    text = question or ""
    needed: set[str] = set()
    if any(marker in text for marker in AMOUNT_MARKERS) or re.search(r"\d", text):
        needed.add("amount")
    if any(marker in text for marker in PERCENT_MARKERS):
        needed.add("percent")
    if any(marker in text for marker in COUNT_MARKERS):
        needed.add("count")
    return needed


def chunk_numeric_types(content: str) -> set[str]:
    """候选块包含哪些类型的数值。

    说明：``count`` 定义为"块内含 ≥3 个数字"，避免"任何含数字的块"都算覆盖
    （全库 64.7% 的块含 ≥6 位数字字符，那样等于不区分）。
    """
    text = content or ""
    found: set[str] = set()
    if AMOUNT_PATTERN.search(text) or any(marker in text for marker in ("万元", "亿元")):
        found.add("amount")
    if PERCENT_PATTERN.search(text):
        found.add("percent")
    if len(NUMBER_PATTERN.findall(text)) >= 3:
        found.add("count")
    return found


def numeric_coverage(question: str, content: str) -> float:
    """目标数值覆盖度 = 块覆盖的所需数值类型数 / 问题所需类型数。"""
    needed = required_numeric_types(question)
    if not needed:
        return 0.0
    have = chunk_numeric_types(content)
    return round(len(needed & have) / len(needed), 4)


def percent_count(text: str) -> int:
    """块内百分比个数（用于"比重分别是多少"这类多值题）。"""
    return len(PERCENT_PATTERN.findall(text or ""))


def digit_density(text: str) -> float:
    """数字字符密度（数字数 / 字符数），用于微弱修正长表霸榜。"""
    raw = text or ""
    if not raw:
        return 0.0
    return len(NUMBER_PATTERN.findall(raw)) / len(raw)


def distinct_value_counts(text: str) -> tuple[int, int]:
    """块内**去重后**的金额个数与百分比个数（"多值题"的判别信号）。"""
    raw = text or ""
    amounts = {match.group(0).replace(" ", "") for match in AMOUNT_PATTERN.finditer(raw)}
    percents = {match.group(0).replace(" ", "") for match in PERCENT_PATTERN.finditer(raw)}
    return len(amounts), len(percents)


def value_coverage_score(question: str, content: str) -> float:
    """数值型问题的"多值覆盖度"：块内去重金额/百分比越齐全，分越高（0~1）。

    为什么需要它：Q260/Q33 这类"分别是多少"的问题，答案要求**同一块内并列的全部数值**；
    实测（工单2 自测）真实证据块（4 个金额 + 4 个比重）在向量/BM25 上不占优，
    因此用"多值齐备度"作为独立的召回与重排信号。
    """
    needed = required_numeric_types(question)
    if "amount" not in needed and "percent" not in needed:
        return 0.0
    amounts, percents = distinct_value_counts(content)
    score = 0.0
    if "amount" in needed:
        score += 0.5 * min(amounts / 3.0, 1.0)
    if "percent" in needed:
        score += 0.5 * min(percents / 3.0, 1.0)
    return round(score, 4)


def question_keywords(question: str) -> list[str]:
    """问题中命中的领域关键词（权重表来自 config.retrieval.keyword_boost）。"""
    weights = get_settings().retrieval.keyword_boost
    text = question or ""
    hits = [word for word in weights if word in text]
    # 剔除被更长关键词包含的短词（如 "收入" 被 "主营业务收入" 覆盖时只保留长的）
    pruned: list[str] = []
    for word in sorted(hits, key=len, reverse=True):
        if not any(word != other and word in other for other in hits):
            pruned.append(word)
    return pruned


def matched_domain_keywords(keywords: list[str], content: str) -> list[str]:
    """候选块命中的领域关键词。"""
    text = content or ""
    return [word for word in keywords if word in text]


def keyword_boost_factor(question: str, content: str) -> tuple[float, list[str]]:
    """领域关键词相对化加权系数：``1 + max * min(命中权重和/提问权重和, 1)``。"""
    settings = get_settings().retrieval
    weights = settings.keyword_boost
    asked = question_keywords(question)
    if not asked:
        return 1.0, []
    total = sum(weights.get(word, 1.0) for word in asked) or 1.0
    hits = matched_domain_keywords(asked, content)
    if not hits:
        return 1.0, []
    hit_sum = sum(weights.get(word, 1.0) for word in hits)
    factor = 1.0 + settings.keyword_boost_max * min(hit_sum / total, 1.0)
    return round(factor, 4), hits


def table_hint(question: str) -> bool:
    """问题是否含表格类提示词（决定用 table_boost 还是 table_boost_hint）。"""
    text = question or ""
    return any(word in text for word in get_settings().retrieval.table_hint_words)


def fragment_penalty(content: str) -> float:
    """碎片降权：半句开头的块读起来不完整，作为证据应降权。"""
    settings = get_settings().retrieval
    text = (content or "").lstrip()
    if not text:
        return 1.0
    if text.startswith(FRAGMENT_PREFIXES):
        return settings.fragment_prefix_penalty
    # 以标点/连接词开头，或首句没有主语结构（前 12 字内无标点且以"的/和/与"开头）
    if text[:1] in {"的", "了", "和", "与", "及", "或"}:
        return settings.fragment_half_penalty
    return 1.0


def company_name_hits(content: str) -> int:
    """块内公司全称出现次数（释义页判定信号之一）。"""
    return (content or "").count(COMPANY_NAME)


def boilerplate_penalty(chunk: Chunk) -> tuple[float, str]:
    """释义页降权（三段取最小值，不再叠乘；下限保护真实证据）。

    Returns:
        ``(系数, 原因)``；系数 1.0 表示不降权。
    """
    settings = get_settings().retrieval
    if chunk.is_boilerplate:
        return max(settings.boilerplate_penalty, settings.boilerplate_penalty_floor), "is_boilerplate"
    if chunk.page <= settings.front_page_cutoff:
        if company_name_hits(chunk.content) >= settings.boilerplate_company_hits:
            return max(settings.boilerplate_company_penalty, settings.boilerplate_penalty_floor), "front_page_company"
        if len(chunk.content or "") < settings.boilerplate_short_chars:
            return max(settings.boilerplate_short_penalty, settings.boilerplate_penalty_floor), "front_page_short"
    return 1.0, ""
