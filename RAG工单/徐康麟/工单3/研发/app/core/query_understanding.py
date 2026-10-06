# -*- coding: utf-8 -*-
"""工单3 查询理解（设计/接口设计.md §2.4、§3.14 冻结，v1.6 调用顺序）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

**调用顺序（v1.6 冻结，多轮场景必守）**：
    ① ``classify_question(用户本轮原文)`` → ``(field_type, expects_numeric, prefer_table_retrieval)``
    ② ``rewrite_query(本轮原文, history)`` → 独立问题（**仅供检索**，不参与任何分类）
    ③ ``classify_subject_expectation(strip_issuer_names(本轮原文))`` → ``expected``

★ 数值信号与主体类型一律取自「用户本轮**原文**」，绝不取自改写后的问题；
   日志必须带 ``classified_from="original"``（否则多轮场景下同一题会被判出不同可答性，结果不可复现）。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .config import AppConfig, discover_issuer_names, get_config
from .errors import RagError
from .language import detect_language
from .text_utils import text_digest

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 数值信号词（严格口径：只认这些词与「数值型字段类型」）
NUMERIC_HINTS: tuple[str, ...] = ("多少", "几", "金额", "比重", "占比", "比例", "百分比", "数量", "数值", "分别")
# 数值型字段（答案必须含数值）
NUMERIC_FIELD_TYPES: tuple[str, ...] = ("收入金额", "收入占比", "注册资本", "发行股数", "持股比例")
# 表格型字段（宽松：追加表块检索）
TABLE_FIELD_TYPES: tuple[str, ...] = NUMERIC_FIELD_TYPES + ("募集资金", "补充流动资金", "关联方")

# 字段关键词表（顺序敏感：先匹配更具体的字段）
FIELD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "补充流动资金": ("补充流动资金", "补充营运资金"),
    "募集资金": ("募集资金", "募集资金投资项目", "募投项目", "募集资金用途"),
    "发行股数": ("发行股数", "发行股份数", "发行数量", "发行后总股本"),
    "注册资本": ("注册资本", "股本总额"),
    "法定代表人": ("法定代表人",),
    "收入占比": ("占主营业务收入", "收入占比", "占比", "比重"),
    "收入金额": ("收入", "营业收入", "销售收入", "军用领域"),
    "持股比例": ("持股比例", "持股", "股权比例"),
    "关联方": ("关联方", "关联交易", "不存在控制关系", "存在控制关系"),
    "上下游": ("上游", "下游", "供应商", "客户"),
    "技术标准": ("技术标准", "规范", "标准"),
    "荣誉奖项": ("科技进步奖", "一等奖", "荣誉", "奖项"),
    "行业地位": ("重要供应商", "行业地位", "市场地位"),
}

# 指代词（触发改写）
PRONOUN_HINTS: tuple[str, ...] = ("它", "其", "该", "此", "上述", "这个", "那个", "该公司", "那", "呢")


def _lazy_logger(logger: Any, module: str = "query_understanding") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class RewrittenQuery:
    """改写结果（设计 §2.4 冻结字段）。"""

    original: str
    rewritten: str
    changed: bool
    language: str
    field_type: str
    expects_numeric: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_question(question: str, *, logger: Any = None) -> tuple[str, bool, bool]:
    """分类问题：返回 ``(field_type, expects_numeric, prefer_table_retrieval)``。

    ★ **输入必须是用户本轮原文**（未经发行人剥离、未经改写），否则数值信号会漂移。
    """
    log = _lazy_logger(logger)
    text = str(question or "")
    field_type = "other"
    for candidate, keywords in FIELD_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            field_type = candidate
            break
    expects_numeric = any(hint in text for hint in NUMERIC_HINTS) or field_type in NUMERIC_FIELD_TYPES
    prefer_table = field_type in TABLE_FIELD_TYPES
    log.log_event("query.classify", field_type=field_type, expects_numeric=expects_numeric,
                  prefer_table_retrieval=prefer_table, classified_from="original",
                  question_digest=text_digest(text, limit=60))
    return field_type, expects_numeric, prefer_table


def needs_rewrite(question: str, history: Sequence[Any] | None) -> bool:
    """是否需要改写：有历史且（含指代词 或 长度 < 12 字）。"""
    text = str(question or "").strip()
    if not history:
        return False
    if len(text) < 12:
        return True
    return any(hint in text for hint in PRONOUN_HINTS)


def _rule_rewrite(question: str, history: Sequence[Any]) -> str:
    """规则法改写（LLM 不可用/超时时的兜底）：把上一轮的用户问题主体拼进本轮问题。"""
    previous_user = ""
    for turn in reversed(list(history or [])):
        if getattr(turn, "role", "") == "user" and str(getattr(turn, "content", "")).strip():
            previous_user = str(turn.content).strip()
            break
    if not previous_user:
        return str(question or "")
    # 取上一轮的问题作为「上下文前缀」，保留本轮原问题全文（禁止丢 token）
    return f"{previous_user} ｜ 追问：{str(question or '').strip()}"


def rewrite_query(question: str, history: Sequence[Any] | None = None, *, cfg: AppConfig | None = None,
                  llm: Any = None, logger: Any = None) -> RewrittenQuery:
    """把问题改写成可独立检索的形式（仅供检索；分类一律用原题）。"""
    config = cfg or get_config()
    log = _lazy_logger(logger)
    original = str(question or "")
    with log.enter("rewrite_query", {"question_digest": text_digest(original, limit=60),
                                     "history_turns": len(list(history or []))}) as span:
        field_type, expects_numeric, _prefer = classify_question(original, logger=log)
        language = detect_language(original)
        rewritten = original
        method = "none"
        started_ts = __import__("time").perf_counter()
        if needs_rewrite(original, history):
            history_text = "\n".join(
                f"{getattr(t, 'role', '?')}：{str(getattr(t, 'content', ''))[:200]}" for t in (history or [])
            )
            if llm is not None:
                try:
                    from pathlib import Path as _Path

                    template = (_Path(__file__).resolve().parents[1] / "prompts" / "query_rewrite_prompt.txt")
                    prompt = template.read_text(encoding="utf-8").format(history=history_text, question=original)
                    result = llm.generate_full(prompt, max_tokens=128, temperature=0.0, logger=log)
                    candidate = result.text.strip().splitlines()[0].strip() if result.text.strip() else ""
                    if candidate and len(candidate) >= 4:
                        rewritten, method = candidate, "llm"
                    else:
                        rewritten, method = _rule_rewrite(original, history or []), "rule_fallback"
                except Exception as exc:  # noqa: BLE001 —— LLM 改写失败必须回退规则法并留痕
                    log.log_event("query.rewrite_fallback", level="WARNING", method="rule",
                                  reason=f"{type(exc).__name__}: {exc}")
                    rewritten, method = _rule_rewrite(original, history or []), "rule_fallback"
            else:
                rewritten, method = _rule_rewrite(original, history or []), "rule"
        elapsed_ms = round((__import__("time").perf_counter() - started_ts) * 1000, 2)
        result = RewrittenQuery(original=original, rewritten=rewritten, changed=(rewritten != original),
                                language=language, field_type=field_type, expects_numeric=expects_numeric)
        log.log_event("query.rewrite", original=text_digest(original, limit=60),
                      rewritten=text_digest(rewritten, limit=60), changed=result.changed,
                      elapsed_ms=elapsed_ms, method=method)
        span.set_output(result.to_dict())
        return result


def strip_issuer_names(question: str, issuer_names: Sequence[str] | None) -> str:
    """把发行人**全称**替换为「该公司」（纯替换，其余 token 一律保留）。

    可逆校验由调用方负责：``any(stripped.replace("该公司", n) == original for n in issuer_names)``，
    且「该公司」出现次数 == 原文中全部发行人全称出现次数之和；校验失败即 fail-open。
    """
    text = str(question or "")
    names = [str(n) for n in (issuer_names or []) if str(n).strip()]
    for name in sorted(names, key=len, reverse=True):     # 长名优先，避免前缀误替换
        text = text.replace(name, "该公司")
    return text


def understand(question: str, history: Sequence[Any] | None = None, *, cfg: AppConfig | None = None,
               llm: Any = None, logger: Any = None) -> dict[str, Any]:
    """按 v1.6 冻结顺序执行三步，并返回一份可直接落日志的理解结果。

    ① ``classify_question(原文)`` → ② ``rewrite_query(原文, history)``（仅供检索）
    → ③ ``classify_subject_expectation(strip_issuer_names(原文))``。
    """
    from .answerability import classify_subject_expectation   # 延迟导入：答案可答性与查询理解互相引用

    config = cfg or get_config()
    log = _lazy_logger(logger)
    issuer_names = discover_issuer_names()
    original = str(question or "")
    field_type, expects_numeric, prefer_table = classify_question(original, logger=log)   # ① 原文分类
    rewritten = rewrite_query(original, history, cfg=config, llm=llm, logger=log)          # ② 改写（仅检索）
    stripped = strip_issuer_names(original, issuer_names)                                  # ③ 主体类型
    stripped_ok = any(stripped.replace("该公司", n) == original for n in issuer_names) or not issuer_names
    if not stripped_ok:
        log.log_event("query.strip_issuer", level="WARNING", count=0, reversible_ok=False,
                      degrade="剥离不可逆 → expected=any（fail-open）")
    expected = classify_subject_expectation(stripped if stripped_ok else original,
                                            issuer_names=issuer_names, logger=log)
    log.log_event("query.strip_issuer", count=stripped.count("该公司"), reversible_ok=stripped_ok)
    return {
        "original": original,
        "rewritten": rewritten.rewritten,
        "field_type": field_type,
        "expects_numeric": expects_numeric,
        "prefer_table_retrieval": prefer_table,
        "expected_subject": expected,
        "stripped_question": stripped,
        "classified_from": "original",
        "language": rewritten.language,
    }
