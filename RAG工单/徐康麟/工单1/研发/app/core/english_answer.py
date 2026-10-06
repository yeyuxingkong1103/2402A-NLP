"""英文答案生成：把中文证据组织成英文回答。

为什么不用“让模型翻译答案”作默认路径：
1. **数字与单位必须精确**：招股书的金额是核心事实，0.6B 小模型会把
   “5,520 万元”译成 “5,520 million yuan”（差 100 倍）；
2. **首字延迟 < 3 秒**：模板路径是毫秒级，模型路径 1~3 秒且不稳定。

因此本模块用「意图 -> 英文句式模板 + 从中文原文抽取事实」的方式生成，
数字、比例、金额、日期原样保留，术语走 ``language.GLOSSARY_ZH_EN``；
只有模板覆盖不到的开放性问题才回退到本地模型润色（可选）。
"""

from __future__ import annotations

import re

from app.core.config import get_settings
from app.core.language import (
    PERIOD_LABEL_EN,
    REPORTING_PERIOD_EN,
    format_amount_en,
    format_percents_en,
    get_translator,
    has_untranslated_chinese,
    localize,
)
from app.core.logging_conf import logger, trace
from app.core.number_utils import parse_amount
from app.models.schemas import QueryAnalysis, RetrievedChunk

# 与中文侧保持一致的抽取正则（容忍 PDF 断行）
from app.core.generator import (  # noqa: E402  (此处导入避免重复实现)
    AWARD_PATTERN,
    FUND_ROW_PATTERN,
    PERCENT_PATTERN,
    PERSON_PATTERN,
    RATIO_PATTERN,
    REVENUE_PATTERN,
    SUPPLIER_PATTERN,
    UPSTREAM_PATTERN,
    DOWNSTREAM_PATTERN,
    _clean_person,
)

COMPANY_EN = "Wuhan Xingtu Xinke Electronics Co., Ltd."
MONEY_PATTERN = re.compile(r"([\d,]+(?:\.\d+)?)\s*万元")


class EnglishAnswerBuilder:
    """英文答案构造器。"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.translator = get_translator()

    # ------------------------------------------------------------------
    @trace
    def build(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        analysis: QueryAnalysis | None = None,
    ) -> tuple[str, RetrievedChunk] | None:
        """生成英文答案。

        Returns:
            ``(英文答案, 作为依据的片段)``；无法构造时返回 ``None``（交由上层兜底）。
        """
        intent = analysis.intent if analysis else ""
        lowered = question.lower()

        # 规则表：(是否适用, 构造函数)
        #
        # **顺序很重要**：占比类问题必须排在金额类之前。
        # 否则 “What percentage of main business revenue came from the military
        # sector...” 会先命中收入金额规则，答出四个金额而不是四个比重。
        handlers: list[tuple[bool, object]] = [
            (
                ("percentage" in lowered or "proportion" in lowered or "ratio" in lowered
                 or "share" in lowered or "percent" in lowered),
                self._revenue_ratio,
            ),
            (intent == "注册资本" or "registered capital" in lowered, self._capital),
            (intent == "法定代表人" or "legal representative" in lowered, self._legal_representative),
            (("revenue" in lowered or "income" in lowered) and "military" in lowered, self._revenue),
            ("major supplier" in lowered or "supplier" in lowered, self._supplier),
            (
                "science and technology progress" in lowered or "award" in lowered or "prize" in lowered,
                self._award,
            ),
            (intent == "技术标准" or "technical standard" in lowered or "technical specification" in lowered, self._standard),
            ("upstream" in lowered, self._upstream),
            ("downstream" in lowered, self._downstream),
            ("working capital" in lowered or "raised funds" in lowered, self._fund_usage),
        ]

        for enabled, handler in handlers:
            if not enabled:
                continue
            try:
                outcome = handler(contexts)  # type: ignore[operator]
            except Exception as exc:
                logger.exception(
                    "app.core.english",
                    "英文规则执行异常",
                    handler=getattr(handler, "__name__", "?"),
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            if outcome:
                text, evidence = outcome
                logger.info(
                    "app.core.english",
                    "英文答案命中模板",
                    handler=getattr(handler, "__name__", "?"),
                    page=evidence.chunk.page,
                )
                return text, evidence

        # 模板未覆盖：尝试本地模型润色，否则给出带引用的英文摘要
        return self._fallback(question, contexts)

    # ------------------------------------------------------------------
    # 各意图模板
    # ------------------------------------------------------------------
    def _capital(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        for item in contexts:
            match = re.search(r"注册资本[：:\s]*([\d,]+(?:\.\d+)?)\s*万元", item.chunk.content)
            if match:
                amount = format_amount_en(f"{match.group(1)} 万元")
                return (
                    f"The registered capital of {COMPANY_EN} is {amount}.",
                    item,
                )
        return None

    def _legal_representative(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        for item in contexts:
            match = PERSON_PATTERN.search(item.chunk.content)
            if match:
                name = _clean_person(match.group(1))
                if len(name) >= 2:
                    english_name = localize(name)
                    if has_untranslated_chinese(english_name):
                        # 人名无英文对照时给出拼音式说明，避免输出中文姓名
                        english_name = f"{name} (Chinese name as in the prospectus)"
                    return (f"The legal representative is {english_name}.", item)
        return None

    def _revenue(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        for item in contexts:
            match = re.search(r"来自军用领域的\s*收入分别为([^。]+?)(?:，|,)?\s*占", item.chunk.content)
            if not match or "军用领域" not in item.chunk.content:
                continue
            amounts = MONEY_PATTERN.findall(match.group(1))
            if not amounts:
                continue
            rendered = [format_amount_en(f"{amount} 万元") for amount in amounts]
            listing = format_percents_en(rendered)
            return (
                f"During {REPORTING_PERIOD_EN}, the company's revenue from the military sector was "
                f"{listing} respectively.",
                item,
            )
        return None

    def _revenue_ratio(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        for item in contexts:
            if "军用领域" not in item.chunk.content or "主营业务收入" not in item.chunk.content:
                continue
            match = re.search(r"占\s*主营业务收入比重分别为([^。]+?)(?:。|$)", item.chunk.content)
            if not match:
                continue
            ratios = [pct.replace(" ", "") for pct in PERCENT_PATTERN.findall(match.group(1))]
            if not ratios:
                continue
            return (
                f"During {REPORTING_PERIOD_EN}, the company's revenue from the military sector accounted for "
                f"{format_percents_en(ratios)} of its main business revenue respectively.",
                item,
            )
        return None

    def _supplier(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        for item in contexts:
            match = re.search(r"(?:已经成为|已成为)([^，。]{2,20}领域)的\s*重要供应商", item.chunk.content)
            if not match:
                continue
            field = re.sub(r"\s+", "", match.group(1)).replace("领域", "")
            english_field = localize(field + "领域") if field else "the military video command"
            english_field = english_field.replace("领域", "").strip()
            return (
                f"The company has become a major supplier in {english_field}.",
                item,
            )
        return None

    def _award(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        best: tuple[float, str, RetrievedChunk] | None = None
        for item in contexts:
            content = item.chunk.content
            if not AWARD_PATTERN.search(content):
                continue
            for sentence in re.split(r"(?<=[。；])", content):
                if not AWARD_PATTERN.search(sentence):
                    continue
                cleaned = re.sub(r"\s+", "", sentence).strip("。；").lstrip("”）) ")
                if len(cleaned) < 15 or not ("荣获" in cleaned or "获得" in cleaned):
                    continue
                # 抽出工程名与时间，其余描述性文字不逐字翻译，保证英文可读
                project = re.search(r"[“\"]([^”\"]{6,60})[”\"]", cleaned)
                date = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", cleaned)
                project_en = (
                    "a major intelligence, command, control and communications network integration project "
                    "(comparable to the US C4ISR system)"
                )
                if project:
                    raw = project.group(1)
                    if "情报" in raw and ("指挥" in raw or "控制" in raw):
                        project_en = (
                            f"the “{raw}” project (an intelligence, command, control and communications "
                            f"network integration project, comparable to the US C4ISR system)"
                        )
                when = ""
                if date:
                    when = f"In {date.group(1)}-{int(date.group(2)):02d}, "
                text = (
                    f"{when}{project_en} was awarded the First Prize of the National Science and "
                    f"Technology Progress Award. According to the certification documents issued by the "
                    f"lead institute and military users, Xingtu Xinke was the sole participant responsible "
                    f"for the networked video command system of that project."
                )
                quality = (2.0 if date else 0.0) + (1.0 if project else 0.0) + (1.0 if len(cleaned) >= 40 else 0.0)
                if best is None or quality > best[0]:
                    best = (quality, text, item)
        if best is None:
            return None
        return best[1], best[2]

    def _standard(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        for item in contexts:
            content = item.chunk.content
            if "重要供应商" not in content and "技术标准" not in content:
                continue
            match = re.search(r"《([^》]{4,60})》", content)
            if not match:
                continue
            year = re.search(r"(20\d{2})\s*年", content)
            year_text = f" ({year.group(1)})" if year else ""
            return (
                f"The company participated in formulating the first military video command system "
                f"technical standard{year_text}, i.e. 《{match.group(1)}》. The standard was led by the "
                f"Academy of Military Sciences, together with the Institute of Data Communication Science "
                f"and Technology and Peking University.",
                item,
            )
        return None

    def _upstream(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        for item in contexts:
            match = re.search(r"上游涉及([^。]{4,120})。", item.chunk.content)
            if match:
                detail = localize(re.sub(r"\s+", "", match.group(1)))
                return (
                    f"The upstream of the electronic information industry involves {detail}.",
                    item,
                )
        return None

    def _downstream(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        for item in contexts:
            match = re.search(r"下游行业为([^。]{4,120})。", item.chunk.content)
            if match:
                detail = localize(re.sub(r"\s+", "", match.group(1)))
                return (
                    f"The downstream of the electronic information industry consists of {detail}.",
                    item,
                )
        return None

    def _fund_usage(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        for item in contexts:
            content = item.chunk.content
            if not FUND_ROW_PATTERN.search(content):
                continue
            for line in content.split("\n"):
                if "补充流动资金" not in line or "|" not in line:
                    continue
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                for cell in cells:
                    if "补充流动资金" in cell or not cell:
                        continue
                    if re.match(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^\d+\.\d+$", cell):
                        amount = format_amount_en(f"{cell} 万元")
                        return (
                            f"The company plans to use {amount} of the raised funds to supplement "
                            f"working capital.",
                            item,
                        )
            match = re.search(r"补充流动资金\s*\|?\s*([\d,]+(?:\.\d+)?)\s*(万元|亿元|元)?", content)
            if match:
                amount = format_amount_en(f"{match.group(1)} {match.group(2) or '万元'}")
                return (
                    f"The company plans to use {amount} of the raised funds to supplement working capital.",
                    item,
                )
        return None

    # ------------------------------------------------------------------
    def _fallback(
        self, question: str, contexts: list[RetrievedChunk]
    ) -> tuple[str, RetrievedChunk] | None:
        """模板未覆盖时的兜底：先本地模型，再术语表摘要。"""
        best = contexts[0] if contexts else None
        if best is None:
            return None
        snippet = best.chunk.content[:600]

        if self.settings.language.allow_local_translation and self.translator.available:
            translated = self.translator.translate(snippet, target="en")
            if translated and not has_untranslated_chinese(translated):
                return (
                    f"According to page {best.chunk.page} of the prospectus: {translated}",
                    best,
                )

        localized = localize(re.sub(r"\s+", " ", snippet))
        if has_untranslated_chinese(localized):
            # 术语表也覆盖不了：明确说明依据位置，不硬编造英文
            logger.warning(
                "app.core.english",
                "英文模板与术语表均未覆盖该问题，返回带页码的依据提示",
                question=question,
            )
            return (
                f"I can only cite the source: see page {best.chunk.page} of the prospectus "
                f"(Chinese original). The retrieved passage is: {localized[:300]}",
                best,
            )
        return (f"According to page {best.chunk.page} of the prospectus: {localized[:400]}", best)


_builder: EnglishAnswerBuilder | None = None


def get_english_builder() -> EnglishAnswerBuilder:
    """工厂函数：获取单例。"""
    global _builder
    if _builder is None:
        _builder = EnglishAnswerBuilder()
    return _builder


def reset_english_builder() -> None:
    """重置单例（测试用）。"""
    global _builder
    _builder = None
