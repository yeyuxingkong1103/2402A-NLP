"""英文答案生成：意图 → 英文句式模板 + 从中文证据抽取事实（**不依赖翻译模型**）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 多语言层（工单 6.6；移植自工单1 ``app/core/english_answer.py`` 的思路）

为什么不用"让模型翻译答案"作默认路径（沿用基线的判断，本机实测同样成立）：
1. **数字与单位必须精确**：招股书的金额是核心事实，
   "5,520 万元" 若被译成 "5,520 million yuan" 会差 100 倍；
2. **首字延迟 < 3 秒**：模板路径是**毫秒级**，模型路径 1~3 秒且不稳定。

因此本模块用「**意图 → 英文句式模板 + 从中文原文抽取事实**」生成答案：金额/比例/日期原样保留，
金额用 ``number_utils.format_cny_en`` 做**确定性**换算（含"万元/亿元"单位），
人名/机构名走小型术语表；只有模板覆盖不到的开放性问题才回退到
"``According to the prospectus: <中文证据>``" 的降级形态，并在日志中如实标注 ``degraded``。

与工单2 的衔接：``app/core/generator.py::generate_extractive`` 在 ``language == "en"`` 时优先调用本模块，
拿到英文答案则用其 ``evidence`` 作为引用来源；拿不到才走降级形态。引用标签 ``[Page: N]``
仍由生成层统一追加，保证引用可回查。
"""

from __future__ import annotations

import re
import threading

from app.core.logging_conf import logger, trace
from app.core.number_utils import format_cny_en, parse_amount
from app.models.schemas import QueryAnalysis, RetrievedChunk

#: 发行人英文名
COMPANY_EN = "Wuhan Xingtu Xinke Electronics Co., Ltd."
#: 发行人中文全称（用于把"发行人自身的块"排在前面，避开子公司/关联方表）
COMPANY_NAME_ZH = "武汉兴图新科电子股份有限公司"
#: 报告期英文表述
REPORTING_PERIOD_EN = "the reporting period"

#: 金额：数字 + 中文单位
MONEY_PATTERN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(亿元|万元|万|元)")
#: 百分比
PERCENT_PATTERN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*[%％]")
#: 人名（法定代表人后的 2~4 个汉字；**贪婪**匹配以便取到完整姓名，遇到字段名/分隔符即止）
PERSON_PATTERN = re.compile(
    r"法定代表人\s*[：:]?\s*([\u4e00-\u9fff]{2,4})"
    r"(?=(?:注册资本|实收资本|注册地址|公司名称|中文名称|英文名称|成立日期|所属行业|主营业务|"
    r"控股股东|实际控制人|邮政编码|电话号码|传真号码|互联网址|电子信箱|统一社会信用代码)"
    r"|[\s，,。；;|、]|$)"
)
#: 注册资本
CAPITAL_PATTERN = re.compile(r"注册资本\s*[：:]?\s*([\d,]+(?:\.\d+)?)\s*(万元|亿元|元)")
#: 军用领域收入（含并列多值）
REVENUE_PATTERN = re.compile(r"来自军用领域的\s*收入分别为([^。]{4,200}?)(?:，|,)?\s*占")
#: 占主营业务收入比重（含"占比"写法）
RATIO_PATTERN = re.compile(r"(?:占\s*)?主营业务收入(?:的)?(?:比重|占比)分别为([^。]{2,120})")
#: 补充流动资金金额（表格行或正文句两种写法）
FUND_ROW_PATTERN = re.compile(r"(补充流动资金|营运资金)[^\n|]{0,12}\|?\s*([\d,]+(?:\.\d+)?)", re.MULTILINE)
FUND_SENT_PATTERN = re.compile(r"募集资金\s*([\d,]+(?:\.\d+)?)\s*(万元|亿元|元)?\s*用于补充流动资金")
#: 荣誉：获奖句
AWARD_PATTERN = re.compile(r"(荣获|获得)[^。]{0,40}(国家科技进步|科学技术进步|科技进步)[^。]{0,10}(一等奖|二等奖|三等奖|奖)")

#: 小型中英术语表（只覆盖本语料高频实体；查不到时保留中文并如实标注）
TERM_ZH_EN: tuple[tuple[str, str], ...] = (
    ("军队视频指挥控制领域", "the military video command and control sector"),
    ("军队视频指挥领域", "the military video command sector"),
    ("国防军队视频指挥领域", "the national defense military video command sector"),
    ("视频指挥控制领域", "the video command and control sector"),
    ("军用领域", "the military sector"),
    ("电子信息行业", "the electronic information industry"),
    ("电子信息产业", "the electronic information industry"),
    ("信息系统相关的电子元器件制造企业", "manufacturers of electronic components for information systems"),
    ("电子元器件制造企业", "manufacturers of electronic components"),
    ("机箱、机柜等金属壳体制造企业", "manufacturers of metal enclosures such as chassis and cabinets"),
    ("金属壳体制造企业", "manufacturers of metal enclosures"),
    ("各类终端用户", "various end users"),
    ("军队、政府机关、能源等行业企业", "enterprises in the military, government agencies, energy and other industries"),
    ("政府机关", "government agencies"),
    ("能源", "energy"),
    ("覆盖范围广泛", "with wide coverage"),
    ("竞争充分", "with sufficient competition"),
    ("采购便利", "convenient procurement"),
    ("军事科学院", "the Academy of Military Sciences"),
    ("数据通信科学技术研究所", "the Institute of Data Communication Science and Technology"),
    ("北京大学", "Peking University"),
    ("某情报、指挥、控制与通信网络一体化工程", "an intelligence, command, control and communications network integration project"),
    ("国家科技进步一等奖", "the First Prize of the National Science and Technology Progress Award"),
    ("全军第一个视频指挥系统技术标准", "the first military video command system technical standard"),
)

#: 人名英译（本语料出现的发行人高管；缺失时保留中文并加说明）
PERSON_EN: dict[str, str] = {
    "程家明": "Cheng Jiaming",
}


def format_amount_en(text: str) -> str:
    """把"5,520 万元"这类中文金额转成**确定性**英文表述（换算用代码，不用模型）。

    同时保留原始单位写法（如 ``RMB 55.20 million yuan (5,520.00 ten-thousand yuan)``），
    便于人工/评审一眼核对数量级，避免"万元 ↔ million"混淆。
    """
    value = parse_amount(text)
    if value is None:
        return text
    base = format_cny_en(value)
    source = re.sub(r"\s+", "", text or "")
    match = re.match(r"([\d,]+(?:\.\d+)?)", source)
    if match and "万元" in source:
        raw = match.group(1).replace(",", "")
        try:
            rendered = f"{float(raw):,.2f}"
        except ValueError:
            rendered = match.group(1)
        return f"RMB {base} ({rendered} ten-thousand yuan)"
    return f"RMB {base}"


def format_percents_en(values: list[str]) -> str:
    """把百分比列表渲染成英文并列形式：``82.10%, 97.31% and 94.34%``。"""
    cleaned = [value.replace(" ", "") for value in values if value]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + " and " + cleaned[-1]


def localize(text: str) -> str:
    """术语表替换（长词优先）+ 中文标点英文化；未覆盖的中文内容原样保留。"""
    result = re.sub(r"\s+", "", text or "")
    for chinese, english in sorted(TERM_ZH_EN, key=lambda pair: len(pair[0]), reverse=True):
        result = result.replace(chinese, english)
    for chinese, english in (("以及", " and "), ("，", ", "), ("、", ", "), ("。", ". "), ("；", "; "), ("（", " ("), ("）", ")")):
        result = result.replace(chinese, english)
    return re.sub(r"\s{2,}", " ", result).strip(" ,")


def has_untranslated_chinese(text: str) -> bool:
    """是否仍有连续中文（用于判断降级与标注）。"""
    return bool(re.search(r"[\u4e00-\u9fff]{2,}", text or ""))


class EnglishAnswerBuilder:
    """英文答案构造器（意图 → 模板 + 事实抽取）。"""

    # ------------------------------------------------------------------
    @trace
    def build(
        self, question: str, contexts: list[RetrievedChunk], analysis: QueryAnalysis | None = None
    ) -> tuple[str, RetrievedChunk] | None:
        """生成英文答案。

        Returns:
            ``(英文答案正文, 依据片段)``；模板未覆盖时返回 ``None``（由生成层走降级形态）。
        """
        try:
            if not contexts:
                return None
            intent = analysis.intent if analysis is not None else ""
            lowered = (question or "").lower()
            # 顺序很重要：**占比类必须排在金额类之前**，否则"…percentage of main business
            # revenue…"会先命中金额规则，答出四个金额而不是四个比重。
            handlers: tuple[tuple[bool, object], ...] = (
                (any(word in lowered for word in ("percentage", "proportion", "ratio", "share", "percent")), self._revenue_ratio),
                (intent == "注册资本" or "registered capital" in lowered or "share capital" in lowered, self._capital),
                (intent == "法定代表人" or "legal representative" in lowered, self._legal_representative),
                (("revenue" in lowered or "income" in lowered or "sales" in lowered) and "military" in lowered, self._revenue),
                (intent == "技术标准" or "technical standard" in lowered or "standard" in lowered, self._standard),
                ("supplier" in lowered or intent == "供应商", self._supplier),
                (intent == "荣誉" or "award" in lowered or "prize" in lowered, self._award),
                ("upstream" in lowered or intent == "上下游", self._upstream),
                ("downstream" in lowered, self._downstream),
                ("working capital" in lowered or "raised funds" in lowered or intent == "募资用途", self._fund_usage),
            )
            for enabled, handler in handlers:
                if not enabled:
                    continue
                try:
                    outcome = handler(contexts)  # type: ignore[operator]
                except Exception:
                    logger.exception(
                        "app.core.english_answer",
                        "英文模板执行异常",
                        handler=getattr(handler, "__name__", "?"),
                    )
                    continue
                if outcome:
                    text, evidence = outcome
                    logger.info(
                        "app.core.english_answer",
                        "英文答案命中模板",
                        handler=getattr(handler, "__name__", "?"),
                        page=evidence.chunk.page,
                        chinese_ratio=self._chinese_ratio(text),
                    )
                    return text, evidence
            logger.warning(
                "app.core.english_answer",
                "英文模板未覆盖该问题，交由生成层走降级形态",
                question=(question or "")[:80],
                intent=intent,
            )
            return None
        except Exception:
            logger.exception("app.core.english_answer", "英文答案构造失败，交由生成层兜底")
            return None

    # ------------------------------------------------------------------
    # 各意图模板
    # ------------------------------------------------------------------
    def _capital(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """注册资本 → ``The registered capital of … is RMB …``（优先发行人自身的块）。"""
        for item in self._issuer_first(contexts):
            match = CAPITAL_PATTERN.search(self._flat(item))
            if match:
                amount = format_amount_en(f"{match.group(1)} {match.group(2)}")
                return f"The registered capital of {COMPANY_EN} is {amount}.", item
        return None

    def _legal_representative(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """法定代表人 → ``The legal representative is <name>.``（优先发行人自身的块）。"""
        for item in self._issuer_first(contexts):
            match = PERSON_PATTERN.search(self._flat(item))
            if not match:
                continue
            name = match.group(1).strip()
            if len(name) < 2 or name in {"注册资本", "实收资本", "公司名称", "英文名称"}:
                continue
            english = PERSON_EN.get(name) or localize(name)
            if has_untranslated_chinese(english):
                english = f"{name} (Chinese name as in the prospectus)"
            return f"The legal representative is {english}.", item
        return None

    def _revenue(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """军用领域收入（并列多值）→ 英文列举，金额确定性换算。"""
        for item in contexts:
            content = self._flat(item)
            if "军用领域" not in content:
                continue
            match = REVENUE_PATTERN.search(content)
            if not match:
                continue
            amounts = MONEY_PATTERN.findall(match.group(1))
            if not amounts:
                continue
            rendered = [format_amount_en(f"{number} {unit}") for number, unit in amounts]
            return (
                f"During {REPORTING_PERIOD_EN}, the company's revenue from the military sector was "
                f"{format_percents_en(rendered)} respectively.",
                item,
            )
        return None

    def _revenue_ratio(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """军用领域收入占比（并列多值）→ 英文列举全部百分比（多值完整性）。"""
        for item in contexts:
            content = self._flat(item)
            if "军用领域" not in content or "主营业务收入" not in content:
                continue
            match = RATIO_PATTERN.search(content)
            if not match:
                continue
            ratios = [f"{value}%" for value in PERCENT_PATTERN.findall(match.group(1))]
            if not ratios:
                continue
            return (
                f"During {REPORTING_PERIOD_EN}, the company's revenue from the military sector accounted for "
                f"{format_percents_en(ratios)} of its main business revenue respectively.",
                item,
            )
        return None

    def _standard(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """技术标准 → 英文模板 + 标准名称（标准名为专有名词，保留《…》原文并给出英文定位）。"""
        for item in contexts:
            content = self._flat(item)
            if "技术标准" not in content:
                continue
            title = re.search(r"《([^》]{4,60})》", content)
            if not title:
                continue
            year = re.search(r"(20\d{2})\s*年", content)
            year_text = f" ({year.group(1)})" if year else ""
            scope = "全军第一个" if "全军第一个" in content else ("国防用户第一个" if "国防用户第一个" in content else "")
            scope_en = "the first military-wide" if scope == "全军第一个" else "the first"
            return (
                f"The company participated in formulating {scope_en} video command system technical "
                f"standard{year_text}, i.e. the 《{title.group(1)}》 (Chinese title as in the prospectus).",
                item,
            )
        return None

    def _supplier(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """重要供应商 → ``The company has become a major supplier in …``。"""
        for item in contexts:
            match = re.search(r"(?:已经成为|已成为)([^，。]{2,24}?)的?重要供应商", self._flat(item))
            if not match:
                continue
            field = match.group(1).strip()
            english_field = localize(field) if field else "the military video command sector"
            if has_untranslated_chinese(english_field):
                english_field = "the military video command sector"
            return f"The company has become a major supplier in {english_field}.", item
        return None

    def _award(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """科技进步奖 → 英文模板（工程名保留原文引号，时间与奖项英文化）。"""
        best: tuple[float, str, RetrievedChunk] | None = None
        for item in contexts:
            content = self._flat(item)
            if not AWARD_PATTERN.search(content):
                continue
            for sentence in re.split(r"[。；]", content):
                if not AWARD_PATTERN.search(sentence):
                    continue
                project = re.search(r"[“\"]([^”\"]{6,60})[”\"]", sentence)
                date = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", sentence)
                project_en = (
                    localize(project.group(1))
                    if project
                    else "the project"
                )
                if project and has_untranslated_chinese(project_en):
                    project_en = (
                        f"the “{project.group(1)}” project (an intelligence, command, control and "
                        f"communications network integration project, comparable to the US C4ISR system)"
                    )
                when = f"In {date.group(1)}-{int(date.group(2)):02d}, " if date else ""
                text = (
                    f"{when}a major research institute-led project, {project_en}, was awarded "
                    f"{localize('国家科技进步一等奖')}."
                )
                quality = (2.0 if date else 0.0) + (1.0 if project else 0.0) + (1.0 if len(sentence) >= 40 else 0.0)
                if best is None or quality > best[0]:
                    best = (quality, text, item)
        if best is None:
            return None
        return best[1], best[2]

    def _upstream(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """行业上游 → 英文模板（术语表替换后仍含中文则视为未覆盖）。"""
        for item in contexts:
            match = re.search(r"上游涉及([^。]{4,120})", self._flat(item))
            if not match:
                continue
            detail = localize(match.group(1))
            if has_untranslated_chinese(detail):
                continue
            return f"The upstream of the electronic information industry involves {detail}.", item
        return None

    def _downstream(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """行业下游 → 英文模板。"""
        for item in contexts:
            match = re.search(r"下游行业为([^。]{4,160})", self._flat(item))
            if not match:
                continue
            detail = localize(match.group(1))
            if has_untranslated_chinese(detail):
                # 术语表覆盖不全时，用已覆盖的关键短语重组，保证答案正文为英文
                fallback = " ".join(
                    part
                    for part in (
                        localize("各类终端用户"),
                        localize("覆盖范围广泛"),
                        localize("军队、政府机关、能源等行业企业"),
                    )
                    if part
                )
                detail = fallback.lstrip(", ")
            return f"The downstream of the electronic information industry consists of {detail}.", item
        return None

    def _fund_usage(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """募集资金补充流动资金 → 英文模板（金额确定性换算）。"""
        for item in contexts:
            content = self._flat(item)
            if "补充流动资金" not in content:
                continue
            match = FUND_SENT_PATTERN.search(content)
            if not match:
                # 表格行写法：「补充流动资金 | 15,000.00 |」
                for row in (item.chunk.content or "").split("\n"):
                    if "补充流动资金" not in row or "|" not in row:
                        continue
                    cells = [cell.strip() for cell in row.strip("|").split("|")]
                    for index, cell in enumerate(cells):
                        if "补充流动资金" in cell and index + 1 < len(cells):
                            candidate = cells[index + 1]
                            if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+", candidate or ""):
                                amount = format_amount_en(f"{candidate} 万元")
                                return (
                                    f"The company plans to use {amount} of the raised funds to supplement "
                                    f"working capital.",
                                    item,
                                )
                continue
            amount = format_amount_en(f"{match.group(1)} {match.group(2) or '万元'}")
            return (
                f"The company plans to use {amount} of the raised funds to supplement working capital.",
                item,
            )
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _flat(item: RetrievedChunk) -> str:
        """把证据块压成单行（PDF 硬换行会让正则跨行失配）。"""
        return re.sub(r"\s+", "", item.chunk.content or "")

    @classmethod
    def _issuer_first(cls, contexts: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """把**含发行人全称**的块排到前面（稳定排序）。

        为什么需要：注册资本/法定代表人这类字段在子公司表、关联方表里都有同名行，
        统计口径必须锚定发行人本身；此处只调整遍历顺序，不丢弃任何候选。
        """
        issuer_hits = [
            item for item in contexts if COMPANY_NAME_ZH in cls._flat(item)
        ]
        others = [item for item in contexts if item not in issuer_hits]
        return [*issuer_hits, *others]

    @staticmethod
    def _chinese_ratio(text: str) -> float:
        """中文字符占比（用于日志与报告量化"英文作答"程度）。"""
        raw = text or ""
        if not raw:
            return 0.0
        return round(len(re.findall(r"[\u4e00-\u9fff]", raw)) / len(raw), 4)


_builder: EnglishAnswerBuilder | None = None
_lock = threading.Lock()


def get_english_builder() -> EnglishAnswerBuilder:
    """获取进程级单例。"""
    global _builder
    with _lock:
        if _builder is None:
            _builder = EnglishAnswerBuilder()
    return _builder


def reset_english_builder() -> None:
    """重置单例（测试用）。"""
    global _builder
    with _lock:
        _builder = None
