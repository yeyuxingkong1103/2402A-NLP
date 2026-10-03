"""中英文双语支持：语言识别、问答语言决策、意图→中文问句桥接、英文抽取式模板。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 多语言层（工单 6.6）

设计要点（环境事实 2.2：本机断网，无翻译模型可下载）：
1. **语言识别**用字符统计（CJK 占比）判定，毫秒级、确定性；
2. 英文提问检索中文语料时按**意图生成完整中文问句**（``bridge_query_to_chinese``），
   而不是把英文句子里的几个词原地替换——**原地替换会产出中英混排串**，实测会被
   ``Retriever.is_answerable`` 判为"问题实义词未出现在片段中"→ 拒答（t11 回归缺陷）；
   本方案与工单1 基线 ``build_chinese_query``（中文问句模板 + 术语表 + 专有名词）同思路；
3. 回答语言跟随提问语言（``resolve_answer_language``）；
4. 无 LLM 时的英文兜底回答由**确定性模板**生成（不伪造英文能力，标注 mode=extractive）。
"""

from __future__ import annotations

import re
from typing import Literal

from app.core.config import get_settings
from app.core.logging_conf import logger, truncate

CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
#: 拉丁实义词（用于判断桥接结果是否仍夹杂英文）
LATIN_WORD_PATTERN = re.compile(r"[A-Za-z]{2,}")
#: 发行人中文全称（跨语言检索的主体名先验；与语料/中文工单问题一致）
COMPANY_NAME_ZH = "武汉兴图新科电子股份有限公司"

#: 英文术语 -> 招股书中文用词（工单 6.2/6.3 点名词全覆盖）
TERM_GLOSSARY: dict[str, str] = {
    "registered capital": "注册资本",
    "legal representative": "法定代表人",
    "upstream and downstream": "上下游",
    "upstream": "上游",
    "downstream": "下游",
    "suppliers": "供应商",
    "supplier": "供应商",
    "customers": "客户",
    "customer": "客户",
    "operating revenue": "主营业务收入",
    "business revenue": "主营业务收入",
    "revenue": "收入",
    "proportion": "占比",
    "percentage": "占比",
    "market share": "市场占有率",
    "ratio": "比重",
    "share": "占比",
    "raised funds": "募集资金",
    "fundraising": "募集资金",
    "use of proceeds": "募集资金运用",
    "working capital": "补充流动资金",
    "technological standard": "技术标准",
    "technical standard": "技术标准",
    "national science and technology progress award": "国家科技进步一等奖",
    "science and technology progress award": "科技进步奖",
    "military industry": "军工",
    "military": "军用领域",
    "reporting period": "报告期内",
    "prospectus": "招股说明书",
    "wuhan xingtu xinke": "武汉兴图新科电子股份有限公司",
    "xingtu xinke": "武汉兴图新科电子股份有限公司",
    "video command": "视频指挥",
    "command and control": "指挥控制",
    "c4isr": "指挥控制",
    "industry chain": "产业链",
    "gross margin": "毛利率",
    "net profit": "净利润",
    "total assets": "总资产",
    "research and development": "研发",
    "main business": "主营业务",
}

#: 中文同义词扩展表（查询扩展用；与术语表方向相反）
ZH_SYNONYMS: dict[str, list[str]] = {
    "占比": ["比重", "比例", "构成"],
    "比重": ["占比", "比例"],
    "募集资金": ["募资", "募集资金运用", "募投项目"],
    "募资": ["募集资金"],
    "上游": ["上游行业", "上游企业"],
    "下游": ["下游行业", "下游企业"],
    "供应商": ["供货商", "采购"],
    "客户": ["顾客", "销售对象"],
    "收入": ["主营业务收入", "营业收入"],
    "科技进步奖": ["国家科技进步一等奖", "科技进步一等奖"],
    "技术标准": ["标准", "行业标准", "国家标准"],
    "法定代表人": ["法人代表", "董事长"],
    "注册资本": ["股本", "实收资本"],
    "军用领域": ["军工", "国防"],
    "补充流动资金": ["流动资金", "募资用途"],
}


def detect_language(text: str) -> Literal["zh", "en"]:
    """判断文本语言：CJK 字符占比 >= 0.15 视为中文，否则英文。

    Args:
        text: 待判定文本（英文提问可能夹带少量中文专名）。

    Returns:
        ``"zh"`` 或 ``"en"``；空文本默认 ``"zh"``（本系统主语言）。
    """
    try:
        raw = (text or "").strip()
        if not raw:
            return "zh"
        cjk = len(CJK_PATTERN.findall(raw))
        latin = len(LATIN_PATTERN.findall(raw))
        if cjk == 0:
            return "en" if latin else "zh"
        return "zh" if (cjk / max(1, cjk + latin)) >= 0.15 or cjk >= latin else "en"
    except Exception:
        logger.exception("app.core.language", "语言识别失败，按中文处理", text=truncate(text, 80))
        return "zh"


def resolve_answer_language(question: str, configured: str = "auto") -> Literal["zh", "en"]:
    """决定回答语言：``auto`` 跟随提问语言，否则用配置强制值。"""
    try:
        if configured in {"zh", "en"}:
            return configured  # type: ignore[return-value]
        return detect_language(question)
    except Exception:
        logger.exception("app.core.language", "回答语言决策失败，按中文处理")
        return "zh"


#: 英文意图规则（英文关键词 -> 中文意图名）；顺序 = 优先级，专指优先于泛指。
#: 与工单1 基线 ``language.detect_english_intent`` 同思路（按意图挑中文问句模板）。
EN_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("法定代表人", ("legal representative", "legal rep", "lawful representative", "who is the legal")),
    ("注册资本", ("registered capital", "registered capitals", "share capital", "paid-in capital")),
    ("技术标准", ("technical standard", "technological standard", "technical standards", "formulating", "formulated")),
    ("募资用途", ("raised funds", "fundraising", "use of proceeds", "working capital", "supplement")),
    ("荣誉", ("science and technology progress award", "award", "prize", "honor", "honour")),
    ("上下游", ("upstream", "downstream", "industry chain")),
    ("供应商", ("supplier", "suppliers")),
    ("客户", ("customer", "customers", "client")),
    ("占比", ("percentage", "proportion", "ratio", "percent", "share of", "share")),
    ("收入", ("revenue", "income", "sales", "turnover")),
    ("注册资本", ("capital",)),
)

#: 意图 -> 完整中文检索问句模板（**纯中文**，无英文残留）。
#: ``{sector}`` 在英文问题含 military 时填「军用领域」，否则为空。
CHINESE_QUERY_TEMPLATES: dict[str, str] = {
    "法定代表人": "公司法定代表人是？",
    "注册资本": "公司注册资本是多少？",
    "募资用途": "募集资金中用于补充流动资金是多少？",
    "技术标准": "公司参与制定了哪个视频指挥系统技术标准？",
    "荣誉": "参与的哪个工程荣获国家科技进步一等奖？",
    "上下游": "电子信息行业的上游和下游分别包括哪些企业？",
    "供应商": "公司的主要供应商有哪些？",
    "客户": "公司的主要客户有哪些？",
    "占比": "{sector}收入占主营业务收入的比重分别是多少？",
    "收入": "{sector}收入分别是多少？",
}

#: 英文线索 -> 中文关键词（模板不可用时的兜底检索串；与基线 ``english_keywords`` 同思路）
QUERY_HINTS_EN_ZH: tuple[tuple[str, str], ...] = (
    ("registered capital", "注册资本"),
    ("legal representative", "法定代表人"),
    ("raised funds", "募集资金"),
    ("working capital", "补充流动资金"),
    ("technical standard", "技术标准"),
    ("science and technology progress award", "科技进步奖"),
    ("military sector", "军用领域"),
    ("military", "军用领域"),
    ("main business revenue", "主营业务收入"),
    ("revenue", "收入"),
    ("percentage", "占比"),
    ("proportion", "占比"),
    ("upstream", "上游"),
    ("downstream", "下游"),
    ("supplier", "供应商"),
    ("customer", "客户"),
)


def detect_english_intent(question: str) -> str:
    """从英文问题里判定意图，返回中文意图名；无法判定时返回空串（对齐基线 API）。"""
    try:
        lowered = (question or "").lower()
        for intent, keywords in EN_INTENT_RULES:
            if any(keyword in lowered for keyword in keywords):
                return intent
        return ""
    except Exception:
        logger.exception("app.core.language", "英文意图识别失败")
        return ""


def english_keywords(question: str) -> list[str]:
    """从英文问题里提取中文关键词线索（模板不可用时的兜底检索串）。"""
    try:
        lowered = (question or "").lower()
        hits: list[str] = []
        for english, chinese in QUERY_HINTS_EN_ZH:
            if english in lowered and chinese not in hits:
                hits.append(chinese)
        return hits
    except Exception:
        logger.exception("app.core.language", "英文关键词提取失败")
        return []


def is_pure_chinese(text: str) -> bool:
    """判断桥接结果是否为**纯中文**（不含拉丁实义词）。

    为什么要判：t11 回归缺陷的根因就是"中英混排串"进入可答性校验后被拒答，
    因此桥接函数必须保证产出干净的中文查询（或明确回退 ``raw``）。
    """
    return bool(CJK_PATTERN.search(text or "")) and not LATIN_WORD_PATTERN.search(text or "")


def issuer_anchored_query(query: str) -> str:
    """给检索串加上**发行人中文全称**前缀（跨语言场景的领域先验）。

    为什么需要：招股书是单一发行人的文档，关键字段（注册资本/法定代表人）在子公司表、
    释义表里都有同名行；不带主体名的问句会命中别的实体——实测英文 Q543 因桥接问句缺主体名
    而召回 p64/p63/p68 的子公司注册资本行。中文工单问题本身都带公司全称，
    这里为该先验补一个**独立检索变体**（不替换主查询，避免把语义方向拉偏）。
    """
    text = (query or "").strip()
    if not text or COMPANY_NAME_ZH in text:
        return text
    return f"{COMPANY_NAME_ZH} {text}"


def build_chinese_query(english_question: str, intent: str = "") -> tuple[str, str]:
    """把英文问题转成**纯中文**检索问句（意图 → 中文问句模板 → 关键词兜底）。

    与工单1 基线 ``build_chinese_query`` 同思路：**中文问句模板 + 术语表 + 专有名词**，
    而不是在英文原句里就地替换几个词（那样会产出中英混排串）。

    Args:
        english_question: 英文原问题。
        intent: Query 理解得到的中文意图（可为空，函数会自行用英文规则补判）。

    Returns:
        ``(中文检索串, 方式)``；方式为 ``"glossary"``（模板/关键词命中）或 ``"raw"``（无中文线索，保留原句）。
    """
    try:
        raw = (english_question or "").strip()
        if not raw:
            return raw, "raw"
        lowered = raw.lower()
        resolved = intent if CHINESE_QUERY_TEMPLATES.get(intent) else detect_english_intent(raw)
        template = CHINESE_QUERY_TEMPLATES.get(resolved, "")
        if template:
            sector = "军用领域" if ("military" in lowered or "army" in lowered or "national defense" in lowered) else ""
            query = template.replace("{sector}", sector).strip()
            logger.info(
                "app.core.language",
                "英文提问已按意图生成纯中文检索问句",
                english=truncate(raw, 140),
                intent=resolved,
                chinese=query,
            )
            return query, "glossary"
        hints = english_keywords(raw)
        if hints:
            query = " ".join(hints)
            logger.info(
                "app.core.language",
                "无匹配问句模板，退回中文关键词检索",
                english=truncate(raw, 140),
                chinese=query,
            )
            return query, "glossary"
        logger.warning(
            "app.core.language",
            "术语表未命中任何中文线索，保留英文原问题检索（跨语言召回有限）",
            english=truncate(raw, 140),
        )
        return raw, "raw"
    except Exception:
        logger.exception("app.core.language", "意图→中文问句桥接失败，回退原问题")
        return english_question, "raw"


def bridge_query_to_chinese(question: str, intent: str = "") -> tuple[str, str]:
    """把英文提问桥接为**可用于检索的纯中文查询串**（工单 6.6）。

    实现顺序（**不再使用原地术语替换**，那是 t11 回归缺陷的根因）：
    1. 意图 → 完整中文问句模板（``build_chinese_query``）；
    2. 无模板命中 → 中文关键词串；
    3. 仍无线索但逐词术语表替换结果**为纯中文**（例如整句都是术语）→ 用替换结果；
    4. 其余 → 原样返回（``raw``），绝不返回中英混排串。

    Returns:
        ``(检索串, 桥接方式)``；方式为 ``"glossary"`` 或 ``"raw"``。
    """
    try:
        raw = (question or "").strip()
        if not raw:
            return raw, "raw"
        settings = get_settings()
        if settings.language.query_bridge == "raw":
            return raw, "raw"
        query, mode = build_chinese_query(raw, intent=intent)
        if mode == "glossary":
            return query, mode
        # 兜底：逐词替换，但**只有产出纯中文**才采用（避免混排串进可答性校验）
        bridged = raw.lower()
        for term in sorted(TERM_GLOSSARY, key=len, reverse=True):
            if term in bridged:
                bridged = bridged.replace(term, f" {TERM_GLOSSARY[term]} ")
        bridged = re.sub(r"\s+", " ", bridged).strip()
        if bridged != raw.lower() and is_pure_chinese(bridged):
            logger.info(
                "app.core.language",
                "术语表逐词替换产出纯中文检索串",
                english=truncate(raw, 140),
                chinese=bridged,
            )
            return bridged, "glossary"
        logger.warning(
            "app.core.language",
            "无法产出纯中文检索串，保留英文原问题（避免中英混排进入可答性校验）",
            english=truncate(raw, 140),
        )
        return raw, "raw"
    except Exception:
        logger.exception("app.core.language", "语言桥接失败，回退原问题")
        return question, "raw"


def expand_with_synonyms(text: str, limit: int = 6) -> list[str]:
    """依据领域同义词表扩展查询词（返回命中的同义词，去重、限制条数）。"""
    try:
        found: list[str] = []
        for key, synonyms in ZH_SYNONYMS.items():
            if key in (text or ""):
                for item in synonyms:
                    if item not in found:
                        found.append(item)
        return found[:limit]
    except Exception:
        logger.exception("app.core.language", "同义词扩展失败")
        return []


def looks_english(text: str) -> bool:
    """兼容辅助：是否判定为英文文本。"""
    return detect_language(text) == "en"


def build_english_extractive_answer(evidence: str, page: int) -> str:
    """无 LLM 时的英文抽取式兜底回答（确定性模板 + 页码引用）。

    诚实性：该答案由抽取式模板生成，``mode="extractive"``，不得描述为"模型英文作答"。
    """
    try:
        snippet = (evidence or "").strip().replace("\n", " ")
        if len(snippet) > 260:
            snippet = snippet[:260].rstrip() + "…"
        return f"According to the prospectus: {snippet} [Page: {page}]"
    except Exception:
        logger.exception("app.core.language", "英文兜底模板生成失败")
        return f"Sorry, the answer is not available in the document. [Page: {page}]"
