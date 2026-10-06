"""多语言支持：语言检测、中英互译、英文答案组织。

工单追加要求：**支持中文与英文问答**。语料是中文招股书，所以英文链路的
核心问题是“英文问题如何命中中文证据、中文证据如何变成英文答案”。

方案（三层，逐层降级，保证任何环境都能用）::

    英文问题 ──① 本地小模型译成中文（Qwen3-0.6B，CPU 1~3 秒）
              └─② 领域术语表 + 形态还原（模型不可用时兜底）
                  └─ 中文混合检索（向量 + BM25）

    中文证据 ──① 本地小模型润色成英文
              └─② 领域术语表 + 模板化英文答案（默认，确定性且快）
                  └─ 数字/比例/金额原样保留（由 number_utils 换算单位）

为什么英文答案默认走“术语表 + 模板”而不是让模型自由翻译：
1. 财务问答里**数字和单位绝不能错**——实测 0.6B 模型会把“5,520 万元”
   译成 “5,520 million yuan”（差 100 倍）；模板由代码控制单位换算，不会出错；
2. 首字延迟要求 < 3 秒，模板路径是毫秒级，模型路径要 1~3 秒。
"""

from __future__ import annotations

import re
import time
from functools import lru_cache

from app.core.config import PROJECT_ROOT, get_settings
from app.core.logging_conf import logger, trace
from app.core.number_utils import parse_amount, to_million_yuan, to_ten_thousand_yuan

# --------------------------------------------------------------------------
# 语言检测
# --------------------------------------------------------------------------
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")

# 常见英文疑问词：用于在“纯 ASCII 但其实是缩写/型号”时辅助判断
EN_QUESTION_WORDS = (
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "does", "do", "did", "can", "could",
    "the", "of", "and", "for", "company", "capital", "revenue",
)


def detect_language(text: str) -> str:
    """判断文本语言，返回 ``"zh"`` 或 ``"en"``。

    判定规则：
    - 含中日韩文字 -> ``zh``；
    - 否则若含拉丁字母且看起来像英文句子（含疑问词或 ≥3 个英文单词）-> ``en``；
    - 其余（纯数字、型号如 ``C4ISR``）按 ``zh`` 处理，避免把中文语料里的
      英文缩写误判成英文提问。
    """
    if not text:
        return "zh"
    if CJK_RE.search(text):
        return "zh"
    lowered = text.lower()
    words = [word for word in re.split(r"[^a-z]+", lowered) if word]
    if not words:
        return "zh"
    if any(word in EN_QUESTION_WORDS for word in words):
        return "en"
    return "en" if len(words) >= 3 else "zh"


def resolve_answer_language(question: str, configured: str = "auto") -> str:
    """决定回答语言：``auto`` 时跟随提问语言。"""
    if configured in {"zh", "en"}:
        return configured
    return detect_language(question)


# --------------------------------------------------------------------------
# 领域术语表（中 -> 英）
# --------------------------------------------------------------------------
GLOSSARY_ZH_EN: dict[str, str] = {
    "武汉兴图新科电子股份有限公司": "Wuhan Xingtu Xinke Electronics Co., Ltd.",
    "兴图新科": "Xingtu Xinke",
    "军用领域": "the military sector",
    "军队视频指挥领域": "the military video command sector",
    "国防军队视频指挥领域": "the national defense and military video command sector",
    "主营业务收入": "main business revenue",
    "注册资本": "registered capital",
    "法定代表人": "legal representative",
    "募集资金": "raised funds",
    "补充流动资金": "supplementing working capital",
    "技术标准": "technical standard",
    "技术规范": "technical specification",
    "国家科技进步一等奖": "the First Prize of the National Science and Technology Progress Award",
    "科技进步一等奖": "the First Prize of the National Science and Technology Progress Award",
    "视频指挥系统": "video command system",
    "视频指挥控制系统": "video command and control system",
    "电子信息行业": "the electronic information industry",
    "上游": "upstream",
    "下游": "downstream",
    "供应商": "supplier",
    "重要供应商": "major supplier",
    "招股意向书": "the prospectus",
    "募集资金投资项目": "raised-fund investment projects",
    "研发中心建设项目": "the R&D Center Construction Project",
    "基于云联邦架构的军用视频指挥平台升级及产业化项目": (
        "the Upgrading and Industrialization Project of the Military Video Command Platform "
        "Based on Cloud Federation Architecture"
    ),
    "电子元器件制造企业": "electronic component manufacturers",
    "金属壳体制造企业": "metal enclosure (chassis and cabinet) manufacturers",
    "军队": "the military",
    "政府机关": "government agencies",
    "能源": "energy",
    "报告期内": "during the reporting period",
    "万元": "ten-thousand yuan",
    "亿元": "hundred million yuan",
    "程家明": "Cheng Jiaming",
}

# 报告期口径（英文表述）
REPORTING_PERIOD_EN = (
    "the reporting period (2016, 2017, 2018 and January-June 2019)"
)
PERIOD_LABEL_EN = ("2016", "2017", "2018", "January-June 2019")


def localize(text: str) -> str:
    """把中文文本里的领域术语替换成英文；数字与百分比原样保留。

    先替换长词，避免“军队视频指挥领域”被“军队”先切走。
    """
    result = text
    for chinese, english in sorted(GLOSSARY_ZH_EN.items(), key=lambda item: -len(item[0])):
        result = result.replace(chinese, english)
    return result


def has_untranslated_chinese(text: str) -> bool:
    """是否仍含未替换的中文（用于判断模板化结果是否可读）。"""
    return bool(CJK_RE.search(text))


# --------------------------------------------------------------------------
# 金额/比例的英文表述（确定性，绝不交给模型换算）
# --------------------------------------------------------------------------
def format_amount_en(chinese_amount: str) -> str:
    """把 ``"5,520 万元"`` 这类中文金额转成英文表述。

    ``5,520 万元`` -> ``"RMB 55.20 million (5,520 ten-thousand yuan)"``
    """
    value = parse_amount(chinese_amount)
    if value is None:
        return chinese_amount
    millions = to_million_yuan(value)
    ten_thousands = to_ten_thousand_yuan(value)
    return f"RMB {millions:,.2f} million ({ten_thousands:,.2f} ten-thousand yuan)"


def format_percents_en(percents: list[str]) -> str:
    """把一组百分比拼成英文列表。"""
    if not percents:
        return ""
    if len(percents) == 1:
        return percents[0]
    return ", ".join(percents[:-1]) + " and " + percents[-1]


# --------------------------------------------------------------------------
# 本地翻译模型（Qwen3-0.6B 等）
# --------------------------------------------------------------------------
_TRANSLATION_PROMPT_ZH = (
    "You are a professional translator. Translate the following English question "
    "about a Chinese IPO prospectus into Chinese. Keep company names, numbers, "
    "percentages and standard names unchanged. Output ONLY the Chinese translation, "
    "with no explanation, no pinyin and no quotes.\n\n{text}"
)
_TRANSLATION_PROMPT_EN = (
    "Translate the following Chinese text into concise English. Keep all numbers, "
    "percentages and amounts exactly as they are, and keep 《》 terms as-is. "
    "Output ONLY the English translation, with no explanation and no quotes.\n\n{text}"
)


class LocalTranslator:
    """基于本地小模型（Qwen3-0.6B）的翻译器。

    关键点：``enable_thinking=False``。
    Qwen3 默认开启思考模式，会把 token 预算全用在 ``<think>`` 上导致译文为空；
    显式关闭后是“直接回答”，实测 CPU 上 1~3 秒。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._tokenizer = None
        self._load_failed = False
        self.load_error = ""

    def model_dir(self):
        """本地翻译模型目录；不存在时返回 None。"""
        configured = self.settings.language.translation_model_dir
        if not configured:
            return None
        path = PROJECT_ROOT / configured
        return path if (path / "config.json").exists() else None

    @property
    def available(self) -> bool:
        return (
            self.settings.language.allow_local_translation
            and not self._load_failed
            and self.model_dir() is not None
        )

    def _ensure_model(self) -> bool:
        """惰性加载模型；失败只记录一次，不反复重试。"""
        if self._model is not None:
            return True
        if not self.available:
            return False
        path = self.model_dir()
        try:
            import torch  # 延迟导入
            from transformers import AutoModelForCausalLM, AutoTokenizer

            started = time.perf_counter()
            self._tokenizer = AutoTokenizer.from_pretrained(str(path))
            self._model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32)
            self._model.eval()
            logger.info(
                "app.core.language",
                "本地翻译模型加载完成",
                model=str(path),
                elapsed_s=round(time.perf_counter() - started, 2),
            )
            return True
        except Exception as exc:
            self._load_failed = True
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "app.core.language",
                "本地翻译模型加载失败，英文链路将使用术语表兜底",
                model=str(path),
                error=self.load_error,
            )
            return False

    @trace
    def translate(self, text: str, target: str) -> str:
        """把 ``text`` 翻译成 ``target``（``"zh"`` 或 ``"en"``）；失败返回空串。"""
        if not text.strip():
            return ""
        if not self._ensure_model():
            return ""
        import torch

        template = _TRANSLATION_PROMPT_ZH if target == "zh" else _TRANSLATION_PROMPT_EN
        prompt = template.format(text=text.strip()[:2000])
        messages = [{"role": "user", "content": prompt}]
        try:
            try:
                rendered = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                # 旧版 transformers 不认识 enable_thinking
                rendered = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            inputs = self._tokenizer(rendered, return_tensors="pt")
            started = time.perf_counter()
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.settings.language.translation_max_tokens,
                    do_sample=False,
                )
            generated = self._tokenizer.decode(
                output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
            ).strip()
            # 兜底剥离可能残留的思考块
            generated = re.sub(r"<think>.*?</think>", "", generated, flags=re.DOTALL).strip()
            logger.info(
                "app.core.language",
                "本地翻译完成",
                target=target,
                chars_in=len(text),
                chars_out=len(generated),
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return generated
        except Exception as exc:
            logger.exception("app.core.language", "本地翻译失败", target=target)
            self.load_error = f"{type(exc).__name__}: {exc}"
            return ""


@lru_cache(maxsize=1)
def get_translator() -> LocalTranslator:
    """翻译器单例（模型较重，全进程复用一个）。"""
    return LocalTranslator()


def reset_translator() -> None:
    """清除单例（测试用）。"""
    get_translator.cache_clear()


# --------------------------------------------------------------------------
# 英文检索查询构造
# --------------------------------------------------------------------------
# 英文 -> 中文检索关键词（术语表反向映射 + 常见问法）。
# 顺序即优先级；命中后会把中文词拼进检索串，让中文 BM25 / 向量都能召回。
QUERY_HINTS_EN_ZH: tuple[tuple[str, str], ...] = (
    ("registered capital", "注册资本"),
    ("legal representative", "法定代表人"),
    ("main business revenue", "主营业务收入"),
    ("military sector", "军用领域"),
    ("military", "军用领域"),
    ("working capital", "补充流动资金"),
    ("raised funds", "募集资金"),
    ("proceeds", "募集资金"),
    ("technical standard", "技术标准"),
    ("technical specification", "技术规范"),
    ("standard", "技术标准"),
    ("science and technology progress", "科技进步一等奖"),
    ("award", "科技进步一等奖"),
    ("prize", "科技进步一等奖"),
    ("proportion", "占比 比重"),
    ("percentage", "占比 比重"),
    ("ratio", "占比 比重"),
    ("share", "占比 比重"),
    ("upstream", "上游"),
    ("downstream", "下游"),
    ("supplier", "供应商"),
    ("customer", "客户"),
    ("prospectus", "招股意向书"),
    ("reporting period", "报告期内"),
    ("r&d center", "研发中心建设项目"),
    ("revenue", "收入"),
    ("income", "收入"),
    ("patent", "专利"),
    ("shareholder", "股东"),
    ("subsidiary", "子公司"),
    ("employee", "员工"),
    ("gross margin", "毛利率"),
)


# 每个意图对应的**完整中文问句模板**。
#
# 为什么必须用完整问句而不是关键词列表：实测同一份索引下，
#   "注册资本"                    -> 最高余弦 0.539，命中错误页
#   "注册资本是多少"               -> 最高余弦 0.694，命中正确页
# 原因是语料里的相关段落本身是完整句子，用词组检索会与“释义表”这类
# 同样含该词组的噪声段落竞争。给检索一个自然语句，语义方向才清晰。
CHINESE_QUERY_TEMPLATES: dict[str, str] = {
    "注册资本": "武汉兴图新科电子股份有限公司的注册资本是多少",
    "法定代表人": "武汉兴图新科电子股份有限公司的法定代表人是谁",
    "技术标准": "公司参与制定了哪个技术标准",
    "荣誉奖项": "公司参与的哪个工程荣获了国家科技进步一等奖",
    "上下游": "电子信息行业的上游和下游分别包括哪些企业",
    "供应商客户": "公司在哪个领域已经成为重要供应商",
    "收入占比": "报告期内公司来自军用领域的收入占主营业务收入的比重分别是多少",
    "收入金额": "报告期内公司来自军用领域的收入分别是多少",
    "募资用途": "公司计划使用本次发行募集资金的多少用于补充流动资金",
    "主营业务": "公司的主营业务是什么",
    "股权结构": "公司的股权结构如何",
    "财务数据": "报告期内公司的主要财务数据是多少",
    "行业地位": "公司在行业中的竞争地位如何",
}


# 英文意图关键词 -> 中文意图名（与 query_understanding.INTENT_RULES 的意图名一致）。
#
# 为什么需要它：意图识别规则表是基于中文语料的，英文提问必然全部落到“其他”，
# 于是选不到中文问句模板，检索只能拿到裸关键词（余弦 0.46~0.54，低于阈值），
# 英文问题就全部答不出来。这里补一层英文意图判定。
EN_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("注册资本", ("registered capital", "registered capitals", "capital stock")),
    ("法定代表人", ("legal representative", "legal person", "chairman of the board")),
    ("募资用途", ("raised funds", "proceeds", "ipo proceeds", "working capital", "use of proceeds")),
    ("技术标准", ("technical standard", "technical specification", "industry standard")),
    ("荣誉奖项", ("science and technology progress", "award", "prize", "honor")),
    ("上下游", ("upstream", "downstream", "industry chain", "value chain")),
    ("供应商客户", ("supplier", "customer", "client")),
    ("收入占比", ("percentage", "proportion", "ratio", "share of", "accounted for")),
    ("收入金额", ("revenue", "income", "sales")),
    ("行业地位", ("major supplier", "market position", "competitive position", "market share")),
    ("主营业务", ("main business", "principal business", "business scope")),
    ("股权结构", ("shareholder", "shareholding", "equity structure", "ownership")),
    ("财务数据", ("net profit", "gross margin", "total assets", "financial data")),
)


def detect_english_intent(question: str) -> str:
    """从英文问题里判定意图，返回中文意图名；无法判定时返回空串。"""
    lowered = (question or "").lower()
    for intent, keywords in EN_INTENT_RULES:
        if any(keyword in lowered for keyword in keywords):
            return intent
    return ""


def english_keywords(question: str) -> list[str]:
    """从英文问题里提取中文关键词（术语表命中的中文线索）。

    这些关键词会作为**独立的检索变体**参与多路召回，
    而不是拼进主查询（拼进去会把语义方向拉向“释义”章节）。
    """
    lowered = (question or "").lower()
    hits: list[str] = []
    for english, chinese in QUERY_HINTS_EN_ZH:
        if english in lowered:
            for token in chinese.split():
                if token not in hits:
                    hits.append(token)
    return hits


@trace
def build_chinese_query(
    english_question: str,
    translator: LocalTranslator | None = None,
    intent: str = "",
) -> tuple[str, str]:
    """把英文问题转成适合检索中文语料的查询串。

    策略由 ``RAG_LANGUAGE__QUERY_BRIDGE`` 决定（默认 ``glossary``）:

    - ``glossary``：**中文问句模板 + 术语表 + 专有名词**。毫秒级、完全确定，
      不会出现“小模型把公司名译错”的问题，实测检索效果与人工中文提问相当；
    - ``model``：本地小模型翻译（专有名词先加保护再翻译），质量上限更高，
      但 CPU 上单次 1~10 秒；
    - ``auto``：有本地模型用 ``model``，否则 ``glossary``。

    Args:
        english_question: 英文原问题。
        translator: 翻译器实例（默认单例）。
        intent: Query 理解得到的意图，用于挑选中文问句模板。

    Returns:
        ``(检索用查询串, 使用的方式)``；方式为 ``"model"`` / ``"glossary"`` / ``"raw"``。
    """
    settings = get_settings()
    translator = translator or get_translator()
    lowered = english_question.lower()
    mode = (settings.language.query_bridge or "glossary").lower()

    protected, placeholders = protect_terms(english_question)
    hints = [chinese for english, chinese in QUERY_HINTS_EN_ZH if english in lowered]
    proper = list(placeholders.values())
    # 中文意图识别对英文提问无效，这里用英文规则补判一次
    resolved_intent = intent if CHINESE_QUERY_TEMPLATES.get(intent) else detect_english_intent(english_question)
    template = CHINESE_QUERY_TEMPLATES.get(resolved_intent, "")

    use_model = mode == "model" or (mode == "auto" and translator.available)
    if use_model and translator.available:
        translated = translator.translate(protected, target="zh")
        if translated and CJK_RE.search(translated):
            translated = restore_terms(translated, placeholders)
            # 即使走模型，也补上确定命中的中文线索，降低漏召回
            extra = " ".join(part for part in (template, *proper, *hints) if part)
            query = f"{translated} {extra}".strip() if extra else translated
            logger.info(
                "app.core.language",
                "英文问题已译为中文用于检索",
                english=english_question,
                chinese=query,
                intent=intent,
            )
            return query, "model"
        logger.warning(
            "app.core.language",
            "本地翻译未产出有效中文，改用模板桥接",
            english=english_question,
        )

    # 模板桥接：主查询只用**干净的中文问句**，关键词单独作为补充变体。
    #
    # 为什么不把关键词拼进主查询：实测同一份索引下，
    #   "注册资本是多少"（干净问句）        -> 最高余弦 0.694，命中正确页
    #   "...是多少 注册资本"（问句+关键词） -> 最高余弦 0.539，命中释义表
    # 关键词会把语义方向重新拉向“释义/基本用语”章节。
    # 因此主查询保持干净，关键词交由检索层的多路召回作为独立变体使用。
    if template:
        keyword_tail = " ".join(hints)
        logger.info(
            "app.core.language",
            "使用中文问句模板构造检索查询",
            english=english_question,
            intent=resolved_intent,
            chinese=template,
            keywords=keyword_tail,
        )
        return template, "glossary"

    if hints:
        query = " ".join([*proper, *hints])
        logger.info(
            "app.core.language",
            "无匹配问句模板，退回中文关键词检索",
            english=english_question,
            chinese=query,
        )
        return query, "glossary"

    if proper:
        logger.info(
            "app.core.language",
            "仅有专有名词中文线索，用其检索",
            english=english_question,
            proper=proper,
        )
        return " ".join(proper), "glossary"

    logger.warning(
        "app.core.language",
        "术语表未命中任何线索，保留英文原问题检索（跨语言召回有限）",
        english=english_question,
    )
    return english_question, "raw"


# 需要保护的专有名词/标准名（英文写法 -> 中文原名）
PROPER_NOUN_MAP: dict[str, str] = {
    "wuhan xingtu xinke electronics co., ltd.": "武汉兴图新科电子股份有限公司",
    "wuhan xingtu xinke electronics co ltd": "武汉兴图新科电子股份有限公司",
    "xingtu xinke electronics": "武汉兴图新科电子股份有限公司",
    "xingtu xinke": "兴图新科",
    "c4isr": "C4ISR",
}


def protect_terms(text: str) -> tuple[str, dict[str, str]]:
    """把专有名词替换成占位符，返回 ``(加保护后的文本, 占位符->中文原名)``。"""
    protected = text
    placeholders: dict[str, str] = {}
    for english, chinese in sorted(PROPER_NOUN_MAP.items(), key=lambda item: -len(item[0])):
        if english in protected.lower():
            # 保留原始大小写匹配：用不区分大小写的正则做一次替换
            pattern = re.compile(re.escape(english), re.IGNORECASE)
            token = f"「PH{len(placeholders)}」"
            protected, count = pattern.subn(token, protected)
            if count:
                placeholders[token] = chinese
    return protected, placeholders


def restore_terms(text: str, placeholders: dict[str, str]) -> str:
    """把占位符还原成中文原名。"""
    restored = text
    for token, chinese in placeholders.items():
        restored = restored.replace(token, chinese)
        # 模型可能改写占位符的括号形式，做一次宽松兜底
        digits = re.sub(r"\D", "", token)
        if digits:
            restored = re.sub(rf"[「『\[]?\s*PH\s*{digits}\s*[」』\]]?", chinese, restored)
    return restored
