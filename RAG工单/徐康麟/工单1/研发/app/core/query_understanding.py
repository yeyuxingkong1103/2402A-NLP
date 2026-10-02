"""Query 理解：意图识别、消歧、分解抽象、多轮改写。

工单要求（5.4）：
- 意图识别：收入、占比、标准、上下游、注册资本、法定代表人、募资用途等；
- 消歧：处理“报告期内”“军用领域”“主营业务收入”等模糊表述；
- 分解与抽象：把复杂问题拆成子问题；
- 多轮对话：结合历史改写当前问题后再检索；
- 返回给用户时只展示最终答案，不展示中间步骤。

实现说明：
- 本模块**默认走确定性规则**（正则 + 领域词典），零依赖、可测试、毫秒级完成，
  满足首字 < 3 秒的硬指标；
- 若配置了可用的 LLM 服务，可选择用 LLM 做更灵活的改写（见 ``use_llm``）。
"""

from __future__ import annotations

import re

import re

from app.core.config import get_settings
from app.core.language import (
    build_chinese_query,
    detect_english_intent,
    detect_language,
    english_keywords,
    resolve_answer_language,
)
from app.core.logging_conf import logger, trace
from app.core.text_utils import dedupe_keep_order
from app.models.schemas import QueryAnalysis

# 公司全称的各种写法：用于构造“去掉主体名”的检索变体
COMPANY_NAME_ALIASES: tuple[str, ...] = (
    "武汉兴图新科电子股份有限公司",
    "武汉兴图新科电子有限公司",
    "兴图新科",
)

# --------------------------------------------------------------------------
# 意图词典：意图 -> (触发词, 检索关键词增强)
# --------------------------------------------------------------------------
INTENT_RULES: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("注册资本", ("注册资本", "股本"), ("注册资本",)),
    ("法定代表人", ("法定代表人", "法人代表", "董事长"), ("法定代表人",)),
    ("募资用途", ("募集资金", "募投", "补充流动资金", "募资"), ("募集资金", "补充流动资金")),
    ("技术标准", ("技术标准", "技术规范", "标准", "制定"), ("技术标准", "技术规范")),
    ("荣誉奖项", ("科技进步奖", "一等奖", "获奖", "荣誉", "荣获"), ("科技进步奖", "一等奖")),
    ("上下游", ("上游", "下游", "产业链"), ("上游", "下游")),
    ("供应商客户", ("供应商", "客户"), ("供应商", "客户")),
    ("收入占比", ("占比", "比重", "比例", "占主营业务收入"), ("占比", "比重", "主营业务收入")),
    ("收入金额", ("收入", "营业收入", "销售额"), ("收入", "军用领域")),
    ("行业地位", ("重要供应商", "市场地位", "竞争地位", "排名"), ("重要供应商",)),
    ("主营业务", ("主营业务", "主要业务", "经营范围"), ("主营业务",)),
    ("股权结构", ("股东", "股权", "持股"), ("股东", "股权")),
    ("财务数据", ("净利润", "毛利率", "资产负债率", "总资产"), ("净利润", "毛利率")),
]

# 模糊表述 -> 消歧说明与补充检索词
AMBIGUITY_RULES: list[tuple[str, str, tuple[str, ...]]] = [
    (r"报告期(内|间)?", "报告期指 2016 年、2017 年、2018 年和 2019 年 1-6 月", ("报告期", "2016", "2019 年 1-6 月")),
    (r"军用领域", "军用领域指公司产品在军队信息化建设中的应用", ("军用领域", "军队")),
    (r"主营业务收入", "主营业务收入指公司核心业务的收入口径", ("主营业务收入",)),
    (r"本次发行|本次募集", "本次发行指公司首次公开发行股票", ("本次发行", "募集资金")),
    (r"公司|发行人|兴图新科", "均指武汉兴图新科电子股份有限公司", ("武汉兴图新科电子股份有限公司",)),
]

# 复杂问题标志：出现这些词说明需要分解
COMPLEX_MARKERS = ("分别", "以及", "和", "并且", "同时", "各自", "各是多少", "有哪些")

# 领域锚点：问题中出现任意一个，说明它已点明文书主体或**本行业专属议题**
# （上游/下游、技术标准、科技进步奖、供应商/客户等只在这份招股书语境下才有意义），
# 属于自带话题的独立问题，无需依赖历史改写。
#
# 注意：这里刻意**不**收录“法定代表人/注册资本/收入/占比”这类通用词——
# 它们既是议题也可能只是省略式追问的对象（“那法定代表人呢？”），
# 收录了会让多轮追问失效。
DOMAIN_ANCHORS: tuple[str, ...] = (
    "公司",
    "发行人",
    "兴图",
    "新科",
    "股份",
    "有限",
    "招股",
    "意向书",
    "说明书",
    "本次发行",
    "募集资金",
    "主营业务",
    "上游",
    "下游",
    "供应商",
    "客户",
    "技术标准",
    "技术规范",
    "科技进步奖",
    "科技进步一等奖",
    "军用",
    "军品",
    "视频指挥",
    "C4ISR",
)

# 页码引用：“第 12 页”“第12页”
PAGE_PATTERN = re.compile(r"第\s*(\d{1,3})\s*页")
# 报告期年份，用于消歧提示
YEAR_PATTERN = re.compile(r"20\d{2}\s*年?")


class QueryUnderstanding:
    """Query 理解服务。"""

    def __init__(self, use_llm: bool = False) -> None:
        self.settings = get_settings()
        self.use_llm = use_llm

    # ------------------------------------------------------------------
    # 意图识别
    # ------------------------------------------------------------------
    def detect_intent(self, question: str) -> tuple[str, list[str]]:
        """返回 ``(意图, 关键词增强)``。

        命中多个意图时按规则表顺序取第一个（规则已按业务重要性排序）。
        """
        keywords: list[str] = []
        for intent, triggers, extra_keywords in INTENT_RULES:
            if any(trigger in question for trigger in triggers):
                keywords.extend(extra_keywords)
                return intent, dedupe_keep_order(keywords)
        return "其他", keywords

    # ------------------------------------------------------------------
    # 消歧
    # ------------------------------------------------------------------
    def detect_ambiguities(self, question: str) -> tuple[list[str], list[str]]:
        """识别模糊表述，返回 ``(消歧说明, 补充检索词)``。"""
        notes: list[str] = []
        extra: list[str] = []
        for pattern, note, keywords in AMBIGUITY_RULES:
            if re.search(pattern, question):
                notes.append(note)
                extra.extend(keywords)
        return notes, dedupe_keep_order(extra)

    # ------------------------------------------------------------------
    # 分解
    # ------------------------------------------------------------------
    def decompose(self, question: str) -> list[str]:
        """把复杂问题拆成子问题；简单问题返回空列表。"""
        if not any(marker in question for marker in COMPLEX_MARKERS):
            return []
        # 按并列连词切分，保留语义完整的片段
        parts = re.split(r"(?:，|,)?(?:以及|并且|同时|和)(?![一二三四五六七八九十])", question)
        subs = [part.strip("，,。?？ ") for part in parts if len(part.strip()) >= 6]
        if len(subs) <= 1:
            # 无法安全切分时，退化为“金额 + 占比”这类常见组合拆解
            if "分别" in question and "占比" in question:
                subs = [
                    re.sub(r"占[^，。]*的?(比重|占比)分别是多少.*", "分别是多少", question).strip("，,。?？ "),
                    "上述金额占主营业务收入的比重分别是多少",
                ]
            else:
                return []
        return dedupe_keep_order(subs)[:4]

    # ------------------------------------------------------------------
    # 页码过滤
    # ------------------------------------------------------------------
    def extract_page_filter(self, question: str) -> list[int]:
        """从问题中解析“第 N 页”形式的页码过滤条件。"""
        return [int(match.group(1)) for match in PAGE_PATTERN.finditer(question)]

    # ------------------------------------------------------------------
    # 多轮改写
    # ------------------------------------------------------------------
    @trace
    def rewrite_with_history(self, question: str, history: list[tuple[str, str]]) -> tuple[str, bool]:
        """结合历史对话改写当前问题。

        Args:
            question: 当前问题。
            history: 历史消息 ``[(role, content)]``，按时间正序。

        Returns:
            ``(改写后的问题, 是否发生了改写)``

        判定规则（**必须同时满足**，避免把无关新问题误当成追问）：

        1. 问题是省略式追问：以指代词/连接词**开头**（那/那么/还有/另外/这个），
           或带有“呢/?”以外的省略标记且极短（≤8 字）；
        2. 问题本身**没有点明文书主体或核心议题**（不含 ``DOMAIN_ANCHORS``）。

        为什么不能只靠“长度 ≤ 20 且不含公司名”：
        「今天天气怎么样？」同样短、同样不含公司名，且其中“那”只是
        “怎么样”的一部分（不能用子串匹配），若因此被改写成
        “<上一轮问题>（追问：今天天气怎么样？）”，检索会命中上一轮主题，
        于是把上一个问题的答案错答给新问题，同时绕过“不清楚”兜底。

        为什么锚点要取窄：追问「那法定代表人呢？」本身不含公司名，
        必须被识别为追问；“法定代表人”是议题而非主体，因此不列入锚点。
        """
        if not history:
            return question, False

        # 条件一：必须是省略式追问。
        # 用 startswith 而非 in：“那”出现在“怎么样”里并不表示追问。
        followup_prefixes = ("那", "那么", "还有", "另外", "这个", "它", "其", "上述", "其中")
        starts_followup = question.startswith(followup_prefixes)
        # 极短的省略式（“占比呢”“2018 年呢”“法定代表人呢”）
        is_elliptical = len(question) <= 10 and question.rstrip("？?。.！!").endswith(("呢", "那"))
        if not (starts_followup or is_elliptical):
            return question, False

        # 条件二：必须没有点明文书主体/核心议题
        if any(anchor in question for anchor in DOMAIN_ANCHORS):
            return question, False

        # 取最近一条用户问题作为实体来源
        previous_question = ""
        for role, content in reversed(history):
            if role == "user":
                previous_question = content
                break
        if not previous_question:
            return question, False

        # 优先做“实体替换”：把上一轮问题里的**旧议题**换成当前追问的**新议题**，
        # 从而把省略式追问还原成一个完整问句。
        #
        # 为什么不能简单拼接（“上一轮问题（追问：xxx）”）：那样会把两个议题
        # 同时塞进检索串，意图识别会命中先出现的旧议题。
        # 例如上一轮问“注册资本”，追问“那法定代表人呢？”，
        # 拼接后仍然会被判成“注册资本”意图，答非所问。
        substituted = self._substitute_topic(previous_question, question)
        if substituted:
            logger.info(
                "app.core.query_understanding",
                "多轮问题改写完成（实体替换）",
                original=question,
                previous=previous_question,
                rewritten=substituted,
            )
            return substituted, True

        # 无法替换时退化为拼接：仍能带回公司等实体，但意图可能偏离，
        # 因此放在替换失败之后，并保留原问题在末尾以便人工/日志追溯。
        rewritten = f"{previous_question}（追问：{question}）"
        logger.info(
            "app.core.query_understanding",
            "多轮问题改写完成（拼接）",
            original=question,
            previous=previous_question,
            rewritten=rewritten,
        )
        return rewritten, True

    @staticmethod
    def _substitute_topic(previous_question: str, followup_question: str) -> str:
        """把追问里的新议题替换进上一轮问题，返回完整问句；无法替换时返回空串。

        做法：
        1. 用「同义槽位」把议题词分组（同一槽位的词可互相替换）；
        2. 在每个问题里找出**命中的最长槽位词**（先长后短，避免“收入”抢先命中
           “军用领域的收入”）；
        3. 若两轮命中不同槽位，就用新词替换旧词，并清理省略式成分与悬空助词。
        """
        # 槽位：按「长词优先」排序后做匹配，保证短语先于其子串命中
        slots: tuple[tuple[str, ...], ...] = (
            ("报告期内", "报告期"),
            ("占主营业务收入的比重", "占主营业务收入比重", "占比", "比重", "比例"),
            ("来自军用领域的收入", "军用领域的收入", "军用领域", "军品收入"),
            ("补充流动资金", "募集资金", "募投项目", "募资用途"),
            ("国家科技进步一等奖", "科技进步一等奖", "科技进步奖"),
            ("法定代表人", "法人代表", "董事长"),
            ("注册资本", "实收资本", "股本"),
            ("技术标准", "技术规范"),
            ("上游", "下游", "上下游"),
            ("供应商", "客户"),
            ("主营业务", "主要业务", "经营范围"),
        )
        # 扁平化并按长度降序，用于“最长匹配”
        all_words = sorted({word for slot in slots for word in slot}, key=len, reverse=True)

        def best_match(text: str) -> tuple[str, tuple[str, ...]] | None:
            for word in all_words:
                if word in text:
                    for slot in slots:
                        if word in slot:
                            return word, slot
            return None

        new_match = best_match(followup_question)
        old_match = best_match(previous_question)
        if new_match is None or old_match is None:
            return ""
        new_word, new_slot = new_match
        old_word, old_slot = old_match
        if new_slot == old_slot:
            return ""

        rewritten = previous_question.replace(old_word, new_word)
        # 清理省略式成分
        for noise in ("那", "那么", "还有", "另外", "呢"):
            rewritten = rewritten.replace(noise, "")
        # 修复替换后残留的悬空助词：“…法定代表人**是**多少” -> “法定代表人是谁”
        rewritten = re.sub(r"(法定代表人|法人代表|董事长|注册资本|股本)(是)(多少)", r"\1是谁", rewritten)
        rewritten = re.sub(r"的(是多少|分别|比重)", r"\1", rewritten)
        rewritten = rewritten.replace("？？", "？").strip("？?。. ")
        if not rewritten.endswith(("？", "?")):
            rewritten += "？"
        return rewritten if len(rewritten) >= 6 and rewritten != previous_question else ""

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    @trace
    def analyze(self, question: str, history: list[tuple[str, str]] | None = None) -> QueryAnalysis:
        """对问题做完整理解，返回 ``QueryAnalysis``。

        英文提问会被识别为 ``language="en"``，并在此完成“语言桥接”：
        把英文问题转成适合检索中文语料的查询串（见 ``search_query``）。
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        language = resolve_answer_language(question, self.settings.language.default_answer_language)
        rewritten, is_followup = self.rewrite_with_history(question, history or [])
        intent, intent_keywords = self.detect_intent(rewritten)
        ambiguities, ambiguity_keywords = self.detect_ambiguities(rewritten)
        sub_questions = self.decompose(rewritten)
        page_filter = self.extract_page_filter(rewritten)

        # 英文提问：构造中文检索串（中文问句模板优先，本地模型可选）
        bridge = "none"
        search_text = rewritten
        if language == "en":
            # 中文意图规则对英文无效，用英文规则补判，选出中文问句模板
            if intent == "其他":
                english_intent = detect_english_intent(rewritten)
                if english_intent:
                    intent = english_intent
            search_text, bridge = build_chinese_query(rewritten, intent=intent)
            # 英文关键词 -> 中文关键词，作为独立召回变体
            intent_keywords = dedupe_keep_order(list(intent_keywords) + english_keywords(rewritten))

        analysis = QueryAnalysis(
            original=question,
            rewritten=rewritten,
            intent=intent,
            keywords=dedupe_keep_order(intent_keywords + ambiguity_keywords),
            sub_questions=sub_questions,
            ambiguities=ambiguities,
            page_filter=page_filter,
            is_followup=is_followup,
            language=language,  # type: ignore[arg-type]
            search_query=search_text,
            language_bridge=bridge,
        )
        logger.info(
            "app.core.query_understanding",
            "Query 理解完成",
            question=question,
            language=language,
            intent=intent,
            keywords=analysis.keywords,
            sub_questions=sub_questions,
            ambiguities=len(ambiguities),
            page_filter=page_filter,
            is_followup=is_followup,
            language_bridge=bridge,
        )
        return analysis

    # ------------------------------------------------------------------
    def search_queries(self, analysis: QueryAnalysis) -> list[tuple[str, str]]:
        """返回用于检索的**查询变体**列表，按优先级排序。

        Returns:
            ``[(查询串, 变体类型)]``，类型为 ``"clean"`` 或 ``"keyword"``：

            - ``clean``   —— 干净的主查询（中文提问时是原问题；
                              英文提问时是中文问句模板）。**不加关键词后缀**，
                              否则语义方向会被拉向“释义/基本用语”章节；
            - ``keyword`` —— 关键词变体，用于补充召回长尾表述。

        多路召回交给 ``Retriever.retrieve_multi`` 合并，以最优结果为准。
        """
        variants: list[tuple[str, str]] = []

        # 变体一：结合历史改写后的问句（多轮追问的实体还原结果）
        if analysis.rewritten:
            variants.append((analysis.rewritten, "clean"))

        # 变体二：去掉公司全称的版本。
        #
        # 放在关键词变体**之前**很重要：公司全称在“释义/发行人基本情况”章节
        # 高频出现，把它留在查询里会把检索拉向定义性段落。
        # 实测“武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？”
        # 的前几名是 p22/p532 这类基础信息页，而去掉全称后 p194/p154
        # 这些真正含结论的页面才浮上来。
        if detect_language(analysis.original) == "zh":
            base = analysis.rewritten or analysis.original
            for alias in COMPANY_NAME_ALIASES:
                base = base.replace(alias, "")
            base = re.sub(r"\s+", " ", base).strip(" ，,。？?的")
            if len(base) >= 4 and all(base != text for text, _ in variants):
                variants.append((base, "clean"))

        # 变体三：语言桥接后的查询
        # - 中文提问：通常与变体一相同，去重后不重复添加；
        # - 英文提问：这里是中文问句模板，是真正能召回中文语料的主查询。
        if analysis.search_query and all(analysis.search_query != text for text, _ in variants):
            variants.append((analysis.search_query, "clean"))

        # 变体四：关键词（意图词 + 消歧词），仅作补充召回。
        # **不含公司全称**：公司名只起到“锁定主体”的作用，不携带议题信息，
        # 放进词袋反而会把“释义/基本情况”这类页面顶上来
        # （实测 “供应商 客户 武汉兴图新科电子股份有限公司” 的最高余弦 0.829
        #   高于真正贴题的变体，若等权合并会答错）。
        topic_keywords = [
            keyword
            for keyword in analysis.keywords
            if keyword not in COMPANY_NAME_ALIASES
        ]
        if topic_keywords:
            keyword_query = " ".join(topic_keywords)
            if all(keyword_query != text for text, _ in variants):
                variants.append((keyword_query, "keyword"))

        return variants

    def search_query(self, analysis: QueryAnalysis) -> str:
        """构造用于检索的主查询串（干净问句，**不拼关键词**）。

        英文提问时优先使用语言桥接得到的中文问句（``analysis.search_query``），
        否则检索单语中文语料会几乎命中不到内容。
        """
        if analysis.search_query:
            return analysis.search_query
        return analysis.rewritten or analysis.original


def get_query_understanding(use_llm: bool = False) -> QueryUnderstanding:
    """工厂函数。"""
    return QueryUnderstanding(use_llm=use_llm)
