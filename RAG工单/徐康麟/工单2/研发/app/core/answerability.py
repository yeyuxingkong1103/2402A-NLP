"""可答性闸门：区分「完全无关」与「文档相关但文档未记载答案」。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 检索优化（工单 6.3 + 验收 4「不知道就回不清楚」）

**为什么需要独立闸门**（t13 实测缺陷）：
现有 ``Retriever.is_answerable`` 只看"问题实义词是否出现在片段中"，对**含公司名/文档名**的
无关问题天然放行——「武汉兴图新科电子股份有限公司的食堂今天中午吃什么？」里
``武汉兴图新科电子股份有限公司`` 命中片段、"公司/招股说明书"也被排除在外，
于是覆盖度很高、被误判为可答，抽取式只好硬取一段（实测答成"发行人基本情况 [页码: 52]"）。

本模块在**生成之前**独立判定两件事：
1. **主题覆盖（rarest-first）**：取问题实义词里**语料中最低频（IDF 最高）**的词，
   要求它出现在检索片段里。高频泛词（公司/行业/期间…）不能作为可答证据——
   这正是"含公司名但文档无语料答案"类问题的漏洞所在。
   **前提**：**提问性虚词**（疑问代词、请求动词）必须先被剔除——
   未登录词在 IDF 上天然最高，一旦疑问词漏进主题词表，它必然成为 rarest 且永远不可能出现在证据里，
   **合法问题会被一票否决**（t1 实测：「它的注册地址在哪里？」→ ``rare_term_missing:哪里``）。
2. **意图 ↔ 答案类型一致性**：按 ``intent`` 声明期望的取值类型
   （注册资本/收入→金额；占比→百分比；法定代表人→人名；技术标准→标准名；荣誉→奖项；上下游/客户/供应商→行业与企业名），
   证据里必须真的出现该类型取值，否则判不可答。
3. **非文档型任务请求闸门**（t1 补，captain 独立复核发现）：创作 / 预测 / 主观评价类请求
   （"帮我写一首…的诗""预测一下明年利润""你觉得…值得买吗"）问的是**让系统生成新内容**，
   文档里不存在可抽取的答案跨度。这类问题必须单独判，因为它们的主题词可能**恰好是语料高频词**
   （「股票」在招股书里遍地都是）→ 主题覆盖必然通过、又没有字段取值类型可校验 → 闸门被绕过。

**为什么"语料未登录词"仍然保留否决权（重要，勿改成全部跳过）**：
未登录词分两类，语义相反，不能一刀切——
- **提问性虚词**（哪里/多少/谁/看看…）→ 已由 ``QUESTION_FILLERS`` + ``INTERROGATIVE_PREFIXES`` 剔除，不参与判据；
- **内容性未登录词**（食堂/高铁/奖学金/新能源汽车…）→ 恰恰是"文档从未记载"的证据，**必须**否决。
  实测反例（t1，同机复算）：问「请问北京到上海的高铁要几个小时？」时，若把未登录词一律跳过，
  剩下的**语料内**实义词只剩「北京/上海」——它们都出现在检索片段里（招股书有北京、上海），
  闸门就会放行 → 该题重新变成误答，**验收 4 的 12/12 立刻失守**。

纪律：**不放宽判分口径**（``check_answer``/``FUZZY_THRESHOLD`` 无关），
也**不靠抬高 ``min_overlap_terms``**——这里只做"问题 → 证据"的类型/主题一致性判断。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from app.core.logging_conf import logger
from app.core.text_utils import STOPWORDS, tokenize
from app.models.schemas import RetrievedChunk

#: 主体/文档称谓：出现在问题里不能作为"可答性"证据（t13 缺陷的根源）
NEUTRAL_TERMS: frozenset[str] = frozenset(
    {
        "武汉兴图新科电子股份有限公司",
        "兴图新科",
        "武汉",
        "本公司",
        "发行人",
        "公司",
        "招股说明书",
        "招股意向书",
        "招股书",
        "意向书",
        "文件",
        "文档",
        "资料",
        "报告",
    }
)

#: 泛化提问/衔接词：不承载主题信息
QUESTION_FILLERS: frozenset[str] = frozenset(
    {
        "什么", "怎么", "怎样", "如何", "是否", "有没有", "哪些", "哪个", "哪一", "多少", "几", "几个",
        "请问", "帮我", "帮忙", "推荐", "介绍", "说说", "讲讲", "提到", "提及", "包括", "主要", "相关",
        "关于", "可以", "能否", "是否", "以及", "还有", "就是", "equals", "等于", "写", "一首", "部",
        "好看", "要", "需要", "知道", "告诉", "请问一下", "吗", "呢", "的", "了",
        # —— 疑问代词补全（t1 缺陷修复）——
        # 为什么必须补：未登录词在 IDF 上天然最高（``_idf_lookup`` 对未登录词返回"语料最大 IDF+1"），
        # 只要疑问代词漏进主题词表，它必然成为 rarest，而它永远不可能出现在证据里，
        # 于是**合法问题被一票否决**——实测「它的注册地址在哪里？」→ ``rare_term_missing:哪里``，
        # 在线多轮用例 test_multiturn_v2 因此由绿转红。
        "哪里", "哪儿", "哪边", "在哪", "哪一年", "哪年", "哪天", "何时", "什么时候", "时候",
        "谁", "是谁", "多久", "多长时间", "几点", "多少种", "多大", "什么样",
        # —— 请求/口语化虚词补全（同类，均为"怎么说"而非"问什么"）——
        "看看", "查查", "找找", "了解一下", "说明", "解释", "分析", "总结", "判断", "评价", "麻烦",
    }
)

#: 疑问代词的**词首**兜底（开集）：以这些字/词起首的词都是"提问方式"而非"主题内容"
#: （哪一天 / 哪些方面 / 何地 / 几个人…）。列表中不可能穷举所有组合，故再加一层前缀规则；
#: 前缀命中的词一律不作为可答性证据。
INTERROGATIVE_PREFIXES: tuple[str, ...] = ("哪", "谁", "什", "怎", "几", "多少", "何", "是否", "有没有")

#: 意图 → 期望的取值类型
#:
#: 注意两套意图命名并存（t13 接线缺陷，此处已对齐）：
#: - 中文提问走 ``query_understanding.INTENT_PATTERNS``，标准类意图名是 **"标准"**；
#: - 英文提问经 ``language.detect_english_intent`` 桥接，标准类意图名是 **"技术标准"**。
#: 只登记其中一个会让另一条路径的"取值类型一致性"**静默失效**（闸门退化成纯主题词覆盖）。
INTENT_VALUE_KINDS: dict[str, tuple[str, ...]] = {
    "注册资本": ("amount",),
    "收入": ("amount", "percent"),
    "占比": ("percent",),
    "募资用途": ("amount",),
    "法定代表人": ("person",),
    "标准": ("standard",),
    "技术标准": ("standard",),
    "荣誉": ("award",),
    "上下游": ("enterprise",),
    "供应商": ("enterprise",),
    "客户": ("enterprise",),
}

AMOUNT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:万元|亿元|元)")
PERCENT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*[%％]")
PERSON_RE = re.compile(r"法定代表人\s*[：:]?\s*[\u4e00-\u9fff]{2,4}")
STANDARD_RE = re.compile(r"《[^》]{2,60}》|技术标准|技术规范|标准")
AWARD_RE = re.compile(r"[一二三]等奖|国家科技进步|科学技术进步奖|科技进步奖|荣获|获奖")
ENTERPRISE_RE = re.compile(r"公司|企业|单位|行业|供应商|客户|用户|研究所|集团")

#: 非文档型任务请求（创作 / 预测 / 主观评价）：问的是"让系统生成新内容"，
#: 文档里**不存在可抽取的答案跨度**——问题类型与证据类型天然不一致。
#:
#: 为什么必须单独判（t1 实测，captain 独立复核发现）：
#: 「帮我写一首关于股票的诗」的唯一主题词是「股票」，而招股书里「股票」遍地都是
#: （股票代码/发行股票…）→ 主题覆盖必然通过，又没有字段取值类型可校验，
#: 闸门放行 → 抽取式硬取一段合同叙述（实测答成「华汇创投与杜成城签订了《关于武汉兴图新科电子股份… [页码: 57]」）。
#:
#: 判据要点（避免误杀"文档里写没写 X"这类合法问法）：
#: 创作动词后**紧跟「了/过/的是」的属于"文档记载"**（排除），如「招股说明书写了公司的简介吗」；
#: 只有「写一段/写一首/生成…」这类**请求生成**句式才判不可答。
NON_DOCUMENT_REQUEST_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "creative_task",
        re.compile(
            r"(?:帮我|给我|请|麻烦|帮忙|替我)?\s*(?:写|作|编|拟|生成|创作|整)"
            r"(?![了过的是明白清楚])"
            r"(?:一)?(?:首|篇|段|个|句|条|封)?[^。？！，,]{0,6}"
            r"(?:诗|诗歌|散文|故事|笑话|段子|小说|歌词|文案|邮件|祝福语|简介|文章|作文|台词|剧本|对联)"
        ),
    ),
    (
        "prediction_task",
        re.compile(
            r"(?:帮我|给我|请|麻烦|帮忙)?\s*(?:预测|展望|预判|预估|猜)(?:一下|下)?[^。？！，,]{0,16}"
            r"(?:明年|后年|未来|下一年|股价|走势|利润|收入|营收|业绩)"
        ),
    ),
    (
        "subjective_task",
        re.compile(r"你觉得|你认为|你怎么看|值不值得|该不该|好不好看|推荐.{0,6}买"),
    ),
)


def detect_non_document_request(question: str) -> str:
    """判定"非文档型任务请求"类别；不是该类返回空串。

    Returns:
        命中类别名（``creative_task`` / ``prediction_task`` / ``subjective_task``）或 ``""``。
    """
    try:
        text = (question or "").strip()
        if not text:
            return ""
        for name, pattern in NON_DOCUMENT_REQUEST_PATTERNS:
            if pattern.search(text):
                return name
        return ""
    except Exception:
        logger.exception("app.core.answerability", "非文档型任务识别失败")
        return ""


def question_topic_tokens(question: str) -> list[str]:
    """提取问题的"主题实义词"：去停用词、去主体/文档称谓、去提问性虚词。

    **提问性虚词必须先剔除**（原因见模块 docstring）：未登录词 IDF 天然最高，
    疑问代词一旦成为 rarest，合法问题必被误拒。
    """
    try:
        tokens = []
        for token in tokenize(question or ""):
            word = token.strip().lower()
            if not word or word in STOPWORDS or word in NEUTRAL_TERMS or word in QUESTION_FILLERS:
                continue
            if word.startswith(INTERROGATIVE_PREFIXES):
                continue
            if len(word) < 2 and not word.isascii():
                continue
            if word.isdigit():
                continue
            tokens.append(word)
        # 去重保序
        seen: set[str] = set()
        unique: list[str] = []
        for word in tokens:
            if word not in seen:
                seen.add(word)
                unique.append(word)
        return unique
    except Exception:
        logger.exception("app.core.answerability", "主题词提取失败")
        return []


def expected_value_kinds(intent: str) -> tuple[str, ...]:
    """按意图返回期望的取值类型（空元组 = 不限定类型）。"""
    return INTENT_VALUE_KINDS.get(intent or "", ())


def evidence_has_kind(text: str, kinds: Iterable[str]) -> bool:
    """证据文本里是否真的出现了该类型的取值。"""
    raw = text or ""
    for kind in kinds:
        if kind == "amount" and AMOUNT_RE.search(raw):
            return True
        if kind == "percent" and PERCENT_RE.search(raw):
            return True
        if kind == "person" and PERSON_RE.search(raw):
            return True
        if kind == "standard" and STANDARD_RE.search(raw):
            return True
        if kind == "award" and AWARD_RE.search(raw):
            return True
        if kind == "enterprise" and ENTERPRISE_RE.search(raw):
            return True
    return False


def join_evidence(contexts: list[RetrievedChunk], limit: int = 5) -> str:
    """把检索片段拼成用于判定的证据文本（单行）。"""
    return re.sub(r"\s+", "", " ".join(item.chunk.content for item in contexts[:limit]))


def check(
    question: str,
    intent: str,
    contexts: list[RetrievedChunk],
    idf_lookup: Callable[[str], float],
) -> tuple[bool, str]:
    """可答性闸门。

    Args:
        question: **实际检索串**（英文提问时为桥接后的中文问句）。
        intent: Query 理解给出的意图。
        contexts: 召回片段（按分数降序）。
        idf_lookup: 词 -> IDF（语料内越罕见值越大；未登录词应返回很大的值）。

    Returns:
        ``(是否可答, 原因)``；原因用于日志（``topic_covered`` / ``rare_term_missing`` / ``value_kind_missing`` / ``no_topic_tokens``）。
    """
    try:
        if not contexts:
            return False, "no_contexts"
        # 0) 非文档型任务请求（创作/预测/主观）：与证据类型天然不一致，先于主题覆盖判定
        task = detect_non_document_request(question)
        if task:
            return False, f"non_document_task:{task}"
        evidence = join_evidence(contexts)
        tokens = question_topic_tokens(question)
        if not tokens:
            # 问题本身没有主题实义词（如"1 加 1 等于几？"）：交给既有实义词覆盖校验裁决
            return True, "no_topic_tokens"
        # 1) rarest-first：语料中最低频的主题词必须出现在证据里
        ranked = sorted(tokens, key=lambda word: idf_lookup(word), reverse=True)
        rarest = ranked[0]
        if rarest not in evidence:
            return False, f"rare_term_missing:{rarest}"
        # 2) 意图 ↔ 答案类型一致性
        kinds = expected_value_kinds(intent)
        if kinds and not evidence_has_kind(evidence, kinds):
            return False, f"value_kind_missing:{'/'.join(kinds)}"
        return True, "topic_covered"
    except Exception:
        logger.exception("app.core.answerability", "可答性闸门异常，按可答处理（不误拒正常问题）")
        return True, "check_error"
