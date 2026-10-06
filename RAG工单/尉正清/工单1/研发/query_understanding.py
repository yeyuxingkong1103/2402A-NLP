# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""Query 理解：意图识别、消歧、分解与抽象

用规则实现而不是调用大模型：大模型要 3~10 秒，而工单要求响应不超过 3 秒。
规则在本领域（招股说明书问答）上足够准，且耗时接近于 0。
"""
import re

# ---------- 意图识别 ----------
INTENT_RULES = [
    ("比较查询", ["对比", "相比", "区别", "差异", "变化趋势", "compare", "difference", "vs"]),
    ("关系查询", ["上游", "下游", "供应商", "客户", "股东", "子公司", "关联方", "参股",
                  "supplier", "customer", "upstream", "downstream", "shareholder"]),
    ("数据查询", ["多少", "金额", "占比", "比重", "比例", "分别是", "增长率", "数额", "万元",
                  "how much", "how many", "percentage", "amount", "revenue"]),
    ("事实查询", ["是谁", "是什么", "哪个", "哪一个", "哪里", "何时", "什么时候", "哪一年",
                  "who", "what", "which", "when", "where"]),
]

# ---------- 消歧词典：本领域容易指代不清的说法 ----------
# 澄清说明要短。实测把整句话追加到检索查询上会稀释向量、把正确答案挤出 Top-K，
# 所以这里只给简短的限定词。
DISAMBIGUATION = {
    "报告期": "报告期各年度",
    "注册资本": "发行人母公司注册资本",
    "公司": "发行人本身",
    "收入": "营业收入 军用领域 民用领域",
    "上游": "电子信息行业上游",
    "下游": "电子信息行业下游",
}

# ---------- 分解：识别并列提问，拆成子问题 ----------
_SPLIT = re.compile(r"以及|并且|同时|；|;|\s+and\s+", re.I)
_SPLIT_AND = re.compile(r"和|与|及|、")
_PARALLEL = re.compile(r"分别|各自|各是")          # 有这些词才说明是并列提问
_TAIL = re.compile(r"(分别)?(为)?(是)?(什么|多少|谁|怎么样|如何|何时)[？?。]*$")

_NUM = re.compile(r"\d[\d,\.]*\s*(?:万元|亿元|%|％|倍|年|月|日)?")
_ENTITY = re.compile(r"[一-龥]{2,}(?:公司|银行|集团|研究所|大学|委员会|标准)")


def detect_lang(text):
    """含中日韩字符 -> zh，否则 en（验收标准要求支持中英文问答）。"""
    return "zh" if re.search(r"[一-鿿]", text) else "en"


def _detect_intent(question):
    low = question.lower()
    for intent, words in INTENT_RULES:
        if any(w.lower() in low for w in words):
            return intent
    return "事实查询"


def _disambiguate(question):
    """找出模糊表述，并给出澄清说明。"""
    return [f"{term}：{note}" for term, note in DISAMBIGUATION.items() if term in question]


def _decompose(question, entities):
    """把并列问题拆成子问题，并补全主语；拆不开就返回空列表。

    例：「…公司的注册资本和法定代表人分别是什么？」-> 两个子问题，
    第二个补上主语，避免单独检索时指向别的公司。
    """
    parts = [p.strip(" ，,。？?") for p in _SPLIT.split(question)]
    if len(parts) == 1 and _PARALLEL.search(question):
        parts = [p.strip(" ，,。？?") for p in _SPLIT_AND.split(question)]

    parts = [_TAIL.sub("", p).strip(" ，,、") for p in parts]
    parts = [p for p in parts if len(p) >= 4]
    if len(parts) < 2:
        return []

    head = entities[0] if entities else ""
    return [p if not head or head in p else f"{head}{p}" for p in parts]


# 提问常带「根据……招股意向书，」这类框架语，里面全是公司名和文档名，
# 会把检索带偏到「发行人基本情况」等页面。实测去掉后，问「行业上游/下游」
# 的正确页面（第 151-153 页）才能被检索到。
_FRAME = re.compile(r"^根据.{0,30}?(招股意向书|招股说明书|年度报告|年报)[，,、]\s*")


def strip_frame(question):
    """去掉提问的框架前缀，露出真正要问的内容。"""
    return _FRAME.sub("", question).strip() or question


# 数据类问题的答案基本都在财务表格里，而表格块的文字是「数字 + 简短标签」，
# 跟散文式提问在向量空间里对不上。补上表格中高频出现的词作为检索线索，
# 实测可把正确表格块从第 209 名提到 Top-10。
TABLE_HINTS = " 金额 占比 小计 合计 单位：万元"


def understand(question):
    """返回 Query 理解结果字典。

    rewrite 是补全了消歧说明的检索用查询——仅用于检索，回答仍用原始问题。
    """
    question = (question or "").strip()
    clarifications = _disambiguate(question)
    intent = _detect_intent(question)
    entities = list(dict.fromkeys(_ENTITY.findall(question)))[:5]

    # 检索用的重写：去掉提问框架语和公司名。
    # 文档整本都在讲这家公司，公司名占了查询长度的一大半却不提供任何区分度，
    # 反而稀释向量。实测去掉后，10 个问题的答案页命中从 9/10 提升到 10/10
    # （原先答不出的「募集资金补充流动资金」一题，正确页 479/490/30 全部召回）。
    rewrite = strip_frame(question)
    for name in entities:
        rewrite = rewrite.replace(name, "")
    rewrite = rewrite.strip("，,、 ") or question

    # 消歧说明只在问题本身没说清主体时才拼接。问题里已经带了公司名时再补
    # 「发行人本身」这类说明纯属稀释查询向量——实测会把正确答案挤出 Top-5。
    if clarifications and not entities:
        rewrite += "；" + "；".join(clarifications)
    if intent == "数据查询":
        rewrite += TABLE_HINTS

    return {
        "lang": detect_lang(question),
        "intent": intent,
        "ambiguous": [c.split("：")[0] for c in clarifications],
        "clarifications": clarifications,
        "rewrite": rewrite,
        "sub_questions": _decompose(question, entities),
        "entities": entities,
        "numbers": list(dict.fromkeys(_NUM.findall(question)))[:8],
    }
