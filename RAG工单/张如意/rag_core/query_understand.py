# -*- coding: utf-8 -*-
"""
Query 理解模块
工单编号：人工智能NLP-RAG-Query 理解优化任务
          人工智能NLP-RAG-基于PDF文档的问答系统

实现四大能力（对应工单01「功能详细需求（1）Query 理解」）：
  1. 意图识别   —— 判断用户想问什么类型的信息（数值/事实/关系/观点/操作）
  2. 消歧       —— 处理多义词与指代不明（"这个公司"指的是谁？）
  3. 分解与抽象 —— 复杂问题拆成可独立检索的子问题
  4. 多轮改写   —— 结合对话历史，把省略句补全成自包含的检索式
                  （工单05 多轮对话的核心）

设计取舍：意图识别与指代消解用「规则 + LLM 混合」——
规则负责高频稳定场景（快、零成本、可解释），LLM 负责长尾复杂场景。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import llm


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class QueryUnderstanding:
    raw: str
    rewritten: str = ""                      # 指代消解后的自包含问题
    intent: str = "事实查询"
    entities: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    needs_retrieval: bool = True
    ambiguity: str | None = None
    method: str = "rule"                     # rule | llm | rule+llm

    def to_dict(self) -> dict:
        return {
            "原始问题": self.raw, "改写后": self.rewritten or self.raw,
            "意图": self.intent, "实体": self.entities,
            "子问题": self.sub_questions, "需检索": self.needs_retrieval,
            "歧义提示": self.ambiguity, "识别方式": self.method,
        }


@dataclass
class Turn:
    """一轮对话记录。"""
    question: str
    answer: str = ""
    understanding: QueryUnderstanding | None = None
    docs: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 意图识别
# ---------------------------------------------------------------------------
INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("数值查询", [r"多少", r"几[个位次]", r"比例", r"比重", r"金额", r"增长率",
                  r"占比", r"分别是", r"是多少"]),
    ("实体属性", [r"是谁", r"法定代表人", r"注册资本", r"注册地", r"成立时间",
                  r"董事长", r"总经理"]),
    ("关系查询", [r"关联方", r"控制关系", r"持股", r"子公司", r"母公司", r"股东"]),
    ("列表查询", [r"有哪些", r"包括哪些", r"哪些企业", r"哪些行业", r"哪些项目",
                  r"哪些公司", r"列出"]),
    ("比较查询", [r"对比", r"相比", r"哪个更", r"区别", r"差异", r"增长最快",
                  r"最高", r"最低", r"负增长"]),
    ("图表查询", [r"图[中里]", r"如上图", r"图表", r"图中可以看"]),
    ("操作指令", [r"^上传", r"^解析", r"^重新", r"^帮我(导出|下载)"]),
    ("闲聊", [r"^你好", r"^谢谢", r"^你是谁", r"^在吗"]),
]


def classify_intent(question: str) -> str:
    """规则优先的意图识别（零成本、可解释）。"""
    for intent, pats in INTENT_PATTERNS:
        for p in pats:
            if re.search(p, question):
                return intent
    return "事实查询"


# ---------------------------------------------------------------------------
# 实体与指代
# ---------------------------------------------------------------------------
# 招股书场景的常见实体形态。
# `(?:（[^）]{2,8}）)?` 用于兼容「中国平安保险（集团）股份有限公司」这类
# 带全角括号的工商登记全称——若不允许括号，正则会在「保险」处提前截断。
_ENTITY_PAT = re.compile(
    r"[一-鿿]{2,10}(?:（[^）]{2,8}）)?(?:股份)?"
    r"(?:有限公司|集团|银行|证券|保险|公司)"
)

# 通用后缀本身不构成实体名（避免把「股份有限公司」当成公司）
_GENERIC_NAMES = {
    "股份有限公司", "有限公司", "公司", "集团", "银行", "证券", "保险",
    "本公司", "该公司", "发行人", "股份公司",
}
# 指代词：多轮对话中需要消解的对象
_ANAPHORA = ["他", "她", "它", "他们", "这个公司", "该公司", "这家公司",
             "上述公司", "那家", "这家", "此公司", "本公司", "其", "该公司呢"]


# 不可能作为公司名首字的「噪声字」：介词、连词、动词、量词等。
# 正则 `[一-鿿]{2,10}(?:股份)?有限公司` 会把紧邻的前缀一起吃进来
# （例如「收入占武汉力源信息技术股份有限公司」会匹配出「入占武汉力源…」），
# 因此需要逐字剥掉这些前缀噪声。
# 只收录「在任何语境下都不会作为企业名称首字」的字，避免误伤正常公司名。
# 例如「中国」「中信」「人民」都可能合法开头，故 中/人 均不在此列。
_NAME_NOISE_CHARS = set(
    "与和及对由在从据根入占向为是使让被把到给跟同自于等或的了着过"
    "来去且但而则就还也都将会可要把比较相以及并且之其此该"
    "司本贵该诸位"          # 「本公司」「贵公司」的残留首字
)


def extract_entities(text: str) -> list[str]:
    """抽取文本中的公司类实体（去重保序）。"""
    seen, out = set(), []
    for m in _ENTITY_PAT.finditer(text):
        e = m.group(0).strip()
        # 逐字剥掉首部噪声字，直到剩下一个像公司名的串
        while len(e) > 4 and e[0] in _NAME_NOISE_CHARS:
            e = e[1:]
        # 兜底：仍以介词开头时按常见前缀直接切
        e = re.sub(r"^(与|和|及|对|由|在|从|据|根据)", "", e)
        if e in _GENERIC_NAMES or len(e) < 4:
            continue
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def has_anaphora(text: str) -> bool:
    """判断是否含指代词（需要多轮消解）。"""
    if any(a in text for a in _ANAPHORA):
        return True
    # 「那 X 公司呢？」这类省略式追问
    return bool(re.search(r"^(那|那么|然后|还有)", text.strip()))


# ---------------------------------------------------------------------------
# 指代消解 / 多轮改写（工单05 核心）
# ---------------------------------------------------------------------------
_REWRITE_SYS = """你是多轮对话改写助手。给定【历史对话】和【当前问题】，把当前问题
改写成不依赖上下文、可独立检索的完整问题。

改写规则：
1. 把「他/她/它/这个公司/该公司」等指代词，替换成历史上文提到的具体公司全称
2. 把省略的谓语补全（如「那武汉力源信息技术股份有限公司呢？」应补成
   「武汉力源信息技术股份有限公司的法定代表人是谁？」——补全的谓语取自上文同类问题）
3. 保留原问题中的所有限定条件（报告期、金额单位、年份等）
4. 不要回答问题，只改写问题
5. 只输出改写后的问题文本，不要任何解释"""


def resolve_coreference(question: str, history: list[Turn]) -> str:
    """
    多轮指代消解：把省略/指代的问题改写为自包含问题。

    例（工单05 演示脚本）：
      历史：...兴图新科法定代表人是谁？  → 答案：程家明
      当前：那武汉力源信息技术股份有限公司呢？
      改写：武汉力源信息技术股份有限公司的法定代表人是谁？
    """
    if not history or not has_anaphora(question):
        return question

    # 快速通道：规则能解决就用规则，省一次 LLM 调用
    rule_result = _rule_rewrite(question, history)
    if rule_result:
        return rule_result

    hist_text = "\n".join(
        f"Q：{t.question}\nA：{(t.answer or '')[:120]}" for t in history[-4:]
    )
    try:
        rewritten = llm.chat(
            [{"role": "system", "content": _REWRITE_SYS},
             {"role": "user", "content": f"【历史对话】\n{hist_text}\n\n【当前问题】\n{question}"}],
            temperature=0.0, max_tokens=200, tag="rewrite",
        ).strip()
        rewritten = re.sub(r"^(改写后[:：]?\s*|问题[:：]\s*)", "", rewritten).strip()
        return rewritten or question
    except Exception as e:
        print(f"  [warn] 指代消解失败，使用原问题：{e}")
        return question


def _rule_rewrite(question: str, history: list[Turn]) -> str | None:
    """
    规则改写：处理「那 X 呢？」这类高频句式。
    从上一轮问题里抽出「疑问部分」（去掉公司名），替换公司名即可。
    """
    m = re.match(r"^(?:那|那么|然后|还有)\s*(.+?)(?:呢|呢？|\?|？)?$", question.strip())
    if not m:
        return None
    new_entity = m.group(1).strip().rstrip("呢？?").strip()
    if not new_entity:
        return None

    prev = history[-1].question
    prev_ents = extract_entities(prev)
    if not prev_ents:
        return None

    # 用上一轮问题的「模板」：把旧实体换成新实体
    template = prev
    for e in prev_ents:
        template = template.replace(e, "{ENTITY}")
    if "{ENTITY}" not in template:
        return None
    new_q = template.replace("{ENTITY}", new_entity)
    return new_q if new_q != question else None


# ---------------------------------------------------------------------------
# 问题分解（复杂问题 -> 子问题）
# ---------------------------------------------------------------------------
_DECOMPOSE_SYS = """你是问题分解专家。把复杂的复合问题拆解为若干个可以独立检索的
简单子问题，用于 RAG 系统的多路检索。

拆解原则：
1. 每个子问题只问一件事，能独立检索到答案
2. 保留原问题的关键限定（公司名、时间、指标名）
3. 如果原问题本身已经很简单（单一事实），就原样返回，不要硬拆
4. 最多拆成 4 个子问题

只输出 JSON：{"sub_questions": ["子问题1", "子问题2"]}"""


def decompose(question: str, use_llm: bool = True) -> list[str]:
    """
    问题分解。规则先判断是否需要拆（含「并/和/以及/分别」等并列结构），
    需要时再调 LLM，避免无谓开销。
    """
    # 简单问题不拆
    if len(question) < 25 and not re.search(r"(并|并且|以及|同时|分别|和.*分别|还包括)", question):
        return [question]

    if not use_llm:
        return _rule_decompose(question)

    try:
        resp = llm.chat_json(
            [{"role": "system", "content": _DECOMPOSE_SYS},
             {"role": "user", "content": question}],
            temperature=0.0, max_tokens=400, tag="decompose",
        )
        subs = [s.strip() for s in resp.get("sub_questions", []) if s and s.strip()]
        return subs[:4] or [question]
    except Exception:
        return _rule_decompose(question)


def _rule_decompose(question: str) -> list[str]:
    """规则分解：按并列连词/问号切分。"""
    parts = re.split(r"[？?]|(?:；)|(?:并且)|(?:以及)", question)
    parts = [p.strip() for p in parts if len(p.strip()) > 6]
    return parts if len(parts) > 1 else [question]


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def understand(
    question: str,
    history: list[Turn] | None = None,
    use_llm: bool = True,
) -> QueryUnderstanding:
    """
    Query 理解主流程：指代消解 → 意图识别 → 实体抽取 → 问题分解。
    """
    history = history or []
    qu = QueryUnderstanding(raw=question)

    # 1) 多轮指代消解
    rewritten = resolve_coreference(question, history)
    qu.rewritten = rewritten
    qu.method = "rule+llm" if rewritten != question else "rule"

    # 2) 意图识别（规则即可，稳定可解释）
    qu.intent = classify_intent(rewritten)

    # 3) 实体抽取（优先用当前问题，缺失时从历史继承）
    qu.entities = extract_entities(rewritten)
    if not qu.entities and history:
        for t in reversed(history):
            ents = extract_entities(t.question)
            if ents:
                qu.entities = ents
                break

    # 4) 闲聊 / 操作类不走检索
    if qu.intent in ("闲聊", "操作指令"):
        qu.needs_retrieval = False

    # 5) 复杂问题分解
    if qu.needs_retrieval:
        qu.sub_questions = decompose(rewritten, use_llm=use_llm)

    # 6) 歧义检测：出现多个公司实体但没有明确主语
    ents = extract_entities(rewritten)
    if len(ents) > 1 and not any(k in rewritten for k in ("分别", "对比", "区别")):
        qu.ambiguity = f"问题中出现多个实体（{'、'.join(ents[:3])}），已按全部实体并行检索"

    return qu
