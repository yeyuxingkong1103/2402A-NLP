# 工单编号：人工智能NLP-RAG-Query 理解优化任务
# 工单05 - Query 理解优化（多轮对话 + 指代消解/省略补全）
"""
Query 理解：把多轮里的追问改写成**不依赖上下文、可独立检索**的问题。

工单05 原文给的 5 轮对话里，有 3 轮是「不完整的」：

    Q1  报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？   ← 完整
    Q2  他参与的哪个工程荣获了国家科技进步一等奖？                          ← 代词「他」
    Q3  这个公司的法定代表人是谁？                                        ← 代词「这个公司」
    Q4  那武汉力源信息技术股份有限公司呢？                                  ← 话题切换 + 省略问点
    Q5  武汉力源…组织结构图中，哪个销售部的销售处最多？有哪些销售处？          ← 完整

**为什么必须改写**：检索链路全靠问题里的字面内容 ——
  · `route()` 按公司专名做 doc_name 硬过滤（`retriever.py:487`）。「他」「这个公司」匹配不到
    任何专名 → 退化成全库检索，工单03 建立的两文档隔离当场失效；
  · `build_context(hits, query=...)` 用 `_tokens(abstract_query(query))` 选窗口
    （`retriever.py:665`）。问「那力源呢？」时查询词几乎为空 → `best_window` 退化成
    截开头（`retriever.py:624`）→ **可能正好切掉答案，而且不报错**。
所以改写结果必须**同时**喂给 `retrieve()` 与 `build_context()`（见 api/chat.py）。

【路线：规则优先 + LLM 兜底】规则层确定性、零延迟、可单测、可演示（"谁改的、依据哪一轮"
直接摊在界面上）；只有规则**认不出**的疑似指代/省略才调模型兜底。5 轮剧本全部规则可解
（`rewrite_llm_calls == 0`），3 秒预算不受影响。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from app.config import settings
from app.core.doc_profiles import DOC_PROFILES, canonical_entity, doc_ids_by_entity
from app.core.session import COMPANY_SLOT

# ----------------------------------------------------------------------
# 短别名：只在**本模块内部**用于识别「那力源呢？」这类口语追问。
#
# 【为什么不加进 doc_profiles.entity_names】`entity_names` 是 `abstract_query()` 的
# 事实来源 —— 往里加短名会改变已交付 16 题的查询视角（抽象掉更多字），
# 直接威胁「16/16 无回归」这条红线。用一张私有表，等于把新能力关在门外。
_SHORT_ALIASES: dict[str, str] = {
    "力源信息": "liyuan", "力源": "liyuan",
    "兴图新科": "xingtu", "兴图": "xingtu",
}


@lru_cache(maxsize=1)
def _entity_index() -> list[tuple[str, str]]:
    """[(专名, doc_key)]，**按长度降序**（最长匹配优先）。"""
    idx: dict[str, str] = dict(doc_ids_by_entity())
    for alias, key in _SHORT_ALIASES.items():
        idx.setdefault(alias, key)
    return sorted(idx.items(), key=lambda kv: len(kv[0]), reverse=True)


@lru_cache(maxsize=512)
def _spaced_re(name: str) -> re.Pattern[str]:
    """把专名编成**空格不敏感**的正则。

    【为什么必须容空格】doc_profiles 里配的是 "Wuhan P&S Information Technology Co.,Ltd."
    （逗号后无空格），而前端英文示例写作 "..., Co., Ltd."（逗号后有空格）。
    不容错的话英文多轮**永远匹配不上**，会静默退化到 LLM 兜底 —— 白花钱还慢。
    """
    return re.compile(r"\s*".join(re.escape(ch) for ch in name))


@dataclass(frozen=True)
class EntitySpan:
    start: int
    end: int
    doc_key: str
    matched: str


def find_entity_span(text: str) -> EntitySpan | None:
    """最长优先地找出问题里的公司专名。找不到返回 None。"""
    if not text:
        return None
    for name, key in _entity_index():
        m = _spaced_re(name).search(text)
        if m:
            return EntitySpan(m.start(), m.end(), key, name)
    return None


# ----------------------------------------------------------------------
# 指代词
# ----------------------------------------------------------------------
# 中文多字说法（整体替换）
_ZH_PHRASES = ("这个公司", "这家公司", "该公司", "本公司", "贵公司", "发行人",
               "上述公司", "他们", "她们", "它们")
# 【「其」必须防误伤】裸替换会把「其中」拆成「武汉兴图新科电子股份有限公司中」——
# 而且**不报错**，检索只是悄悄变差。所以用前后视排除这些固定搭配。
# 中文多字说法（整体替换，**并吃掉紧随其后的「的」**，见 replace_pronouns）
_ZH_PHRASE_RE = re.compile(
    "(?:" + "|".join(re.escape(p) for p in _ZH_PHRASES) + ")的?")
# 单字代词同样吃掉紧随的「的」—— 理由见 replace_pronouns：
# 让「他的法定代表人是谁？」与「这个公司的法定代表人是谁？」改写出**同一个串**。
_QI_RE = re.compile(r"(?:(?<![尤及])其(?![中他余间实它]))的?")
# 「他」排除「他人」；「它」排除「其它」。同样吃掉紧随的「的」（见 replace_pronouns）
_TA_RE = re.compile(r"(?:(?<!其)他(?![人])|(?<!其)她|(?<!其)它)的?")
# 英文：**不做裸 it**（"is it possible" 会误伤）→ 交给 LLM 兜底
_EN_RE = re.compile(
    r"\b(he|she|they|this company|the company|that company|the issuer)\b", re.I)


def has_pronoun(text: str) -> bool:
    if not text:
        return False
    if any(p in text for p in _ZH_PHRASES):
        return True
    return bool(_QI_RE.search(text) or _TA_RE.search(text) or _EN_RE.search(text))


def replace_pronouns(text: str, entity: str) -> str:
    """把指代词整体替换成实体名。

    【为什么连后面的「的」一起吃掉】**所有**代词（含单字的 他/她/它/其）都吃掉紧随的「的」，
    于是「他的法定代表人是谁？」「这个公司的法定代表人是谁？」改写出**同一个串**
    「武汉兴图新科电子股份有限公司法定代表人是谁？」—— 与工单题面 id=531 逐字相同。

    若不吃「的」，会多出一个字（「…股份有限公司**的**法定代表人是谁？」）：
      · 查询串不同 → 稠密向量不同 → 排序可能与已验收的 16/16 漂移；
      · 两种代词说法产出两个不同的查询串，行为随说法漂移，不好讲也不好测。
    招股书原文本来就用「公司法定代表人」这种不加「的」的写法，所以吃掉它更贴语料。
    """
    out = _ZH_PHRASE_RE.sub(entity, text)
    out = _QI_RE.sub(entity, out)
    out = _TA_RE.sub(entity, out)
    out = _EN_RE.sub(entity, out)
    return out


# ----------------------------------------------------------------------
# 省略式追问：「那X呢？」「X呢？」「what about X?」
# ----------------------------------------------------------------------
_LEAD_RE = re.compile(
    r"^\s*(?:那|那么|请问|麻烦问一下|and|what about|how about|then)\s*", re.I)
_TAIL_RE = re.compile(r"\s*(?:呢|怎么样|如何)?\s*[？?。！!.,，]*\s*$")
_PUNCT_RE = re.compile(r"[\s，,。.、；;：:？?！!「」\"'“”‘’（）()\[\]【】\-—…~/\\]+")
_FILLER = "的了吧啊呀嘛"


def _denuded(text: str) -> str:
    """剥掉引导语、尾部语气词与全部标点空白，剩下的是「实义内容」。"""
    s = _TAIL_RE.sub("", _LEAD_RE.sub("", text.strip()))
    return _PUNCT_RE.sub("", s).strip(_FILLER)


def detect_subject_only_ellipsis(question: str) -> str | None:
    """「纯省略式追问」→ 返回它点名的那份文档 key；否则 None。

    【判据：剥完只剩空】不是模式匹配 —— 模式匹配会把 Q5 那种
    「武汉力源…组织结构图中，哪个销售部…最多？」（点名了公司、但还带一整套新问点）
    也误判成省略式追问。剥完还剩内容 = 本轮自带问点 = 不该继承旧问点。
    """
    if not question or not question.strip():
        return None
    s = _LEAD_RE.sub("", question.strip())
    span = find_entity_span(s)
    if span is None:
        return None
    rest = _denuded(s[:span.start] + s[span.end:])
    return span.doc_key if not rest else None


def _looks_like_ellipsis_surface(question: str) -> bool:
    """表面像省略式追问（留给 LLM 兜底判断的入口条件）。"""
    s = question.strip()
    if not s:
        return False
    if _LEAD_RE.match(s) and len(s) <= 16:
        return True
    return bool(_TAIL_RE.fullmatch(s) and s.rstrip("？?。！!").endswith(("呢", "怎么样", "如何")))


# ----------------------------------------------------------------------
# 问点模板（槽位）
# ----------------------------------------------------------------------
def slotize(text: str) -> tuple[str, str]:
    """把公司专名抽成槽位 → `(模板, 命中的专名)`。无专名则原样返回。"""
    span = find_entity_span(text)
    if span is None:
        return text, ""
    return text[:span.start] + COMPANY_SLOT + text[span.end:], span.matched


def compose(template: str, name: str) -> str:
    return template.replace(COMPANY_SLOT, name, 1)


# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RewriteResult:
    """改写结果。`method` 与 `evidence` 直接上前端 —— 演示时不用口头解释。"""

    original: str
    rewritten: str
    # none / rule-coref（代词消解）/ rule-switch（话题切换+问点继承）/ llm / llm-failed
    method: str = "none"
    resolved_entity: str = ""     # 消解出的规范全称
    carried_intent: str = ""      # 继承的问点（模板去槽位），仅供展示
    evidence: str = ""            # 人话依据

    @property
    def changed(self) -> bool:
        return self.rewritten != self.original

    def as_dict(self) -> dict:
        return {
            "original": self.original,
            "rewritten": self.rewritten,
            "method": self.method,
            "resolved_entity": self.resolved_entity,
            "carried_intent": self.carried_intent,
            "evidence": self.evidence,
        }


def _none(question: str, **kw) -> RewriteResult:
    return RewriteResult(original=question, rewritten=question, method="none", **kw)


# ----------------------------------------------------------------------
def resolve_rules(question: str, *, focus_entity: str = "",
                  intent_template: str = "") -> RewriteResult | None:
    """规则层。返回 `RewriteResult` = 规则给了结论；返回 `None` = 规则弃权（可交 LLM 兜底）。

    决策顺序**是关键**，不能调换：
      ① 省略式追问（点名了公司、但剥掉专名后什么都没剩） → 继承问点模板
      ② 否则若本轮点名了公司 → 不改写（点名即消解）
      ③ 否则若有代词且会话有焦点实体 → 代词消解
      ④ 其余 → None（交给 LLM 兜底 / 最终原样透传）

    【① 必须在 ② 之前】Q4「那武汉力源信息技术股份有限公司呢？」**确实点名了公司**，
    若先判 ② 就会短路成"不改写"，白白丢掉问点继承 —— 这正是本工单的核心考点。
    """
    if not question or not question.strip():
        return _none(question)

    # ① 纯省略式追问 → 话题切换 + 继承问点
    doc_key = detect_subject_only_ellipsis(question)
    if doc_key is not None:
        template = intent_template or ""
        if template and COMPANY_SLOT in template:
            name = canonical_entity(doc_key, english=_is_english(question))
            return RewriteResult(
                original=question,
                rewritten=compose(template, name),
                method="rule-switch",
                resolved_entity=name,
                carried_intent=template.replace(COMPANY_SLOT, ""),
                evidence=f"话题切换 + 继承上一轮问点「{template.replace(COMPANY_SLOT, '…')}」",
            )
        # 问到了一份文档，但会话里还没有可继承的问点 → 无从补全，原样透传（如实标注）
        return _none(question, evidence="识别到话题切换，但会话中没有可继承的问点")

    # ② 本轮点名了公司 → 消解已完成，绝不改写
    #    （这条同时解掉「一句话里既有实体又有代词」的歧义句与 Q5 被误判两个坑）
    span = find_entity_span(question)
    if span is not None:
        return _none(question, resolved_entity=canonical_entity(span.doc_key),
                     evidence="本轮已点名公司，无需改写")

    # ③ 代词消解
    if has_pronoun(question) and focus_entity:
        rewritten = replace_pronouns(question, focus_entity)
        if rewritten != question:
            return RewriteResult(
                original=question, rewritten=rewritten, method="rule-coref",
                resolved_entity=focus_entity,
                evidence=f"指代词指向会话焦点实体「{focus_entity}」",
            )

    # ④ 规则弃权
    return None


def needs_llm_fallback(question: str) -> bool:
    """规则没解出、但**表面像**指代/省略 → 值得花一次模型调用。"""
    if not question or not question.strip():
        return False
    if find_entity_span(question) is not None:
        return False                       # 点名了公司 → 规则②已给出结论
    return has_pronoun(question) or _looks_like_ellipsis_surface(question)


def _is_english(text: str) -> bool:
    """与 generator/_is_english 同口径：ASCII 字母占比超过三成即视为英文。"""
    letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return letters > len(text) * 0.3


# ----------------------------------------------------------------------
_REWRITE_SYSTEM = (
    "你是多轮对话的查询改写器。任务：把用户最新问题改写成**不依赖上下文、可直接检索**的完整问题。"
    "只输出 JSON，不要解释、不要 markdown 代码块。"
)
_REWRITE_USER = """上下文（可能为空）：
最近一轮用户问题：{prev_question}
最近一轮改写后的问题：{prev_rewritten}
当前会话焦点实体：{focus_entity}

用户最新问题：{question}

若最新问题里有代词（他/她/它/这个公司/其）或省略（「那X呢？」「what about X?」），
请补全成独立完整的问题；若本来就不需要改写，**原样返回**。
只输出 JSON：{{"rewritten": "...", "reason": "..."}}"""


def _parse_json(raw: str) -> dict | None:
    import json
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(s[i:j + 1])
    except Exception:  # noqa: BLE001
        return None
    return obj if isinstance(obj, dict) else None


class LLMRewriter:
    """兜底改写器。**只在规则弃权且表面像指代/省略时**被调用。"""

    def __init__(self, client=None) -> None:
        self._client = client

    def _get_client(self):
        if self._client is None:
            from app.core.ollama_client import get_client
            self._client = get_client()
        return self._client

    async def rewrite(self, question: str, *, prev_question: str = "",
                      prev_rewritten: str = "", focus_entity: str = "") -> RewriteResult:
        msgs = [
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user", "content": _REWRITE_USER.format(
                prev_question=prev_question or "（无）",
                prev_rewritten=prev_rewritten or "（无）",
                focus_entity=focus_entity or "（无）",
                question=question)},
        ]
        try:
            raw = await self._get_client().chat(
                msgs, fmt="json", temperature=0.0,
                max_tokens=settings.query_rewrite_max_tokens,
                timeout=settings.query_rewrite_timeout,
                # 【绝不能传更小的 num_ctx】改 Ollama 的上下文长度会触发**模型重载**，
                # 实测冷启动 7.2s —— 为"省 tokens"付出的代价是直接击穿 ≤3s 预算。
            )
        except Exception as e:  # noqa: BLE001
            return RewriteResult(original=question, rewritten=question,
                                 method="llm-failed",
                                 evidence=f"兜底改写失败，已按原问题检索：{e}"[:160])

        obj = _parse_json(raw)
        text = (str(obj.get("rewritten", "")).strip() if obj else "")
        if not text:
            return RewriteResult(original=question, rewritten=question,
                                 method="llm-failed",
                                 evidence="兜底改写未返回可解析结果，已按原问题检索")
        return RewriteResult(original=question, rewritten=text, method="llm",
                             evidence=str(obj.get("reason", ""))[:160])


class QueryUnderstanding:
    """改写入口：规则优先，规则弃权时才走 LLM 兜底。"""

    def __init__(self, *, llm_enabled: bool | None = None, rewriter: LLMRewriter | None = None):
        self.llm_enabled = (settings.query_rewrite_llm_fallback
                            if llm_enabled is None else llm_enabled)
        self.rewriter = rewriter or LLMRewriter()

    async def rewrite(self, question: str, *, focus_entity: str = "",
                      intent_template: str = "", prev_question: str = "",
                      prev_rewritten: str = "") -> RewriteResult:
        rr = resolve_rules(question, focus_entity=focus_entity,
                           intent_template=intent_template)
        if rr is not None:
            return rr
        if self.llm_enabled and needs_llm_fallback(question):
            return await self.rewriter.rewrite(
                question, prev_question=prev_question,
                prev_rewritten=prev_rewritten, focus_entity=focus_entity)
        return _none(question)


def intent_template_of(rewritten_question: str) -> str:
    """从改写后的问题里抽出问点模板（供会话写回）。无公司专名时返回空串。"""
    tmpl, matched = slotize(rewritten_question)
    return tmpl if matched else ""


def next_focus_entity(*, resolved_entity: str = "", doc_key: str = "",
                      previous: str = "") -> str:
    """本轮结束后，会话焦点实体应当是什么。

    优先级：改写消解出的实体 > 本轮路由到的文档的规范全称 > 沿用上一轮的焦点。
    【为什么路由命中也算】完整问题（如 Q5）没有"消解出的实体"，但它点名了公司；
    下一轮若说「他的…」，指的就是这家。沿用旧焦点会让跨文档追问指错公司。
    """
    if resolved_entity:
        return resolved_entity
    if doc_key and doc_key in DOC_PROFILES:
        return canonical_entity(doc_key)
    return previous
