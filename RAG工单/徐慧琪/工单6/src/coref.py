# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化
# 工单编号：人工智能NLP-RAG-混合检索任务
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""
模块：指代消解与省略补全（多轮 Query 理解核心）

分四层，逐层降级：

  L0 直通判定  已含公司全称/别名、或疑问结构完整无代词 → 原样放行，零开销。
               （5 轮验收里 Q1、Q5 都走这层，不付改写成本）
  L1 规则检测  只**定位**不定夺：找出代词/省略的位置与候选实体，产出 Span。
  L2 实体判定  LLM 受约束子任务（候选集内二选一）。
  L3 生成      LLM 生成完整问题（指代走填空、省略走模板补全）。
  L4 兜底      规则模板拼接，保证改写永不静默失败。

为什么不做「把 history 丢给 LLM 让它自由改写」（本机实测，见设计文档 §2.2）：
  qwen2.5:3b 会把「他」换成「这个公司」——代词换代词的无效替换；
  qwen2.5:0.5b 直接原样吐回。把任务拆成受约束子任务后两个模型都可靠。
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import re
import time
from dataclasses import dataclass, field

from src import config
from src import dialogue
from src.dialogue import DialogueState

# ---------------------------------------------------------------------------
# 代词与省略的词表
# ---------------------------------------------------------------------------
PERSON_ZH = ("他们", "她们", "它们", "他", "她", "它")
PERSON_EN = ("they", "them", "he", "him", "she", "her", "it")
DEMONSTRATIVE_ZH = (
    "这个公司", "这家公司", "该家公司", "该公司", "该企业", "该发行人",
    "上述公司", "本公司", "这家企业", "这家", "那个公司",
)
DEMONSTRATIVE_EN = (
    "this company", "the company", "that company", "this issuer",
    "the issuer", "this firm", "the firm",
)
# 单字「其」风险高（「其中」「其他」「尤其」「极其」「与其」都会误命中），要求其前后不是这些构词
_QI_RE = re.compile(r"(?<![尤其中何如极与])其(?![中他实它次余])")
# 单字「他」同理会撞「其他」「他人」「他日」「他乡」「他国」等构词（中文无词边界）。
# 多字项（他们/她们/它们）不受影响：_find_pronoun 取最左最长，同起点时更长者先命中。
_TA_RE = re.compile(r"(?<![其])他(?![人日乡国们])")
# 单字「它」与「他」同类：「其它」是最常见的误命中构词（比「他日/他乡/他国」常见得多）。
# 多字项「它们」仍走 q.find，最左最长保证其优先于单字「它」。
_TA_TWO_RE = re.compile(r"(?<!其)它")

# 省略：「那 X 呢」/「X 呢」/「What about X?」/「How about X?」
_ELLIPSIS_ZH_RE = re.compile(
    r"^(?:那|那么|那末|还有)?\s*(?P<entity>[^，,。？?！!]{2,40}?)\s*(?:呢|呢？|\?|？)?\s*$"
)
_ELLIPSIS_EN_RE = re.compile(
    r"^\s*(?:and\s+)?(?:what|how)\s+about\s+(?P<entity>.{2,60}?)\s*\??\s*$",
    re.IGNORECASE,
)


@dataclass
class Span:
    """一个待消解的指代/省略位置。"""
    kind: str                       # passthrough | person | demonstrative | zero | ellipsis
    text: str = ""                  # 命中的原文片段
    start: int = -1
    end: int = -1
    entity: str = ""                # 已确定的实体（省略句可直接给出；其余待 L2 判定）
    candidates: list[str] = field(default_factory=list)


def _ascii_letter_ratio(text: str) -> float:
    letters = sum(1 for c in text if c.isascii() and c.isalpha())
    return letters / max(len(text), 1)


def detect_language(text: str) -> str:
    """中英判定：ASCII 字母占比 > 0.6 视为英文，与工单3 `query_rewrite.detect_language` 同口径。

    在此基础上**有意多加一道「剥离已知公司实体后复判」**（这是与 query_rewrite
    版本的刻意差异，不是抄漏）：问句外壳是英文、实体却是中文公司全称时
    （如 "what about 武汉力源…?"），占比会被实体名稀释，纯占比会把英文问句判成
    zh，进而送去中文改写模板（L3/L4 的模板语言取决于本判定）。query_rewrite
    没有这条验收用例，故那边保持纯占比未同步；两处差异是有意为之。
    """
    t = text or ""
    if not t:
        return "zh"
    if _ascii_letter_ratio(t) > 0.6:
        return "en"
    shell = t
    for alias, _full in dialogue._ALIASES:
        shell = shell.replace(alias, " ")
    return "en" if _ascii_letter_ratio(shell) > 0.6 else "zh"


# ---------------------------------------------------------------------------
# L0 直通判定
# ---------------------------------------------------------------------------
def is_standalone(question: str, state: DialogueState | None = None) -> bool:
    """问题能否不依赖历史独立成立。

    判定与 `detect_spans` 严格一致：只有 `passthrough` 才算独立。
    两者必须是同一个判据——否则「L0 直通」的调用方会与零指代分支给出相反结论
    （有历史时「注册资本是多少？」应走 zero 消解，而不是被 L0 静默放行）。
    """
    st = state if state is not None else dialogue.DialogueState()
    return detect_spans(question, st)[0].kind == "passthrough"


def _find_pronoun(q: str) -> tuple[str, int, int, str] | None:
    """返回最左最长的代词命中 (kind, start, end, text)；无则 None。"""
    hits: list[tuple[int, int, str, str]] = []      # (start, -len, text, kind)
    low = q.lower()
    for w in DEMONSTRATIVE_ZH:
        p = q.find(w)
        if p >= 0:
            hits.append((p, -len(w), w, "demonstrative"))
    for w in DEMONSTRATIVE_EN:
        p = low.find(w)
        if p >= 0:
            hits.append((p, -len(w), q[p:p + len(w)], "demonstrative"))
    m = _QI_RE.search(q)
    if m:
        hits.append((m.start(), -1, m.group(0), "demonstrative"))
    for w in PERSON_ZH:
        if w == "他":
            # 单字「他」必须走构词保护正则，不能用 q.find（见 _TA_RE）
            tm = _TA_RE.search(q)
            p = tm.start() if tm else -1
        elif w == "它":
            # 单字「它」同理（见 _TA_TWO_RE）；「它们」仍走 q.find，最长者胜
            tm = _TA_TWO_RE.search(q)
            p = tm.start() if tm else -1
        else:
            p = q.find(w)
        if p >= 0:
            hits.append((p, -len(w), w, "person"))
    for w in PERSON_EN:
        p = re.search(rf"\b{re.escape(w)}\b", low)
        if p:
            hits.append((p.start(), -len(w), q[p.start():p.end()], "person"))
    if not hits:
        return None
    hits.sort()                                     # start 升序；同起点取更长
    start, neg_len, text, kind = hits[0]
    return kind, start, start + len(text), text


def _match_ellipsis(q: str) -> tuple[str, str] | None:
    """命中省略句式则返回 (新实体全称, 原文片段)。"""
    m = _ELLIPSIS_EN_RE.match(q)
    if m:
        raw = m.group("entity").strip(" ?？")
        ents = dialogue.extract_companies(raw)
        if ents:
            return ents[0], m.group(0)
    m = _ELLIPSIS_ZH_RE.match(q)
    if m:
        raw = m.group("entity").strip()
        ents = dialogue.extract_companies(raw)
        # 必须是「点名了一家公司」才算省略；否则「注册资本是多少」也会被误判
        # 括号显式化：`A and B or (A and C)` 读起来像另一种结合方式，行为以括号为准
        if (ents and raw == ents[0]) or (ents and len(raw) <= len(ents[0]) + 4):
            return ents[0], m.group(0)
    return None


# ---------------------------------------------------------------------------
# L1 检测
# ---------------------------------------------------------------------------
def detect_spans(question: str, state: DialogueState) -> list[Span]:
    """检测主 span。多命中时取「省略 > 指代 > 零指代」的优先级。"""
    q = dialogue.normalize(question)
    cands = state.companies_seen()

    ell = _match_ellipsis(q)
    if ell:
        entity, raw = ell
        start = q.find(raw) if raw in q else 0
        return [Span(kind="ellipsis", text=raw, start=start, end=start + len(raw),
                     entity=entity, candidates=[entity])]

    pr = _find_pronoun(q)
    if pr:
        kind, start, end, text = pr
        return [Span(kind=kind, text=text, start=start, end=end, candidates=list(cands))]

    # 零指代：有历史焦点、本句无实体、且是完整疑问句
    if cands and not dialogue.extract_companies(q) and \
            re.search(r"[？?]|多少|是什么|是谁|有哪些", q):
        return [Span(kind="zero", text="", start=0, end=0,
                     entity=cands[-1], candidates=list(cands))]

    return [Span(kind="passthrough")]


# ---------------------------------------------------------------------------
# 改写结果
# ---------------------------------------------------------------------------
@dataclass
class RewriteResult:
    resolved: str                       # 改写后的独立问题；""表示无法解析
    mode: str = "passthrough"           # passthrough|coref_llm|coref_rule|ellipsis_llm|ellipsis_rule|unresolved
    span: dict | None = None
    candidates: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    degraded: bool = False              # 是否走了 L4 兜底
    reason: str = ""                    # 降级/未解析原因（成功改写为空串）；Task 9 的退化率诊断依赖它


# ---------------------------------------------------------------------------
# 提示词（受约束子任务——不是「自由改写」，见模块 docstring）
# ---------------------------------------------------------------------------
_JUDGE_SYS_ZH = """你是对话指代判定器。给定对话历史和当前问题，判断当前问题里的代词指的是哪家公司。
候选公司只有：{cands}
只输出公司全称，不要任何解释。若无法判断，只输出：无法判断"""

_JUDGE_SYS_EN = """You resolve pronouns to a company. Given the dialogue history and the current question,
decide which company the pronoun refers to.
Candidates: {cands}
Output ONLY the full company name. If undecidable, output only: 无法判断"""

_GEN_SYS_ZH = """你是多轮对话查询改写器。把用户当前问题改写成不依赖上下文、可独立检索的完整问题。
规则：
1. 只把代词（他/她/它/这个公司/该公司/其）替换成正确的公司全称，其余文字保持不变。
2. 只输出改写后的一句话，不要解释，不要加引号，不要重复本指令。
3. 改写结果必须包含完整的公司全称。"""

_GEN_SYS_EN = """You rewrite a follow-up question into a standalone question that does not depend on context.
Rules:
1. Replace pronouns (he/she/it/this company/the company) with the correct full company name; keep the rest unchanged.
2. Output ONLY the rewritten question. No explanation, no quotes, no repeating these instructions.
3. The result must contain the full company name."""

_INSTRUCTION_TEMPLATES = (_JUDGE_SYS_ZH, _GEN_SYS_ZH, _GEN_SYS_EN, _JUDGE_SYS_EN)


def _default_llm(messages, model: str, **kw):
    from src import llm
    # 改写调用受 ≤3s 预算约束：短超时 + 短生成长度（一句话改写用不着 512 token）。
    # 超时会被 resolve 的 except 捕获 → 降级到 L4（这是设计行为，不是静默失效）。
    return llm.chat(messages, model=model, temperature=0.0,
                    timeout=config.REWRITE_TIMEOUT_S,
                    num_predict=config.REWRITE_NUM_PREDICT)


def _ngrams(text: str, n: int = 4) -> set[str]:
    t = re.sub(r"\s+", "", text or "")
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 0))}


def _content_chars(text: str) -> str:
    """只保留内容字符（去空白与标点、统一小写），用于「最小编辑」判定。"""
    return re.sub(r"[\W_]+", "", (text or ""), flags=re.UNICODE).lower()


def _is_subsequence(needle: str, hay: str) -> bool:
    """needle 是否为 hay 的子序列（允许中间插入代词替换成的公司全称）。"""
    if not needle:
        return True
    it = iter(hay)
    return all(ch in it for ch in needle)


def validate_rewrite(raw: str, original: str, span: Span,
                     state: DialogueState) -> tuple[bool, str]:
    """改写结果校验。任何一条不过都转 L4 兜底——**永不静默放行坏改写**。

    第 3、4 条直接对应两个实机复现过的失败模式（见设计文档 §2.2）。
    """
    s = (raw or "").strip().strip('"“”\'')
    if not s:
        return False, "空输出"
    if len(s) > config.REWRITE_MAX_CHARS:
        return False, f"超长({len(s)})"

    # 3. 指代消解必须含某个候选实体。
    #    ellipsis 同样要查：LLM 对省略句可能返回「他不理解」这类与实体无关的废话，
    #    若只对 person/demonstrative/zero 设卡，坏改写会被当成 ellipsis_llm 放行
    #    （tests/test_coref_resolve.py::test_resolve_ellipsis_falls_back_to_rule 实测）。
    if span.kind in ("person", "demonstrative", "zero", "ellipsis"):
        cands = span.candidates or state.companies_seen()
        if not cands:
            return False, "无候选实体"
        if not any(c in s for c in cands):
            return False, "改写结果未包含任何候选实体"
        # 4. 不得残留任何代词（实测失败模式：3b 把「他」换成「这个公司」）。
        #    判据是「输出还必须不依赖上下文」——不限 kind：跨 kind 的残留
        #    （person 输入 → 输出里剩「这个公司」）同样上下文依赖，必须拒绝。
        leftover = _find_pronoun(s)
        if leftover:
            return False, f"残留代词({leftover[3]})"

    # 5. 不得复述指令（实测失败模式：0.5b 原样吐回指令）
    sn = _ngrams(s)
    for tpl in _INSTRUCTION_TEMPLATES:
        tn = _ngrams(tpl)
        if tn and len(sn & tn) / max(len(sn), 1) > config.REWRITE_NGRAM_REPEAT_THRESHOLD:
            return False, "疑似复述指令"

    if s == (original or "").strip() and span.kind != "passthrough":
        return False, "与原文完全相同（未改写）"

    # 6. 「只替换代词」：原文去掉代词后的文字必须以**子序列**形式保留在改写里。
    #    实测失败模式（Task9 三方评测 d2_person_pronoun_en，唯一输给单轮基线的一轮）：
    #      原文 "Which project did he participate in that won the National Science and
    #            Technology Progress Award?"
    #      3b 改写 "Which project did Microsoft participate in that won t武汉兴图新科电子
    #            股份有限公司 National Science and Technology Progress Award?"
    #    它塞进一个**不存在的公司名**（Microsoft）并吃掉了原文的 "the"，却因「含候选实体 +
    #    无残留代词 + 非空非超长」而**全票通过**旧校验；检索随即一无所获（0 命中），
    #    同一题单轮基线反而命中。L3 指令本就写着「其余文字保持不变」（规则 1），
    #    故这里把这条约束落成校验：非最小编辑一律拒绝 → 走 L4 规则替换（确定性且正确）。
    #    只对替换类 span 生效；ellipsis 是模板补全，句子结构本就不同（另见 rule_fallback）。
    if span.kind in ("person", "demonstrative", "zero"):
        orig = dialogue.normalize(original)
        if 0 <= span.start <= span.end <= len(orig):
            needle = _content_chars(orig[:span.start] + orig[span.end:])
        else:
            needle = _content_chars(orig)
        if not _is_subsequence(needle, _content_chars(s)):
            return False, "改写丢失/改写了原文内容（非最小编辑）"
    return True, ""


def rule_fallback(question: str, span: Span, state: DialogueState) -> str:
    """L4：确定性兜底，保证任何情况下都有可用的独立问题。"""
    q = dialogue.normalize(question)
    if span.kind in ("person", "demonstrative", "zero"):
        entity = span.entity or state.focus_company
        if not entity:
            return ""
        if span.kind == "zero":
            return f"{entity}的{q.lstrip('的')}"
        return q[:span.start] + entity + q[span.end:]
    if span.kind == "ellipsis":
        entity = span.entity
        aspect = state.last_aspect
        if not entity:
            return ""
        if not aspect:
            return ""                      # 无模板可继承 → 交给 unresolved，不猜
        return f"{entity}的{aspect}是谁？"
    return q


def _candidates_for(span: Span, state: DialogueState) -> list[str]:
    cands = list(span.candidates) or state.companies_seen()
    if not cands and state.focus_company:
        cands = [state.focus_company]
    return cands


def _focus_entity(cands: list[str], state: DialogueState) -> str:
    """多候选且 L2 未判定时的实体选择：当前焦点公司在候选集内则优先。

    `focus_company` 是「当前在谈哪家公司」（用户最近点名的），与零指代分支
    `cands[-1]` 的「最新提及优先」同口径。若不做这一步，L4 兜底会取 `cands[0]`
    ——本会话最早提到的公司，实测会把「他」解析到 4 轮前那家公司（错误归属）。
    """
    if state.focus_company and state.focus_company in cands:
        return state.focus_company
    return cands[0] if cands else ""


def resolve(question: str, state: DialogueState, use_llm: bool = True,
            llm_fn=None) -> RewriteResult:
    """把多轮问题改写为独立完整问题（L0→L4 逐层降级）。"""
    t0 = time.time()
    q = dialogue.normalize(question)
    span = detect_spans(q, state)[0]
    llm_caller = llm_fn or (lambda m, **k: _default_llm(m, k.get("model", config.REWRITE_MODEL_GEN)))

    if span.kind == "passthrough":
        return RewriteResult(resolved=q, mode="passthrough",
                             latency_ms=round((time.time() - t0) * 1000, 1))

    cands = _candidates_for(span, state)
    if not cands:
        # 硬性要求 8：指代无法解析时不猜测
        return RewriteResult(resolved="", mode="unresolved", candidates=[],
                             span={"kind": span.kind, "text": span.text},
                             latency_ms=round((time.time() - t0) * 1000, 1),
                             degraded=True, reason="无候选实体")

    # 存在唯一候选时不需要问 LLM——直接定实体，省一次调用
    entity = span.entity or (cands[0] if len(cands) == 1 else "")
    degrade_reason = "" if use_llm else "use_llm=False（纯规则）"

    # L2：实体判定（受约束二选一，实测 0.5b 可靠且快）
    if use_llm and not entity and len(cands) > 1:
        sys_prompt = (_JUDGE_SYS_EN if detect_language(q) == "en" else _JUDGE_SYS_ZH).format(
            cands="、".join(cands))
        try:
            resp = llm_caller([{"role": "system", "content": sys_prompt},
                               {"role": "user", "content": q}], model=config.REWRITE_MODEL_JUDGE)
            picked = (resp.get("answer") or "").strip()
            entity = next((c for c in cands if c in picked), "")
            if not entity:
                degrade_reason = "L2 判定未命中候选"
        except Exception as exc:                                # noqa: BLE001
            entity = ""
            degrade_reason = f"L2 判定异常({type(exc).__name__})"

    # L3：生成完整问题。
    # ellipsis 必须有可继承的 last_aspect（模板）才允许生成：否则 LLM 会凭空编出
    # 一个信息点（实测首轮「那武汉力源…呢？」→ 编出「注册资本」），违反「无法解析
    # 时不猜」；此时直接交给 L4，rule_fallback 返回 "" → unresolved。
    if use_llm and entity and (span.kind != "ellipsis" or state.last_aspect):
        target = entity
        hint = (f"代词「{span.text}」指的是：{target}" if span.kind != "ellipsis"
                else f"当前问题在换话题：新公司是「{target}」，"
                     f"沿用上一轮问的信息点「{state.last_aspect or ''}」补全")
        sys_prompt = _GEN_SYS_EN if detect_language(q) == "en" else _GEN_SYS_ZH
        try:
            resp = llm_caller([{"role": "system", "content": sys_prompt},
                               {"role": "user", "content": f"{hint}\n当前问题：{q}\n改写："}],
                              model=config.REWRITE_MODEL_GEN)
            raw = (resp.get("answer") or "").strip()
            if entity not in raw and target not in raw:
                raw = raw.replace(span.text, target) if span.text else raw
            probe = Span(kind=span.kind, text=span.text, start=span.start, end=span.end,
                         entity=target, candidates=cands)
            ok, why = validate_rewrite(raw, q, probe, state)
            if ok:
                mode = "ellipsis_llm" if span.kind == "ellipsis" else "coref_llm"
                return RewriteResult(resolved=raw, mode=mode,
                                     span={"kind": span.kind, "text": span.text, "entity": target},
                                     candidates=cands,
                                     latency_ms=round((time.time() - t0) * 1000, 1))
            degrade_reason = why
        except Exception as exc:                                # noqa: BLE001
            degrade_reason = f"L3 生成异常({type(exc).__name__})"

    # L4：规则兜底。多候选且 L2 未判定时，当前焦点公司优先（见 _focus_entity）。
    probe = Span(kind=span.kind, text=span.text, start=span.start, end=span.end,
                 entity=entity or _focus_entity(cands, state), candidates=cands)
    out = rule_fallback(q, probe, state)
    if not out:
        return RewriteResult(resolved="", mode="unresolved", candidates=cands,
                             span={"kind": span.kind, "text": span.text},
                             latency_ms=round((time.time() - t0) * 1000, 1), degraded=True,
                             reason=degrade_reason or "无可继承的模板")
    mode = "ellipsis_rule" if span.kind == "ellipsis" else "coref_rule"
    return RewriteResult(resolved=out, mode=mode,
                         span={"kind": span.kind, "text": span.text,
                               "entity": probe.entity},
                         candidates=cands, degraded=True,
                         latency_ms=round((time.time() - t0) * 1000, 1),
                         reason=degrade_reason or "规则兜底")
