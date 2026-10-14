# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
# 工单编号：人工智能NLP-RAG-Query理解优化
# 工单编号：人工智能NLP-RAG-混合检索任务
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""
模块：对话状态（多轮对话的记忆中枢）

工单5 要求「系统必须维护对话历史」，并在每轮把「省略/指代后的短问题」补全成
「完整独立问题」。补全需要两类上下文：

  · `focus_company` —— 当前在说哪家公司。**只在用户明确点名公司时更新**，其余轮次继承。
    这是 Q2「他」、Q3「这个公司」能指回武汉兴图新科的机制。
  · `last_aspect`   —— 上一轮问的「信息点」（如「法定代表人」）。
    这是 Q4「那武汉力源信息技术股份有限公司呢？」能补全成
    「武汉力源信息技术股份有限公司的法定代表人是谁？」的模板来源。

历史长度控制（工单硬性要求 1「滑动窗口 + 摘要」）：
  · `turns`        —— 全量保留（评测与逐轮留痕需要，不删）
  · `recent_turns` —— 送 LLM 的滑动窗口（默认 3 轮）
  · `summary`      —— 超出窗口的轮次压成的结构化摘要（规则拼接，零延迟）
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import re
from collections import OrderedDict
from dataclasses import dataclass, field

from src import config

# ---------------------------------------------------------------------------
# 公司实体：全称与别名（口径来自 config.DOC_ALIASES，避免两处维护）
# ---------------------------------------------------------------------------
# 工单07 改动：source→公司全称 映射提升到 config.COMPANY_BY_SOURCE（单一事实源），
# 这里直接引用。原硬编码只覆盖 2 份招股书，本语料是 9 份年报，硬编码会漏。
COMPANY_BY_SOURCE: dict[str, str] = dict(config.COMPANY_BY_SOURCE)


def _build_alias_table() -> list[tuple[str, str]]:
    """别名 → 全称。按别名长度降序，保证「武汉兴图新科」先于「兴图」命中。"""
    pairs: list[tuple[str, str]] = []
    for source, full in COMPANY_BY_SOURCE.items():
        pairs.append((full, full))
        for alias in config.DOC_ALIASES.get(source, ()):
            pairs.append((alias, full))
    # 长别名优先，避免短别名截断长别名
    return sorted(set(pairs), key=lambda p: -len(p[0]))


_ALIASES = _build_alias_table()
_COMPANY_FULLS = tuple(COMPANY_BY_SOURCE.values())

# 「问什么」抽取：在**剥离公司实体之后**再匹配，否则「武汉力源信息技术股份有限公司注册资本是多少」
# 会把公司名前缀并进 aspect（实测推演：非锚定的第三条会捕到「司注册资本」而非「注册资本」）。
_ASPECT_PATTERNS = [
    re.compile(r"的(?P<a>[^的，,。？?]{2,14}?)(?:分别是|是多少|是什么|是谁|有哪些|有哪|为多少|多大)"),
    re.compile(r"^(?P<a>[^的，,。？?]{2,14}?)(?:是多少|是什么|是谁|有哪些|分别是|为多少|多大)"),
]
# 疑问词本身不该成为 aspect
_ASPECT_STOPWORDS = {"问题", "什么", "哪个", "哪些", "怎么", "如何", "情况"}


@dataclass
class Turn:
    """一轮对话的完整留痕（评测逐轮判定、引用溯源都要用）。"""
    idx: int
    raw_question: str
    resolved_question: str = ""
    rewrite_mode: str = "passthrough"
    coref_span: dict | None = None
    focus_company: str = ""
    aspect: str = ""
    intent: str = "fact"
    lang: str = "zh"
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


@dataclass
class DialogueState:
    """一个会话的全部状态。可序列化（Streamlit 会话、评测落盘都依赖这一点）。"""
    session_id: str = "default"
    turns: list[Turn] = field(default_factory=list)
    focus_company: str | None = None
    last_aspect: str | None = None
    summary: str = ""
    history_window: int = field(default_factory=lambda: config.DIALOGUE_HISTORY_WINDOW)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def companies_seen(self) -> list[str]:
        """本会话明确出现过的公司全称（去重保序）——指代消解的候选集。"""
        out: list[str] = []
        for t in self.turns:
            for c in extract_companies(t.raw_question):
                if c not in out:
                    out.append(c)
        if self.focus_company and self.focus_company not in out:
            out.append(self.focus_company)
        return out


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_companies(text: str) -> list[str]:
    """抽取文中出现的公司全称（含别名归一），按出现位置排序去重。"""
    t = normalize(text)
    found: list[tuple[int, str]] = []
    consumed: list[tuple[int, int]] = []       # 已占用的字符区间，防止短别名重复命中
    for alias, full in _ALIASES:
        start = 0
        while True:
            pos = t.find(alias, start)
            if pos < 0:
                break
            end = pos + len(alias)
            if not any(s < end and pos < e for s, e in consumed):
                consumed.append((pos, end))
                found.append((pos, full))
            start = pos + 1
    out: list[str] = []
    for _, full in sorted(found, key=lambda x: x[0]):
        if full not in out:
            out.append(full)
    return out


def extract_aspect(question: str) -> str:
    """抽取「问什么」。抽不到返回空串（由调用方用 `core_query` 兜底）。

    先剥离公司实体再匹配——否则「武汉力源信息技术股份有限公司注册资本是多少？」
    会把公司名的尾部并进 aspect。
    """
    q = normalize(question)
    for alias, _full in _ALIASES:
        q = q.replace(alias, "")
    q = q.lstrip("的，,、。；： ")
    for pat in _ASPECT_PATTERNS:
        m = pat.search(q)
        if m:
            a = m.group("a").strip("的了 ")
            if a and a not in _ASPECT_STOPWORDS:
                return a
    return ""


def update_focus(state: DialogueState, raw_question: str) -> str | None:
    """点名公司则更新焦点；否则继承。返回更新后的焦点。"""
    found = extract_companies(raw_question)
    if found:
        state.focus_company = found[-1]        # 一句里出现多家时，以最后提到的为准
    return state.focus_company


def append_turn(state: DialogueState, turn: Turn) -> DialogueState:
    """回填焦点/aspect 后追加，并维护窗口摘要与硬上限。"""
    update_focus(state, turn.raw_question)
    turn.focus_company = state.focus_company or ""

    aspect = turn.aspect or extract_aspect(turn.resolved_question or turn.raw_question)
    turn.aspect = aspect
    if aspect:
        state.last_aspect = aspect
    # 代词轮抽不到 aspect → last_aspect 保持上一轮的值（Q4 省略补全的模板来源）

    state.turns.append(turn)
    _rebuild_summary(state)
    # 硬上限：超出则丢弃最早轮次（摘要已覆盖其信息）
    if len(state.turns) > config.DIALOGUE_MAX_TURNS:
        state.turns = state.turns[-config.DIALOGUE_MAX_TURNS:]
    return state


def recent_turns(state: DialogueState, n: int | None = None) -> list[Turn]:
    """最近 n 轮；n=None 用窗口大小，n<=0 返回空（0 不能被当成默认值，也不能因 [-0:] 取全量）。"""
    n = n if n is not None else state.history_window
    if n <= 0:
        return []
    return state.turns[-n:]


def _rebuild_summary(state: DialogueState) -> None:
    """把窗口之外的轮次压成结构化摘要（规则拼接，零延迟）。

    只保留「哪家公司 + 问了什么」，这正是指代消解与省略补全唯一需要的历史信息，
    比逐字保留原文短得多。
    """
    older = state.turns[: -state.history_window] if state.history_window else state.turns
    if not older:
        state.summary = ""
        return
    lines: list[str] = []
    for t in older:
        who = t.focus_company or "（未指名）"
        what = t.aspect or "未知信息点"
        lines.append(f"{who}：{what}")
    state.summary = "；".join(lines)[-config.DIALOGUE_SUMMARY_MAX_CHARS:]


# ---------------------------------------------------------------------------
# 会话表（LRU）
# ---------------------------------------------------------------------------
_SESSIONS: "OrderedDict[str, DialogueState]" = OrderedDict()


def get_state(session_id: str = "default") -> DialogueState:
    sid = session_id or "default"
    if sid in _SESSIONS:
        _SESSIONS.move_to_end(sid)
        return _SESSIONS[sid]
    st = DialogueState(session_id=sid)
    _SESSIONS[sid] = st
    while len(_SESSIONS) > config.DIALOGUE_MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    return st


def reset_state(session_id: str = "default") -> None:
    _SESSIONS.pop(session_id or "default", None)


def reset_all() -> None:
    _SESSIONS.clear()
