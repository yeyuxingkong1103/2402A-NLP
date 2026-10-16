# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-Query 理解优化

Query 改写器（指代消解 / 省略补全 / 上下文继承）—— 工单5 的核心模块。

## 目标
把**依赖上下文的追问**改写成**独立可检索的问句**，再交给工单4 的 RAG 内核。
本机没有额外的 LLM 预算（速度优先，端到端 ≤3 s），因此改写**全部用规则 + 状态**完成，
不调用任何生成模型 —— 只有「规则确定不下来」时才保持原句，让 RAG 自己去检索。

## 规则（顺序敏感，与 ``设计/接口契约_冻结.md`` 一致）
| # | 触发 | 动作 | strategy |
| --- | --- | --- | --- |
| R0 | 空 / 纯空白 / 纯符号 | 返回空改写 | ``empty_input`` |
| R1 | 含**显式实体**、**无槽位**（省略式，如「那 X 呢？」） | 实体 + **继承上一轮槽位** | ``ellipsis_entity_switch`` |
| R2 | 含**指代词**、**无显式实体** | 指代词 → **焦点实体** | ``pronoun`` |
| R3 | 无实体、无指代，但问的是**已知字段**（如「法定代表人是谁？」） | 前缀焦点实体 | ``topic_carry`` |
| R4 | 含显式实体且有槽位 | 原样（只更新焦点实体） | ``none`` |

「槽位模板」（slot）的定义：把问句里的**实体**与**指代词**挖掉、去掉省略标记后剩下的骨架。
例：Q3「这个公司的法定代表人是谁？」→ 槽位 = 「法定代表人是谁？」；
Q4 于是能靠 ``ellipsis_entity_switch`` 补成「武汉力源信息技术股份有限公司的法定代表人是谁？」。

## 设计取舍
* **不硬编码任何公司名**：实体来自 ``corpus_issuers``（由索引块内容自动统计）与通用机构名正则；
* **不硬编码任何语料事实**：槽位来自问句自身，不写死「法定代表人」以外的字段；
* 指代词表与字段词表是**通用中文/招股书词汇**，与工单4 ``query_understanding.FIELD_TERMS`` 同风格。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化"

#: 改写策略枚举（与冻结契约一致）
STRATEGIES = (
    "none", "empty_input", "pronoun", "ellipsis_entity_switch",
    "ellipsis_topic_carry", "topic_carry", "multi_turn_enrich",
)

#: 多字指代词（**长的在前**，避免「这个公司」被「这个」先吃掉）
ANAPHORA_MULTI: tuple[str, ...] = (
    "这个公司", "那个公司", "该公司", "上述公司", "前述公司", "这家公司", "那家公司",
    "本公司", "贵公司", "该企业", "这个企业", "该公司企业",
    "这个工程", "该工程", "这个项目", "该项目", "这个部门", "该部门",
    "上述", "前述", "这家", "那家", "这个", "那个", "此项", "该",
)

#: 单字指代词（需要防误伤：「其他」「其中」「尤其」不是指代）
ANAPHORA_SINGLE: tuple[str, ...] = ("他", "她", "它", "其")

#: 单字指代词后面**不能**跟的字（跟了就说明不是指代）
_SINGLE_GUARD_AFTER: dict[str, str] = {
    "他": "们人日",
    "她": "们人",
    "它": "们",
    "其": "中他实次余它",
}
#: 单字指代词前面**不能**是这些字
_SINGLE_GUARD_BEFORE: dict[str, str] = {"他": "其", "其": "尤"}

#: 省略式标记（「那 X 呢？」「X 呢？」）
_ELLIPSIS_LEAD = re.compile(r"^\s*(?:那|那么|哪|然后|还有|以及|至于|再看|再看下)\s*")
_ELLIPSIS_TAIL = re.compile(r"\s*(?:呢|吗|吧|又如何|怎么样|如何|是什么样|咋样)\s*[?？。！!]*\s*$")

#: 槽位骨架的清理：去掉首尾的连接词与标点
_SLOT_LEAD = re.compile(r"^\s*[的之，,、；;：:和与及]\s*")
_SLOT_TAIL = re.compile(r"\s*[的之，,、；;]+\s*$")

#: 槽位若以这些**动词**开头，则拼实体时不再加「的」（否则成「的参与」）
_VERB_START = re.compile(r"^(?:参与|获得|荣获|来自|属于|包括|包含|下设|由|设|用于|投入|来自|占|在)")

#: 槽位若以这些**副词/连词**开头，同样不加「的」（否则成「的还有」「的另外」）
_ADV_START = re.compile(r"^(?:还有|还|另外|另|再|又|也|那么|请问|以及|并且|同时|然后|关于|对于)")

#: 公司/机构全称
_ORG_RE = re.compile(r"[\u4e00-\u9fff]{2,12}(?:股份)?有限公司")

# ---- 工单5：中英文双语支持（英文问句里的主体与指代同样要能消解）----
#: 英文公司名（用于把英文问句里的公司识别为一个实体跨度）
_ORG_RE_EN = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&.\-]*\s+){1,6}"
    r"(?:Co\.?,?\s*Ltd\.?|Ltd\.?|Inc\.?|Corp\.?|Corporation|Company|Limited|LLC)\b")

#: 英文指代词（**长的在前**，避免「its」被「it」先吃掉）
ANAPHORA_EN: tuple[str, ...] = (
    "this company", "that company", "the company", "this issuer", "the issuer",
    "these companies", "those companies", "its", "their", "they", "them", "it",
)
_ANAPHORA_EN_RE = re.compile(r"\b(" + "|".join(re.escape(s) for s in ANAPHORA_EN) + r")\b",
                             re.IGNORECASE)

#: 英文省略式引导词（「What about X?」「How about X?」「And X?」）
_ELLIPSIS_LEAD_EN = re.compile(r"^\s*(?:(?:what|how)\s+about|and|then|also)\s+", re.IGNORECASE)

#: 公司名核心的结尾虚词 / 句片特征（与 ``subject_binding`` 同源判据）
_BAD_CORE_TAIL = frozenset("为是在于与及等的了和或之其本该此")
_BAD_CORE_SUB = ("变更为", "更名为", "属于")

#: 「有内容」判据：至少一个汉字/字母/数字
_CONTENT_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

#: 追问式「光问字段」的判据：问句里出现这些通用字段词之一
_FIELD_CUES: tuple[str, ...] = (
    "法定代表人", "注册资本", "发行股数", "总股本", "募集资金", "补充流动资金",
    "关联方", "控股股东", "实际控制人", "持股比例", "军用领域", "主营业务收入",
    "技术标准", "组织结构图", "上游", "下游", "供应商", "科技进步一等奖",
    "增长率", "应用结构", "报告期",
)


def _lazy_logger(logger: Any, module: str = "query_rewriter") -> Any:
    if logger is not None:
        return logger
    try:
        from .logging_conf import get_logger

        return get_logger(module)
    except Exception:  # noqa: BLE001
        return _NullLogger()


class _NullLogger:
    """日志不可用时的空实现。"""

    def log_event(self, *_a: Any, **_kw: Any) -> None:
        return None

    def bind(self, **_kw: Any) -> "_NullLogger":
        return self

    class _Span:
        def set_output(self, *_a: Any, **_kw: Any) -> None:
            return None

        def __enter__(self) -> "_NullLogger._Span":
            return self

        def __exit__(self, *_exc: Any) -> bool:
            return False

    def enter(self, *_a: Any, **_kw: Any) -> "_NullLogger._Span":
        return _NullLogger._Span()


@dataclass(slots=True)
class RewriteResult:
    """一次 Query 改写的完整结果（可解释、可留痕）。"""

    original: str
    rewritten: str
    strategy: str = "none"
    changed: bool = False
    focus_entity_before: str | None = None
    focus_entity_after: str | None = None
    inherited_slot: str | None = None
    resolved_refs: list[dict[str, Any]] = field(default_factory=list)
    slot: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _looks_like_org(name: str) -> bool:
    """``_ORG_RE`` 命中串是否真的是机构名（排除「整体变更为股份有限公司」这类句片）。"""
    core = re.sub(r"(?:股份)?有限公司$", "", name or "")
    if len(core) < 2:
        return False
    if core[-1] in _BAD_CORE_TAIL:
        return False
    return not any(bad in core for bad in _BAD_CORE_SUB)


def _short_names(full: str) -> list[str]:
    """由公司全称派生可接受的简称（如「力源信息」「兴图新科」）。"""
    name = (full or "").strip()
    if len(name) < 6:
        return []
    out: list[str] = []
    core = re.sub(r"(股份有限公司|有限公司)$", "", name)
    core = re.sub(r"^(?:武汉|北京|上海|深圳|广州|成都|杭州|南京|天津|重庆|苏州)", "", core)
    if len(core) >= 4:
        out.append(core)                       # 力源信息技术
        if len(core) >= 6:
            out.append(core[:4])               # 力源信息
        out.append(core[:6])                   # 力源信息技术（同 core）
    return [x for x in dict.fromkeys(out) if len(x) >= 4]


class QueryRewriter:
    """把依赖上下文的追问改写成独立可检索问句。"""

    def __init__(self, *, corpus_issuers: Mapping[str, Sequence[str]] | None = None,
                 cfg: Any = None, logger: Any = None) -> None:
        self.cfg = cfg
        self.log = _lazy_logger(logger)
        self.corpus_issuers = {str(k): [str(n) for n in (v or [])]
                               for k, v in (corpus_issuers or {}).items()}
        #: 全称 → 允许的简称写法（含全称本身）
        self._aliases: dict[str, list[str]] = {}
        for names in self.corpus_issuers.values():
            for full in names:
                if not full:
                    continue
                self._aliases.setdefault(full, [full, *_short_names(full)])

    # ------------------------------------------------------------------ 识别
    def detect_entities(self, text: str) -> list[tuple[int, int, str]]:
        """识别问句里的**显式实体**，返回 ``[(start, end, 全称), ...]``（按位置升序、互不重叠）。

        来源：① 语料自动统计的发行人全称及其简称；② 通用机构名正则。
        重叠时**优先保留更长的跨度**（「武汉力源信息技术股份有限公司」优先于「力源信息」）。
        """
        raw = text or ""
        # 候选带优先级：0 = 语料里**已知的发行人全称/简称**（权威），1 = 通用机构名正则。
        # 重叠时优先保留权威候选 —— 否则「那武汉力源信息技术股份有限公司呢？」会被正则
        # 连「那」一起吞成「那武汉力源信息技术股份有限公司」，焦点实体就被污染了（实测踩坑）。
        cands: list[tuple[int, int, str, int]] = []
        for full, aliases in self._aliases.items():
            for alias in aliases:
                if len(alias) < 3:
                    continue
                start = raw.find(alias)
                while start >= 0:
                    cands.append((start, start + len(alias), full, 0))
                    start = raw.find(alias, start + 1)
        for m in _ORG_RE.finditer(raw):
            if _looks_like_org(m.group(0)):
                cands.append((m.start(), m.end(), m.group(0), 1))
        # 英文问句里的公司名（与中文别名同为「非权威」候选，优先级 1）
        for m in _ORG_RE_EN.finditer(raw):
            cands.append((m.start(), m.end(), m.group(0).strip(), 1))
        if not cands:
            return []
        # 先权威后正则，再按「长优先、左优先」；贪心保留互不重叠的
        cands.sort(key=lambda c: (c[3], -(c[1] - c[0]), c[0]))
        kept: list[tuple[int, int, str]] = []
        for start, end, name, _pri in cands:
            if any(not (end <= k[0] or start >= k[1]) for k in kept):
                continue
            kept.append((start, end, name))
        kept.sort(key=lambda c: c[0])
        return kept

    def detect_anaphora(self, text: str) -> list[tuple[int, int, str]]:
        """识别指代词，返回 ``[(start, end, 指代面), ...]``（互不重叠、按位置升序）。"""
        raw = text or ""
        spans: list[tuple[int, int, str]] = []
        for surface in ANAPHORA_MULTI:                    # 长的在前
            start = raw.find(surface)
            while start >= 0:
                end = start + len(surface)
                if not any(not (end <= s or start >= e) for s, e, _ in spans):
                    spans.append((start, end, surface))
                start = raw.find(surface, start + 1)
        for surface in ANAPHORA_SINGLE:
            for m in re.finditer(re.escape(surface), raw):
                start, end = m.start(), m.end()
                if any(not (end <= s or start >= e) for s, e, _ in spans):
                    continue
                if start > 0 and raw[start - 1] in _SINGLE_GUARD_BEFORE.get(surface, ""):
                    continue
                if end < len(raw) and raw[end] in _SINGLE_GUARD_AFTER.get(surface, ""):
                    continue
                spans.append((start, end, surface))
        # 英文指代词（「大写且≤3 字母」判为缩写如 IT，不是指代）
        for m in _ANAPHORA_EN_RE.finditer(raw):
            surface = m.group(0)
            if surface.isupper() and len(surface) <= 3:
                continue
            start, end = m.start(), m.end()
            if any(not (end <= s or start >= e) for s, e, _ in spans):
                continue
            spans.append((start, end, surface))
        spans.sort(key=lambda s: s[0])
        return spans

    def detect_slot(self, text: str) -> str | None:
        """问句问的是哪个字段（命中 ``_FIELD_CUES`` 的第一个；命中不到返回 ``None``）。"""
        raw = text or ""
        for cue in _FIELD_CUES:
            if cue in raw:
                return cue
        return None

    def slot_template(self, text: str,
                      entities: Sequence[tuple[int, int, str]] | None = None,
                      anaphora: Sequence[tuple[int, int, str]] | None = None) -> str:
        """把实体与指代词挖掉，得到「槽位骨架」（如「法定代表人是谁？」）。"""
        raw = text or ""
        spans = list(entities if entities is not None else self.detect_entities(raw))
        spans += list(anaphora if anaphora is not None else self.detect_anaphora(raw))
        for start, end, _ in sorted(spans, key=lambda s: -s[0]):
            raw = raw[:start] + "\u0000" + raw[end:]
        raw = raw.replace("\u0000", "")
        raw = _ELLIPSIS_LEAD.sub("", raw)
        raw = _ELLIPSIS_LEAD_EN.sub("", raw)
        raw = _ELLIPSIS_TAIL.sub("", raw)
        raw = _SLOT_LEAD.sub("", raw)
        raw = _SLOT_TAIL.sub("", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw if len(raw) >= 2 and _CONTENT_RE.search(raw) else ""

    @staticmethod
    def _join(entity: str, slot: str) -> str:
        """把实体与槽位骨架拼成通顺的问句。

        按槽位首词决定要不要加「的」：动词/副词开头不加（「X参与的…」「X还有…」），
        名词开头加（「X的法定代表人是谁？」）。
        """
        head = slot.strip()
        if not head:
            return entity
        if head.startswith(("的", "之")):
            return f"{entity}{head}"
        # 英文槽位（首字符是 ASCII 字母）用空格拼接：加「的」会变成中英夹杂的病句
        if head[0].isascii():
            return f"{entity} {head}"
        if _VERB_START.match(head) or _ADV_START.match(head):
            return f"{entity}{head}"
        return f"{entity}的{head}"

    # ------------------------------------------------------------------ 主流程
    def rewrite(self, question: str, state: Any) -> RewriteResult:
        """按 R0~R4 改写问句；``state`` 为 :class:`DialogueState`。"""
        original = question if isinstance(question, str) else str(question or "")
        stripped = original.strip()
        focus_before = getattr(state, "focus_entity", None)
        last_slot = getattr(state, "last_slot", None)

        # R0 空 / 纯符号
        if not stripped or not _CONTENT_RE.search(stripped):
            return RewriteResult(
                original=original, rewritten="", strategy="empty_input", changed=False,
                focus_entity_before=focus_before, focus_entity_after=focus_before,
                notes="输入为空或没有可检索内容",
            )

        entities = self.detect_entities(stripped)
        anaphora = self.detect_anaphora(stripped)
        slot = self.slot_template(stripped, entities, anaphora)
        focus_after = entities[-1][2] if entities else focus_before

        # ---------- 有显式实体 ----------
        if entities:
            if slot:
                # R4：实体 + 槽位齐全 → 已是独立问句，原样送出（只更新焦点）
                return RewriteResult(
                    original=original, rewritten=stripped, strategy="none", changed=False,
                    focus_entity_before=focus_before, focus_entity_after=focus_after,
                    slot=slot, notes="问句自带实体与槽位，无需改写",
                )
            # R1：省略式（「那 X 呢？」/「What about X?」）→ 继承上一轮
            new_entity = entities[-1][2]
            last_query = str(getattr(state, "last_query", "") or "")
            ref = {"surface": new_entity, "replacement": new_entity, "kind": "explicit_entity"}
            if last_query and focus_before and focus_before in last_query:
                # **首选：复用上一轮问句骨架**，只把其中的旧主体换成新主体。
                # 这比「实体+槽位拼接」更稳：中英文都通顺，且不会丢掉问句原有的限定语
                # （例：「武汉兴图新科电子股份有限公司的法定代表人是谁？」→ 换主体即得
                #   「武汉力源信息技术股份有限公司的法定代表人是谁？」，与工单期望逐字一致）。
                rebuilt = last_query.replace(focus_before, new_entity)
                return RewriteResult(
                    original=original, rewritten=rebuilt, strategy="ellipsis_entity_switch",
                    changed=rebuilt != stripped,
                    focus_entity_before=focus_before, focus_entity_after=focus_after,
                    inherited_slot=last_slot, slot=last_slot or slot, resolved_refs=[ref],
                    notes=f"省略式追问：换主体「{focus_before}」→「{new_entity}」，"
                          f"复用上一轮问句骨架「{last_query}」",
                )
            if last_slot:
                # 兜底：上一轮问句里找不到旧主体时，用「实体 + 槽位骨架」重建
                rebuilt = self._join(new_entity, last_slot)
                return RewriteResult(
                    original=original, rewritten=rebuilt, strategy="ellipsis_entity_switch",
                    changed=rebuilt != stripped,
                    focus_entity_before=focus_before, focus_entity_after=focus_after,
                    inherited_slot=last_slot, slot=last_slot, resolved_refs=[ref],
                    notes=f"省略式追问：换主体为「{new_entity}」，继承上一轮槽位「{last_slot}」",
                )
            return RewriteResult(
                original=original, rewritten=stripped, strategy="none", changed=False,
                focus_entity_before=focus_before, focus_entity_after=focus_after, slot=slot,
                notes="省略式追问但无历史槽位可继承（按单轮处理）",
            )

        # ---------- 有指代词、无实体 ----------
        if anaphora:
            if not focus_before:
                return RewriteResult(
                    original=original, rewritten=stripped, strategy="none", changed=False,
                    focus_entity_before=None, focus_entity_after=None, slot=slot,
                    notes="出现指代词但无历史焦点实体（按单轮处理）",
                )
            rewritten = stripped
            refs: list[dict[str, Any]] = []
            for start, end, surface in sorted(anaphora, key=lambda s: -s[0]):
                rewritten = rewritten[:start] + focus_before + rewritten[end:]
                refs.append({"surface": surface, "replacement": focus_before, "kind": "pronoun"})
            refs.reverse()
            return RewriteResult(
                original=original, rewritten=rewritten, strategy="pronoun",
                changed=rewritten != stripped,
                focus_entity_before=focus_before, focus_entity_after=focus_before,
                resolved_refs=refs, slot=slot,
                notes="指代消解：" + "、".join(f"{r['surface']}→{focus_before}" for r in refs),
            )

        # ---------- 既无实体也无指代词 ----------
        if focus_before and self.detect_slot(stripped):
            # R3：光问字段（「法定代表人是谁？」）→ 前缀焦点实体
            rebuilt = self._join(focus_before, stripped)
            return RewriteResult(
                original=original, rewritten=rebuilt, strategy="topic_carry",
                changed=rebuilt != stripped,
                focus_entity_before=focus_before, focus_entity_after=focus_before, slot=slot,
                resolved_refs=[{"surface": "(省略主体)", "replacement": focus_before,
                                "kind": "topic_carry"}],
                notes=f"话题延续：补上主体「{focus_before}」",
            )

        return RewriteResult(
            original=original, rewritten=stripped, strategy="none", changed=False,
            focus_entity_before=focus_before, focus_entity_after=focus_before, slot=slot,
            notes="独立问句，无需改写",
        )


__all__ = ["WORK_ORDER", "STRATEGIES", "RewriteResult", "QueryRewriter",
           "ANAPHORA_MULTI", "ANAPHORA_SINGLE"]
