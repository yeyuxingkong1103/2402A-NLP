# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单
# （本文件复用自 工单7/工单10 基座；工单17 只补头部编号，未改动其逻辑）
"""工单：人工智能NLP-RAG-Query 理解优化

对话历史压缩（``compress_history`` / ``CompressedHistory``）—— 把较早的轮次压成一段
紧凑中文摘要（digest），只保留最近 K 轮原文，从而在长对话里把送进 LLM 的上下文从
「O(轮数)」降到「O(1)」。

## 为什么压缩要「面向指代消解」而不是「面向摘要」
本工单的核心能力是**多轮指代消解**：Q3「这个公司的法定代表人是谁？」能否答对，取决于
焦点实体还在不在上下文里。所以 digest 必须保留三类**可复用**信息：

1. **出现过的公司主体**：来自 ``focus_entity`` 字段或问句里 ``…公司`` 片段；
2. **每轮问的字段槽位**：``法定代表人`` / ``注册资本`` / ``主营业务`` …
   （来自 ``slot`` 字段，缺失时从问句里探测字段词）；
3. **最近的结论性答案片段**：最近被丢弃轮次的答案前缀（让「上面说的那个数」有指代对象）。

## 严禁编造
digest **只能**由 ``turns`` 里已有的文本裁剪/拼接而成（公司名、字段名、答案片段都是原文
子串）：不生成、不改写、不补全、不推断。宁可 digest 短，也绝不引入原文没有的事实。

## 硬上限
``max_chars`` 是**硬上限**：先按预算裁剪答案片段，若仍超长则截断并追加 ``…``，
保证 ``len(digest) <= max_chars``（``max_chars <= 0`` 视为禁用摘要，返回空串）。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

WORK_ORDER = "人工智能NLP-RAG 项目-金融问答系统部署"

#: 公司 / 机构名后缀（用于从问句里抽取主体，**原文子串**）
_ORG_SUFFIX = ("股份有限公司", "有限责任公司", "有限公司", "集团", "公司")

#: 字段槽位词表（按长度降序匹配，先长后短避免「资本」盖住「注册资本」）
_SLOT_WORDS = (
    "统一社会信用代码", "国家科技进步一等奖", "法定代表人", "实际控制人", "控股股东",
    "主营业务", "营业收入", "注册资本", "研发投入", "销售部门", "组织结构", "销售处",
    "军用领域", "董事长", "总经理", "成立时间", "注册地址", "办公地址", "员工人数",
    "募集资金", "发行价格", "保荐机构", "会计师事务所", "资产总额", "净利润", "毛利率",
    "客户", "供应商", "专利", "商标", "资质", "股东", "董事", "监事", "高管", "行业",
    "产品", "技术", "市场", "收入", "利润", "产能", "工程", "奖项", "荣誉", "面积",
)

#: 指代词（不作为主体候选）
_ANAPHORA = ("这个公司", "该公司", "这家公司", "本公司", "上述公司", "他", "她", "它",
             "其", "该", "这个", "那个", "上述", "前述", "此项", "这家")

#: 结论性答案片段的最短长度（太短（如「不清楚」）不作为结论）
_MIN_CONCLUSION_CHARS = 8
#: 抽取答案片段时只在答案前若干字符里找公司名（避免正文深处的公司名被误当主体）
_ANSWER_SCAN_CHARS = 48
_MAX_SLOT_WORDS = 6
#: 摘要窗口：最多看「最早 N 轮 + 最近 M 轮」的被丢弃轮次。
#: 依据：digest 有硬上限（默认 240 字符），塞不下更多；而摘要只需保留**主体 / 槽位 / 最近结论**，
#: 首尾各取一段即可覆盖，避免 1000 轮会话里摘要本身变成 O(轮数) 的开销。
_SCAN_HEAD_TURNS = 5
_SCAN_TAIL_TURNS = 30
_DIGEST_SEP = "；"
_ELLIPSIS = "…"

#: 摘要内部的固定小标题（都是本模块自己拼的框架词，不是对外事实）
_HEAD = "前情"

__all__ = ["WORK_ORDER", "CompressedHistory", "compress_history", "normalize_turn",
           "approx_tokens", "detect_slots", "extract_companies"]


def _get(turn: Any, name: str, default: Any = None) -> Any:
    """兼容 ``TurnRecord``（dataclass，slots）与 ``dict`` 两种输入。"""
    if isinstance(turn, dict):
        value = turn.get(name, default)
    else:
        value = getattr(turn, name, default)
    return default if value is None else value


def normalize_turn(turn: Any) -> dict[str, Any]:
    """把一轮归一化成 dict（缺字段填空，不编造内容）。"""
    text = str(_get(turn, "answer", "") or "")
    return {
        "question": str(_get(turn, "question", "") or ""),
        "rewritten_query": str(_get(turn, "rewritten_query", "") or ""),
        "answer": text,
        "slot": _get(turn, "slot", None) or None,
        "focus_entity": _get(turn, "focus_entity", None) or None,
        "strategy": str(_get(turn, "strategy", "none") or "none"),
        "refused": bool(_get(turn, "refused", False) or False),
        "cited_pages": list(_get(turn, "cited_pages", []) or []),
        "ts": float(_get(turn, "ts", 0.0) or 0.0),
        "latency_ms": float(_get(turn, "latency_ms", 0.0) or 0.0),
        "trace_id": str(_get(turn, "trace_id", "") or ""),
    }


def approx_tokens(text: str) -> int:
    """极简 token 近似（中文 ≈ 1 字 1 token）：``len(text)``。仅用于对比压缩前后量级。"""
    return len(text or "")


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def extract_companies(text: str) -> list[str]:
    """从文本里抽出「…公司」主体（**原文子串**，最长优先，绝不编造）。

    * 指代词（「这个公司」「该公司」…）**不算主体**，一律剔除；
    * 与已选更长名字重叠的短名（「…公司」被「…股份有限公司」包含）自动去重。
    """
    src = text or ""
    if not src:
        return []
    found: list[tuple[int, str]] = []
    for suffix in _ORG_SUFFIX:
        start = 0
        while True:
            idx = src.find(suffix, start)
            if idx < 0:
                break
            end = idx + len(suffix)
            # 向前吃掉名称字符（中文 / 字母数字），最多 20 字
            cut = idx
            while cut > 0 and (end - (cut - 1)) <= 20 and re.match(r"[\u4e00-\u9fffA-Za-z0-9]", src[cut - 1]):
                cut -= 1
            name = src[cut:end]
            while name and name[0] in "的了和与及在从对把为是":
                name = name[1:]
            if len(name) >= len(suffix) + 2:
                found.append((cut, name))
            start = idx + len(suffix)
    # 长名优先（「…股份有限公司」胜过「…公司」）
    found.sort(key=lambda pair: (-len(pair[1]), pair[0]))
    names: list[str] = []
    for _pos, name in found:
        if any(word in name for word in _ANAPHORA):
            continue           # 「这个公司」「该公司」是指代词，不是主体
        if len(name) < 4:
            continue
        if name in names:
            continue
        if any(name in kept and name != kept for kept in names):
            continue           # 已被更长的同族名字覆盖
        names.append(name)
    return _dedupe_keep_order(names)


def detect_slots(text: str, limit: int = _MAX_SLOT_WORDS) -> list[str]:
    """从问句里探测字段槽位词（按出现顺序，最多 ``limit`` 个）。

    被更长槽位包含的短词会被剔除（例如同一句里同时命中「营业收入」与「收入」时只留前者），
    避免摘要里出现无信息的重复字段。
    """
    src = text or ""
    if not src:
        return []
    hits: list[tuple[int, str]] = []
    for word in _SLOT_WORDS:
        idx = src.find(word)
        if idx >= 0:
            hits.append((idx, word))
    hits.sort(key=lambda pair: pair[0])
    ordered = _dedupe_keep_order([word for _pos, word in hits])
    kept: list[str] = []
    for word in ordered:
        if any(word != other and word in other for other in ordered):
            continue           # 被更长槽位覆盖的短词
        kept.append(word)
    return kept[:max(1, int(limit))]


def _snippet(text: str, budget: int) -> str:
    """取答案片段（只裁剪，不改写）：先切到首个句末标点，再按预算截断。"""
    src = (text or "").strip()
    if not src or budget <= 0:
        return ""
    cut = len(src)
    for mark in "。！？；\n":
        pos = src.find(mark)
        if 0 <= pos < cut:
            cut = pos
    piece = src[:cut].strip()
    if not piece:
        piece = src.strip()
    if len(piece) > budget:
        piece = piece[:budget].rstrip() + _ELLIPSIS
    return piece


def _truncate(text: str, max_chars: int) -> str:
    """硬上限截断：``len(结果) <= max_chars``（截断时末位是 ``…``）。"""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars == 1:
        return _ELLIPSIS
    return text[:max_chars - 1].rstrip() + _ELLIPSIS


@dataclass(slots=True)
class CompressedHistory:
    """压缩结果：一段摘要 + 最近 K 轮原文 + 丢弃轮数 + 焦点实体。"""

    digest: str
    kept_turns: list[dict[str, Any]] = field(default_factory=list)
    dropped_turns: int = 0
    focus_entity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compress_history(turns: Sequence[Any], *, keep_recent: int = 3,
                     max_chars: int = 240,
                     focus_entity: str | None = None) -> CompressedHistory:
    """把较早的轮次压成一段摘要，只保留最近 ``keep_recent`` 轮原文。

    参数：
        turns:        ``TurnRecord`` 或 ``dict`` 序列（两种都支持；时间正序）。
        keep_recent:  保留最近多少轮原文（``>=1``）。
        max_chars:    digest 的**硬上限**（字符数；``<=0`` 表示不产出摘要）。
        focus_entity: 当前焦点实体（用于 ``CompressedHistory.focus_entity`` 回填）。

    返回：
        :class:`CompressedHistory`。无历史 / 无需压缩时 ``digest == ""``、``dropped_turns == 0``。
    """
    items = [normalize_turn(t) for t in (turns or [])]
    keep_n = max(1, int(keep_recent))
    limit = int(max_chars)
    if not items:
        return CompressedHistory(digest="", kept_turns=[], dropped_turns=0,
                                focus_entity=focus_entity or None)

    kept = items[-keep_n:]
    dropped = items[:-keep_n] if len(items) > keep_n else []
    resolved_focus = focus_entity or _last_focus(items)
    if not dropped:
        return CompressedHistory(digest="", kept_turns=kept, dropped_turns=0,
                                focus_entity=resolved_focus)

    digest = _build_digest(dropped, limit)
    return CompressedHistory(digest=digest, kept_turns=kept, dropped_turns=len(dropped),
                             focus_entity=resolved_focus)


def _last_focus(items: list[dict[str, Any]]) -> str | None:
    """最近的显式 ``focus_entity``（没有则 None，**不从答案里猜**）。"""
    for item in reversed(items):
        name = str(item.get("focus_entity") or "").strip()
        if name:
            return name
    return None


def _scan_window(dropped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """摘要扫描窗口：最早 ``_SCAN_HEAD_TURNS`` 轮 + 最近 ``_SCAN_TAIL_TURNS`` 轮（去重、时间正序）。

    digest 有硬上限，扫描全部历史既无用又让耗时随轮数线性增长；首段带得来「最早出现的
    主体」，尾段带得来「最近结论」，已足够指代消解使用。
    """
    if len(dropped) <= _SCAN_HEAD_TURNS + _SCAN_TAIL_TURNS:
        return dropped
    head = dropped[:_SCAN_HEAD_TURNS]
    tail = dropped[-_SCAN_TAIL_TURNS:]
    return head + tail


def _build_digest(dropped: list[dict[str, Any]], max_chars: int) -> str:
    """用被丢弃轮次的**原文子串**拼摘要（不生成、不改写）。"""
    if max_chars <= 0 or not dropped:
        return ""
    scan = _scan_window(dropped)

    # ① 主体：focus_entity 字段优先，其次问句/改写问句里的「…公司」
    entities: list[str] = []
    for item in scan:
        name = str(item.get("focus_entity") or "").strip()
        if name:
            entities.append(name)
    for item in scan:
        entities.extend(extract_companies(str(item.get("question") or "")))
        entities.extend(extract_companies(str(item.get("rewritten_query") or "")))
    entities = _dedupe_keep_order(entities)[:4]

    # ② 槽位：slot 字段优先，其次问句/改写问句里探测到的字段词
    slots: list[str] = []
    for item in scan:
        slot = str(item.get("slot") or "").strip()
        if slot:
            slots.extend(detect_slots(slot, limit=2))
    for item in scan:
        slots.extend(detect_slots(str(item.get("question") or ""), limit=3))
        slots.extend(detect_slots(str(item.get("rewritten_query") or ""), limit=3))
    slots = _dedupe_keep_order(slots)[:_MAX_SLOT_WORDS]

    # ③ 结论：最近 2 个被丢弃轮次的答案前缀（拒绝/过短的不算结论）
    conclusions: list[str] = []
    for item in reversed(dropped):
        if len(conclusions) >= 2:
            break
        if item.get("refused"):
            continue
        raw = str(item.get("answer") or "").strip()
        if len(raw) < _MIN_CONCLUSION_CHARS:
            continue
        # 去掉与本轮主体同名的前缀，避免「武汉力源…：武汉力源…」重复
        head = raw[:_ANSWER_SCAN_CHARS]
        for name in entities:
            if head.startswith(name):
                raw = raw[len(name):].lstrip("：: ，,的")
                break
        if len(raw) < _MIN_CONCLUSION_CHARS:
            continue
        conclusions.append(raw)
    conclusions.reverse()      # 恢复时间正序

    # 预算分配：结论片段拿剩余额度的 ~45%，且每段不超过 40 字符
    head_len = len(_HEAD) + 4
    named_len = sum(len(e) + 1 for e in entities)
    slot_len = sum(len(s) + 1 for s in slots)
    reserve = head_len + named_len + slot_len + 24
    budget = max(0, max_chars - reserve)
    per_conclusion = min(40, max(0, int(budget * 0.45)))
    pieces = [_snippet(text, per_conclusion) for text in conclusions]
    pieces = [p for p in pieces if p]

    parts: list[str] = [f"{_HEAD}（已压缩 {len(dropped)} 轮）"]
    if entities:
        parts.append("主体：" + "、".join(entities))
    if slots:
        parts.append("问过：" + "、".join(slots))
    if pieces:
        parts.append("结论：" + "、".join(pieces))
    return _truncate(_DIGEST_SEP.join(parts), max_chars)
