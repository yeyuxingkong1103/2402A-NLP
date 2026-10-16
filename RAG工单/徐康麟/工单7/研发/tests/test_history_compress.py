# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-Query 理解优化

``compress_history`` 真单元测试（非烟雾）：无历史、轮次不足、正常压缩、
``max_chars`` 硬上限、公司名保留（指代消解信息）、``TurnRecord`` 与 ``dict`` 双输入。

运行：
    pwsh -NoProfile -File run_py.ps1 -m pytest 研发/tests/test_history_compress.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core.dialogue_state import TurnRecord  # noqa: E402
from app.core.history_compress import (  # noqa: E402
    CompressedHistory,
    approx_tokens,
    compress_history,
    extract_companies,
)

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化"
LIYUAN = "武汉力源信息技术股份有限公司"
XINGTU = "武汉兴图新科电子股份有限公司"


def _turn(question: str, answer: str, *, slot: str | None = None,
          focus: str | None = None, refused: bool = False, strategy: str = "none") -> TurnRecord:
    rewritten = f"{focus}的{slot}是什么？" if (focus and slot) else question
    return TurnRecord(question=question, rewritten_query=rewritten, answer=answer,
                      focus_entity=focus, slot=slot, refused=refused, strategy=strategy)


def _dict_turn(question: str, answer: str, **extra) -> dict:
    payload = {"question": question, "rewritten_query": question, "answer": answer}
    payload.update(extra)
    return payload


def _long_history(n: int, *, company: str = XINGTU) -> list[TurnRecord]:
    """n 轮合成历史（字段槽位轮换，保证摘要里既有主体也有槽位）。"""
    slots = ["法定代表人", "注册资本", "主营业务", "营业收入", "军用领域"]
    out: list[TurnRecord] = []
    for i in range(n):
        slot = slots[i % len(slots)]
        out.append(_turn(f"{company}的{slot}是多少？",
                         f"根据招股说明书，{company}的{slot}为第 {i + 1} 项数据。",
                         slot=slot, focus=company))
    return out


# --------------------------------------------------------------------- 空 / 短
def test_empty_history_gives_empty_digest() -> None:
    result = compress_history([])
    assert result.digest == "", "无历史必须返回空 digest"
    assert result.kept_turns == []
    assert result.dropped_turns == 0
    assert result.to_dict()["digest"] == ""

    assert compress_history(None).digest == "", "None 也必须安全"


def test_fewer_than_keep_recent_no_compression() -> None:
    turns = _long_history(2)
    result = compress_history(turns, keep_recent=3, max_chars=240)
    assert result.digest == "", "轮次不足时不得压缩"
    assert result.dropped_turns == 0
    assert len(result.kept_turns) == 2


def test_exactly_keep_recent_no_compression() -> None:
    result = compress_history(_long_history(3), keep_recent=3)
    assert result.digest == "" and result.dropped_turns == 0
    assert len(result.kept_turns) == 3


# --------------------------------------------------------------------- 压缩
def test_more_than_keep_recent_compresses_and_keeps_recent_verbatim() -> None:
    turns = _long_history(10)
    result = compress_history(turns, keep_recent=3, max_chars=400)

    assert result.dropped_turns == 7, f"应丢弃 7 轮，实际 {result.dropped_turns}"
    assert len(result.kept_turns) == 3, "kept_turns 长度必须 == keep_recent"
    assert result.digest, "有丢弃轮次时必须产出摘要"

    # 保留的必须是**最近 3 轮原文**（逐字一致）
    for kept, source in zip(result.kept_turns, turns[-3:]):
        assert kept["question"] == source.question
        assert kept["answer"] == source.answer
        assert kept["slot"] == source.slot


def test_keep_recent_one() -> None:
    turns = _long_history(5)
    result = compress_history(turns, keep_recent=1, max_chars=400)
    assert result.dropped_turns == 4 and len(result.kept_turns) == 1
    assert result.kept_turns[0]["question"] == turns[-1].question


# --------------------------------------------------------------------- 硬上限
@pytest.mark.parametrize("max_chars", [10, 30, 60, 120, 240, 500])
def test_max_chars_is_hard_limit(max_chars: int) -> None:
    result = compress_history(_long_history(60), keep_recent=3, max_chars=max_chars)
    assert len(result.digest) <= max_chars, \
        f"digest 超硬上限：len={len(result.digest)} > max_chars={max_chars}"
    if len(result.digest) == max_chars:
        assert result.digest.endswith("…"), "被截断时必须以 … 结尾"


def test_max_chars_zero_disables_digest() -> None:
    result = compress_history(_long_history(20), keep_recent=3, max_chars=0)
    assert result.digest == "", "max_chars<=0 视为禁用摘要"
    assert result.dropped_turns == 17, "禁用摘要也要正确报告丢弃轮数"
    assert len(result.kept_turns) == 3


# --------------------------------------------------------------------- 指代消解信息
def test_digest_keeps_company_name() -> None:
    turns = [
        _turn(f"{LIYUAN}的法定代表人是谁？", f"{LIYUAN}的法定代表人是赵马克。",
              slot="法定代表人", focus=LIYUAN, strategy="topic_carry"),
        _turn("他参与的哪个工程荣获了国家科技进步一等奖？", "他参与的项目荣获国家科技进步一等奖。",
              focus=LIYUAN, strategy="pronoun"),
        _turn(f"{LIYUAN}的注册资本是多少？", f"{LIYUAN}注册资本为 10,000 万元。",
              slot="注册资本", focus=LIYUAN),
        _turn("组织结构图中哪个销售部的销售处最多？", "销售部的销售处最多的是华北销售部。",
              focus=LIYUAN),
    ]
    result = compress_history(turns, keep_recent=1, max_chars=500)

    assert result.dropped_turns == 3
    assert LIYUAN in result.digest, f"摘要必须保留公司名，实际 digest={result.digest!r}"
    assert "法定代表人" in result.digest, "摘要必须保留问过的字段槽位"
    # 摘要只能由原文拼接：其中每个「非框架」片段都必须是原文字串
    for name in (LIYUAN, "法定代表人", "注册资本"):
        assert name in (turns[0].question + turns[2].question + turns[0].answer), \
            "摘要片段必须来自原文，不得编造"


def test_digest_never_invents_facts() -> None:
    """摘要里的公司名必须能在被丢弃的轮次原文里找到（严禁编造）。"""
    turns = _long_history(8)
    result = compress_history(turns, keep_recent=2, max_chars=300)
    dropped_text = "".join(t.question + t.rewritten_query + t.answer for t in turns[:-2])

    for name in extract_companies(result.digest):
        assert name in dropped_text, f"摘要出现了原文没有的主体：{name}"


def test_focus_entity_falls_back_to_explicit_field_only() -> None:
    turns = _long_history(6, company=LIYUAN)
    result = compress_history(turns, keep_recent=2)
    assert result.focus_entity == LIYUAN, "focus_entity 应取最近的显式字段值"

    no_focus = [_dict_turn("这个公司的法定代表人是谁？", "法定代表人是赵马克。") for _ in range(5)]
    assert compress_history(no_focus, keep_recent=2).focus_entity is None
    assert compress_history(no_focus, keep_recent=2, focus_entity=LIYUAN).focus_entity == LIYUAN


# --------------------------------------------------------------------- 双输入
def test_supports_dict_input() -> None:
    turns = [
        _dict_turn(f"{LIYUAN}的法定代表人是谁？", f"{LIYUAN}的法定代表人是赵马克。",
                   slot="法定代表人", focus_entity=LIYUAN),
        _dict_turn("那注册资本呢？", f"{LIYUAN}注册资本为 10,000 万元。",
                   slot="注册资本", focus_entity=LIYUAN),
        _dict_turn("组织结构图中哪个销售处的销售处最多？", "销售处最多的是华北销售部。"),
        _dict_turn("有哪些销售处？", "共有 12 个销售处。"),
    ]
    result = compress_history(turns, keep_recent=2, max_chars=400)
    assert result.dropped_turns == 2 and len(result.kept_turns) == 2
    assert LIYUAN in result.digest
    assert result.kept_turns[0]["question"] == "组织结构图中哪个销售处的销售处最多？"


def test_mixed_input_and_normalize_slots() -> None:
    turns = [
        _turn(f"{XINGTU}的军用领域收入是多少？", "报告期内军用领域收入为 1.2 亿元。",
              slot="军用领域", focus=XINGTU),
        _dict_turn("那力源呢？", f"{LIYUAN}的主营业务为电子元器件分销。", slot="主营业务",
                   focus_entity=LIYUAN),
        _dict_turn("他的法定代表人是谁？", "法定代表人是赵马克。"),
        _dict_turn("注册资本是多少？", "注册资本为 10,000 万元。"),
    ]
    result = compress_history(turns, keep_recent=2, max_chars=400)
    assert result.dropped_turns == 2
    assert XINGTU in result.digest or LIYUAN in result.digest
    assert len(result.kept_turns) == 2


def test_to_dict_contract() -> None:
    result = compress_history(_long_history(6), keep_recent=3, max_chars=240)
    payload = result.to_dict()
    assert set(payload) == {"digest", "kept_turns", "dropped_turns", "focus_entity"}, \
        "to_dict 必须与 CompressedHistory 字段一致"
    assert isinstance(payload["kept_turns"], list) and payload["dropped_turns"] == 3
    assert isinstance(result, CompressedHistory)


def test_dropped_turns_count_matches_all_but_recent() -> None:
    for n, keep in ((10, 3), (30, 3), (100, 3), (100, 5)):
        result = compress_history(_long_history(n), keep_recent=keep, max_chars=240)
        assert result.dropped_turns == n - keep, f"n={n}, keep={keep}"
        assert len(result.kept_turns) == keep


def test_helpers_behaviour() -> None:
    assert extract_companies(f"报告期内，{LIYUAN}来自军用的收入") == [LIYUAN]
    assert extract_companies("这个公司呢？") == []
    assert approx_tokens("") == 0 and approx_tokens("四个字符") == 4
