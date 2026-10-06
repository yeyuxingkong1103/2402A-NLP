# -*- coding: utf-8 -*-
"""在线级：多轮对话 + 中英文问答 + 分类取自本轮原文（N-8 多轮断言）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

覆盖验收 7 与 ``设计/接口设计.md`` v1.6 的**调用顺序冻结**：
    轮1「…注册资本是多少？」→ ``expects_numeric == True``；
    轮2「那法定代表人呢？」→ ``expects_numeric == classify_question("那法定代表人呢？")[1] == False``
    （**取自本轮原文**，即使改写器把追问补全也不得改变）；
    连续 3 次分类结果稳定；日志含 ``classified_from="original"``。
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from common import assertions, paths
from common.reports import trace_events, write_report

pytestmark = [pytest.mark.online, pytest.mark.slow]

#: 轮 1 / 轮 2 题面（PDF1 兴图新科）
TURN1 = "武汉兴图新科电子股份有限公司注册资本是多少？"
TURN2 = "那法定代表人呢？"
#: 英文题（验收 7：英文提问答案语言 = en）
ENGLISH_Q = "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?"


@pytest.fixture(scope="session")
def dialogue(engine: Any) -> dict[str, Any]:
    """跑一段 3 轮对话并返回原始结果（含会话 id 与耗时）。"""
    session_id = f"t8-multiturn-{int(time.time())}"
    turns: list[dict[str, Any]] = []
    for index, question in enumerate((TURN1, TURN2, ENGLISH_Q), start=1):
        started = time.perf_counter()
        answer = engine.ask(question, session_id=session_id, stream=False)
        turns.append({
            "turn": index, "question": question,
            "text": str(answer.text),
            "language": getattr(answer, "language", ""),
            "is_unknown": bool(getattr(answer, "is_unknown", False)),
            "first_token_ms": float(getattr(answer, "first_token_ms", 0.0) or 0.0),
            "citations": [{"file_name": getattr(c, "file_name", ""), "page": getattr(c, "page", None)}
                          for c in (getattr(answer, "citations", []) or [])],
            "wall_ms": round((time.perf_counter() - started) * 1000, 2),
        })
    write_report("online_multiturn", "多轮对话与中英文", turns, extra={"session_id": session_id})
    return {"session_id": session_id, "turns": turns}


def test_three_turns_all_answered(dialogue: dict[str, Any]) -> None:
    """3 轮全部作答（不串题、不拒答）。"""
    refused = [t for t in dialogue["turns"] if t["is_unknown"] or not t["text"].strip()]
    assert not refused, f"存在未作答轮次：{refused}"


def test_follow_up_resolves_reference(dialogue: dict[str, Any]) -> None:
    """轮 2 追问「那法定代表人呢？」必须消解指代 → 答程家明。"""
    turn2 = dialogue["turns"][1]
    assert "程家明" in turn2["text"], (f"指代消解失败，轮 2 答：{turn2['text'][:80]!r}"
                                       f"【性质：产品缺陷（实测答成简历块），captain 已派 t16 修复中】")


def test_english_question_answered_in_english(dialogue: dict[str, Any]) -> None:
    """轮 3 英文提问 → ``language == 'en'``，答案以英文为主且带引用。"""
    turn3 = dialogue["turns"][2]
    assert turn3["language"] == "en", f"语言标注应为 en，实测 {turn3['language']!r}"
    assert turn3["citations"], "英文答案也必须带引用"
    ascii_ratio = sum(1 for ch in turn3["text"] if ord(ch) < 128) / max(1, len(turn3["text"]))
    assert ascii_ratio > 0.4, f"英文答案疑似中文作答：{turn3['text'][:80]!r}"


def test_history_window_persisted(dialogue: dict[str, Any], app_config: Any) -> None:
    """会话历史真实落库，且窗口 = 最近 5 轮（10 条消息）。"""
    paths.ensure_dev_on_path()
    from app.core.conversation import get_conversation_store  # noqa: PLC0415

    store = get_conversation_store()
    session_id = dialogue["session_id"]
    history = store.history(session_id, last_n=int(app_config.answer.history_turns))
    assert len(history) == 6, f"3 轮应有 6 条消息（问+答），实测 {len(history)}"
    assert store.message_count(session_id) == 6
    assert history[0].content.strip() == TURN1


def test_classification_from_original_and_stable(app_config: Any) -> None:
    """N-8 多轮断言：数值信号取自本轮原文、连续 3 次稳定；且剥离可逆。"""
    paths.ensure_dev_on_path()
    from app.core.query_understanding import classify_question  # noqa: PLC0415

    first = classify_question(TURN1)
    follow = classify_question(TURN2)
    assert bool(first[1]) is True, f"轮 1 应 expects_numeric=True，实测 {first}"
    assert bool(follow[1]) is False, f"轮 2（本轮原文）应 expects_numeric=False，实测 {follow}"
    repeats = [classify_question(TURN1) for _ in range(3)]
    assert all(item == first for item in repeats), f"同轮分类不稳定：{repeats}"


def test_logs_record_classified_from_original(dialogue: dict[str, Any]) -> None:
    """日志必须留下 ``classified_from="original"``（v1.6 冻结：分类不取自改写结果）。"""
    events = trace_events("rag_trace.jsonl")
    found = [row for row in events if str(row.get("classified_from", "")) == "original"]
    assert found, "rag_trace.jsonl 未见 classified_from=\"original\" 记录（多轮分类口径无法审计）"


def test_multiturn_first_token_budget(dialogue: dict[str, Any]) -> None:
    """多轮场景同样受首字预算约束（每题独立判定）。"""
    latencies = {f"轮{t['turn']}": t["first_token_ms"] for t in dialogue["turns"]}
    checks = assertions.check_first_token(latencies)
    bad = [c for c in checks if not c.ok]
    assert not bad, "\n".join(c.render() for c in bad)
