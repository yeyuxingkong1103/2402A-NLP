"""guard.resolve_citation_outcome 统一处置阶梯的回归测试（批次 32）。

抽取背景：引用失败处置阶梯曾有一份逐字平行的实现分别在
service.chat（同步）与 chat_stream（流式），批次 28 漏改 streaming 一处导致
SSE 主路径行为不变。本文件锁两件事：
1. 对同一异常，统一函数给出确定的结果（回答处置 + 事件名逐字）；
2. 结构上"一处实现、两处调用"——事件名字符串只允许出现在 guard.py，
   service.py / streaming.py 必须各含恰好一次 resolve_citation_outcome 调用
   （AST 扫描，防止以后有人再把阶梯复制回调用方）。
"""

import ast
from pathlib import Path

import pytest

from app.chat.citation_check import (
    CitationError,
    NoCitationError,
    UncitedConclusionError,
    UnknownLawCitationError,
)
from app.chat.guard import (
    CITATION_FALLBACK_NOTICE,
    NO_CITATION_WARNING,
    UNCITED_CONCLUSION_WARNING,
    UNKNOWN_LAW_WARNING,
    resolve_citation_outcome,
)

CHAT_DIR = Path(__file__).resolve().parents[1] / "app" / "chat"

# 事件名前缀（与 guard.py 内取值逐字一致）
EVENT_UNKNOWN_LAW = "citation_warning_unknown_law"
EVENT_NO_CITATION = "citation_warning_no_citation"
EVENT_UNCITED_CONCLUSION = "citation_warning_uncited_conclusion"
EVENT_CHECK_FAILED = "citation_check_failed"


def test_unknown_law_keeps_answer_and_event():
    answer, event = resolve_citation_outcome("主体回答", UnknownLawCitationError("提了清单外法规"))
    assert answer.startswith("主体回答") and UNKNOWN_LAW_WARNING in answer
    assert event == f"{EVENT_UNKNOWN_LAW}: 提了清单外法规"


def test_no_citation_keeps_answer_and_event():
    answer, event = resolve_citation_outcome("主体回答", NoCitationError("零引用"))
    assert answer.startswith("主体回答") and NO_CITATION_WARNING in answer
    assert event == f"{EVENT_NO_CITATION}: 零引用"


def test_uncited_conclusion_keeps_answer_and_event():
    answer, event = resolve_citation_outcome("主体回答", UncitedConclusionError("漏了一句"))
    assert answer.startswith("主体回答") and UNCITED_CONCLUSION_WARNING in answer
    assert event == f"{EVENT_UNCITED_CONCLUSION}: 漏了一句"


def test_hard_violation_replaces_answer():
    answer, event = resolve_citation_outcome("主体回答", CitationError("引用编号 [6] 越界"))
    assert answer == CITATION_FALLBACK_NOTICE
    assert event == f"{EVENT_CHECK_FAILED}: 引用编号 [6] 越界"


def test_warning_appended_only_once():
    """重复处置（理论上的双路径重复调用）不得把警示行追加两遍。"""
    once, _ = resolve_citation_outcome("主体回答", NoCitationError("零引用"))
    twice, _ = resolve_citation_outcome(once, NoCitationError("零引用"))
    assert once == twice


@pytest.mark.parametrize(
    "error",
    [
        UnknownLawCitationError("清单外法规 X"),
        NoCitationError("零引用"),
        UncitedConclusionError("漏了一句"),
        CitationError("越界"),
    ],
)
def test_same_exception_same_outcome(error):
    """同一异常重复调用，结果必须完全一致（两条路径共用时对拍的基础）。"""
    assert resolve_citation_outcome("主体回答", error) == resolve_citation_outcome("主体回答", error)


def _source_of(name: str) -> str:
    return (CHAT_DIR / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("file_name", ["service.py", "streaming.py"])
def test_callers_delegate_to_shared_function(file_name):
    """调用方必须经 resolve_citation_outcome 处置，且不得内联事件名字符串。"""
    source = _source_of(file_name)
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_citation_outcome"
    ]
    assert len(calls) == 1, f"{file_name} 应恰好调用一次 resolve_citation_outcome，实际 {len(calls)} 次"
    for prefix in (EVENT_UNKNOWN_LAW, EVENT_NO_CITATION, EVENT_UNCITED_CONCLUSION, EVENT_CHECK_FAILED):
        assert prefix not in source, f"{file_name} 不应再内联事件名 {prefix}（唯一出处是 guard.py）"


def test_event_names_defined_only_in_guard():
    """事件名字符串的唯一出处是 guard.py（service/streaming 的副本已删）。"""
    guard_source = _source_of("guard.py")
    for prefix in (EVENT_UNKNOWN_LAW, EVENT_NO_CITATION, EVENT_UNCITED_CONCLUSION, EVENT_CHECK_FAILED):
        assert prefix in guard_source
