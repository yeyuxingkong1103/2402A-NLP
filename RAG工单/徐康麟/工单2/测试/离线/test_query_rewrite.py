"""T3 离线测试 ⑤：多轮对话与查询改写。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
覆盖（设计/验收标准.md 验收 5、工单 6.3/6.4）：

- 含指代追问的多轮对话，改写必须把**省略问题**变成**可独立检索**的问题；
- 追问被识别（``is_followup``）且改写结果不同于原问题、且含主体/字段线索；
- 检索查询变体（``search_queries``）非空且含改写后的独立问题；
- **无 LLM 时优雅降级**：``use_llm=False`` 仍能改写与检索（不抛异常、不返回空）。
"""

from __future__ import annotations

import pytest


def test_rewrite_followup_into_standalone_question(query_understanding):
    """指代追问必须被改写为可独立检索的问题（验收 5 的核心）。"""
    history = [
        ("武汉兴图新科电子股份有限公司法定代表人是谁？", "法定代表人是程家明。"),
    ]
    followup = "那它的注册资本呢？"
    rewritten, is_followup = query_understanding.rewrite_with_history(followup, history)
    print(f"\n[改写] 原问句={followup!r} → 改写={rewritten!r}（is_followup={is_followup}）")
    assert rewritten != followup, "追问未被改写，仍是原省略句"
    assert rewritten.strip(), "改写结果为空"
    assert "注册资本" in rewritten, f"改写未带上字段线索「注册资本」: {rewritten!r}"
    assert any(tok in rewritten for tok in ("公司", "兴图", "武汉")), (
        f"改写未补全主体（公司名），无法独立检索: {rewritten!r}"
    )
    assert is_followup is True, "指代追问未被标记为 is_followup"


def test_analyze_marks_followup_and_standalone(query_understanding):
    """``analyze`` 必须区分追问与独立问题，并给出可检索的 search_query。"""
    history = [("武汉兴图新科电子股份有限公司法定代表人是谁？", "法定代表人是程家明。")]
    a_follow = query_understanding.analyze("那它的注册资本呢？", history)
    a_standalone = query_understanding.analyze("武汉兴图新科电子股份有限公司注册资本是多少？", [])
    print(f"\n[分析] 追问: is_followup={a_follow.is_followup} rewritten={a_follow.rewritten!r} "
          f"search_query={a_follow.search_query!r}")
    print(f"[分析] 独立问题: is_followup={a_standalone.is_followup} language={a_standalone.language}")
    assert a_follow.is_followup is True, "含「它」的追问未被识别为 is_followup"
    assert a_standalone.is_followup is False, "独立问题被误判为追问"
    assert a_follow.search_query.strip(), "追问的 search_query 为空"
    assert "注册资本" in a_follow.search_query, (
        f"追问的 search_query 未含字段线索: {a_follow.search_query!r}"
    )


def test_search_queries_variants_non_empty_and_deduped(query_understanding):
    """检索变体必须非空、去重、且含改写后的独立问题（多查询融合的前提）。"""
    history = [("武汉兴图新科电子股份有限公司法定代表人是谁？", "法定代表人是程家明。")]
    analysis = query_understanding.analyze("那它的注册资本呢？", history)
    variants = query_understanding.search_queries(analysis)
    texts = [q for q, _ in variants]
    print(f"\n[变体] {len(variants)} 条: {variants}")
    assert variants, "检索变体为空"
    assert len(texts) == len(set(texts)), f"检索变体存在重复: {texts}"
    assert any("注册资本" in t for t in texts), f"变体未含字段线索: {texts}"
    for _, kind in variants:
        assert isinstance(kind, str) and kind, f"变体类型缺失: {variants}"


def test_no_llm_graceful_degradation():
    """无 LLM 时必须优雅降级：仍能改写/分析，不抛异常、不返回空（工单 6.7）。"""
    from app.core.query_understanding import QueryUnderstanding

    qu_off = QueryUnderstanding(use_llm=False)
    history = [("武汉兴图新科电子股份有限公司法定代表人是谁？", "法定代表人是程家明。")]
    rewritten, is_followup = qu_off.rewrite_with_history("那它的注册资本呢？", history)
    analysis = qu_off.analyze("那它的注册资本呢？", history)
    assert rewritten and rewritten.strip(), "无 LLM 时改写结果为空（未优雅降级）"
    assert analysis.search_query.strip(), "无 LLM 时 search_query 为空"
    assert qu_off.search_queries(analysis), "无 LLM 时检索变体为空"
    print(f"\n[降级] use_llm=False 改写={rewritten!r} search_query={analysis.search_query!r}")


def test_intent_and_page_filter(query_understanding):
    """意图识别与页码过滤必须可用（工单 6.3 Query 理解）。"""
    a = query_understanding.analyze("武汉兴图新科电子股份有限公司注册资本是多少？", [])
    assert a.intent, "意图为空"
    assert a.keywords, "关键词为空"
    print(f"\n[意图] {a.intent} 关键词={a.keywords}")
    # 显式页码过滤
    b = query_understanding.analyze("第 129 页军用领域收入是多少？", [])
    assert b.page_filter == [129] or 129 in b.page_filter, (
        f"显式页码未被识别: page_filter={b.page_filter}"
    )


def test_history_window_is_bounded(query_understanding):
    """多轮历史的改写必须只看最近若干轮，不能无限膨胀（验收 5：最近 5 轮）。"""
    history = [(f"武汉兴图新科电子股份有限公司问题{i}？", f"答案{i}。") for i in range(12)]
    rewritten, _ = query_understanding.rewrite_with_history("那它的注册资本呢？", history)
    assert rewritten.strip(), "长历史下改写为空"
    print(f"\n[历史窗口] 12 轮历史下改写={rewritten!r}")
