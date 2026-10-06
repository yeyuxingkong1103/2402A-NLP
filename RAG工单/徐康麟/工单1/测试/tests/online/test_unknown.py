"""在线测试：不知道 / 兜底处理（工单 9.2 / 5.8 / 验收标准 4）。

测试目标：
1. 统一的兜底文案就是配置里的 ``不清楚``，且任何兜底路径都不得抛异常；
2. 没有索引、没有检索片段时，必须回复"不清楚"，且不携带任何引用；
3. 与语料无关的问题（"今天天气怎么样""红烧肉怎么做"）必须被判为无法回答
   （依据检索器的原始余弦阈值 ``min_confidence_cosine``）；
4. 无论最终是否兜底，答案都必须**有据可依**：引用片段必须真实来自 PDF，
   禁止凭空编造（招股书里的数字不可能来自模型记忆）。

索引缺失时依赖索引的用例会跳过（空索引路径的用例仍然执行）。
"""

from __future__ import annotations

import pytest

from app.core.config import get_settings

# 与《招股说明书1.pdf》完全无关的问题
UNRELATED_QUESTIONS = [
    "今天天气怎么样？",
    "红烧肉怎么做才好吃？",
    "推荐一本好看的小说。",
]


# ==========================================================================
# 1. 兜底文案与配置
# ==========================================================================
def test_unknown_answer_is_configured_text() -> None:
    """全局兜底文案必须是『不清楚』（工单 1.4 / 5.8 硬性要求）。"""
    assert get_settings().app.unknown_answer == "不清楚", (
        f"兜底文案应为『不清楚』，实际为 {get_settings().app.unknown_answer!r}"
    )


def test_empty_index_returns_unknown(empty_engine) -> None:
    """索引为空（未建索引）时必须回复"不清楚"，且不得抛异常。"""
    for question in UNRELATED_QUESTIONS + ["武汉兴图新科电子股份有限公司注册资本是多少？"]:
        answer = empty_engine.ask(question, save=False)
        assert answer.is_unknown is True, f"无索引时『{question}』未走兜底回复：{answer.answer[:60]!r}"
        assert answer.answer == "不清楚", f"兜底文案错误：{answer.answer!r}"
        assert answer.unknown_reason, "兜底回复必须记录原因（禁止静默失败）"


def test_empty_index_unknown_has_no_citations(empty_engine) -> None:
    """兜底回复不得携带任何引用，否则会误导用户以为答案有出处。"""
    answer = empty_engine.ask("红烧肉怎么做？", save=False)
    assert answer.citations == [], f"兜底回复不应带引用，实际：{answer.citations}"
    assert answer.mode == "fallback", f"兜底回复的 mode 应为 fallback，实际 {answer.mode}"


# ==========================================================================
# 2. 无关问题：必须拒答
# ==========================================================================
@pytest.mark.parametrize("question", UNRELATED_QUESTIONS, ids=["天气", "菜谱", "小说"])
def test_unrelated_question_returns_unknown(engine, question, requires_index) -> None:
    """与语料无关的问题必须回复"不清楚"，不得拿招股书里的无关段落搪塞。

    该判定依赖检索器的原始余弦阈值（``min_confidence_cosine``）：
    实测相关问题 ≥0.73、无关问题 ≤0.39，阈值 0.55，区分度充足。
    """
    answer = engine.ask(question, save=False)
    assert answer.is_unknown is True, (
        f"『{question}』与《招股说明书1.pdf》无关，却给出了答案：{answer.answer[:80]!r}"
        f"（命中页码 {answer.pages}）"
    )
    assert answer.answer == "不清楚", f"兜底文案应为『不清楚』，实际 {answer.answer!r}"
    assert answer.citations == [], "兜底回复不得携带引用"


# ==========================================================================
# 3. 兜底之外的答案必须有据可依（不编造）
# ==========================================================================
def test_answer_is_grounded_in_cited_chunk(engine, requires_index) -> None:
    """无论问题是否相关，只要给出了答案，其引用片段必须真实来自 PDF 原文。"""
    for question in UNRELATED_QUESTIONS + ["武汉兴图新科电子股份有限公司注册资本是多少？"]:
        answer = engine.ask(question, save=False)
        if answer.is_unknown:
            assert answer.citations == [], "兜底回复不应携带引用"
            continue

        assert answer.citations, f"『{question}』给出了答案却没有引用，无法追溯"
        citation = answer.citations[0]
        chunk = engine.store.get_chunk(citation.chunk_id)
        assert chunk is not None, f"引用片段 {citation.chunk_id} 在数据库中不存在（疑似编造引用）"
        assert chunk.page == citation.page, (
            f"引用页码 {citation.page} 与片段实际页码 {chunk.page} 不一致"
        )
        # 引用摘要必须能在原文中找到（截断时会添加省略号）
        snippet = citation.snippet.rstrip("…")
        assert snippet[:40] in chunk.content, (
            f"引用摘要并非来自原文片段，疑似编造：摘要={citation.snippet[:40]!r}"
        )


def test_unknown_when_retrieval_returns_nothing(engine, requires_index, monkeypatch) -> None:
    """检索器返回空结果时，必须走兜底回复（不依赖分数阈值的兜底路径）。"""
    monkeypatch.setattr(engine.retriever, "retrieve", lambda *args, **kwargs: [])
    answer = engine.ask("武汉兴图新科电子股份有限公司注册资本是多少？", save=False)
    assert answer.is_unknown is True, "检索结果为空时未走兜底回复"
    assert answer.answer == "不清楚", f"兜底文案错误：{answer.answer!r}"
    assert answer.citations == [], "兜底回复不应携带引用"
