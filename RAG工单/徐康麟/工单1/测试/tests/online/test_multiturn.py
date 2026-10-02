"""在线测试：多轮对话（工单 9.2 / 5.7 / 验收标准 6）。

测试目标：
1. 同一 ``conversation_id`` 连续追问时，历史必须按顺序落库（user/assistant 交替）；
2. 省略式追问（"那注册资本呢？""那法定代表人呢？"）必须被识别为追问，
   并结合上一轮问题还原成一个完整问句（实体替换或拼接），且**意图切换到新议题**；
3. 无关问题（"今天天气怎么样？"）不得被误判成追问；
4. 历史窗口受 ``conversation.max_history_rounds`` 限制（工单要求保留最近 5 轮）；
5. 不同会话之间必须完全隔离；清空会话后历史必须真正消失；
6. 会话标题取首轮问题（并在后续轮次保持稳定）。

索引缺失时整个模块跳过：
``索引未就绪：请先运行 python scripts/build_index.py``。
"""

from __future__ import annotations

import pytest

QUESTION_LEGAL = "武汉兴图新科电子股份有限公司法定代表人是谁？"
QUESTION_CAPITAL = "武汉兴图新科电子股份有限公司注册资本是多少？"
# 省略式追问：换到新议题「注册资本」，改写后应是一个完整问句
QUESTION_FOLLOWUP = "那注册资本呢？"
# 与语料无关的问题：即使上一轮问的是公司信息，也不能被改写成追问
UNRELATED_QUESTION = "今天天气怎么样？"


# --------------------------------------------------------------------------
# 已知缺陷说明（对应用例已用 @pytest.mark.xfail(strict=False) 标记）：
# 1) auto_title 通过 INSERT OR REPLACE 重写会话行 → message_count 被重置为 0；
# 2) 由 1) 导致 auto_title 的"超过 2 条不再改名"守卫失效 → 标题被每一轮问题覆盖。
# 两处都真实执行、真实断言，缺陷修复后会自动变成 XPASS。
# --------------------------------------------------------------------------


# ==========================================================================
# 1. 历史落库
# ==========================================================================
def test_history_is_recorded_in_order(engine, requires_index) -> None:
    """同一会话的两轮问答必须按 user/assistant 交替顺序落库。"""
    conversation_id = engine.new_conversation(title="多轮-落库")

    engine.ask(QUESTION_LEGAL, conversation_id=conversation_id)
    engine.ask(QUESTION_FOLLOWUP, conversation_id=conversation_id)

    messages = engine.get_messages(conversation_id)
    assert len(messages) == 4, f"两轮问答应产生 4 条消息，实际 {len(messages)}"
    assert [message.role for message in messages] == ["user", "assistant", "user", "assistant"], (
        f"消息角色顺序错误：{[message.role for message in messages]}"
    )
    assert messages[0].content == QUESTION_LEGAL, "首轮用户问题落库不正确"
    assert messages[2].content == QUESTION_FOLLOWUP, "追问内容落库不正确"

    history = engine.conversations.history_pairs(conversation_id)
    assert [role for role, _ in history] == ["user", "assistant", "user", "assistant"], (
        "history_pairs 返回的历史角色顺序错误"
    )


def test_history_window_is_limited(engine, settings, requires_index) -> None:
    """历史窗口不得超过配置的轮数（默认最近 5 轮 = 10 条消息）。"""
    conversation_id = engine.new_conversation(title="多轮-窗口上限")
    rounds = settings.conversation.max_history_rounds + 2
    for index in range(rounds):
        engine.ask(f"第 {index} 轮：公司注册资本是多少？", conversation_id=conversation_id)

    history = engine.conversations.history_pairs(conversation_id)
    assert len(history) == settings.conversation.max_history_rounds * 2, (
        f"历史窗口应为 {settings.conversation.max_history_rounds * 2} 条，实际 {len(history)} 条"
    )

    all_messages = engine.get_messages(conversation_id)
    assert len(all_messages) == rounds * 2, f"库中应保存全部 {rounds * 2} 条消息，实际 {len(all_messages)}"


# ==========================================================================
# 2. 追问改写
# ==========================================================================
def test_followup_is_rewritten_with_previous_question(engine, requires_index) -> None:
    """省略式追问必须被识别为追问，并还原成"主题+公司主体"的完整问句。"""
    conversation_id = engine.new_conversation(title="多轮-改写")
    engine.ask(QUESTION_LEGAL, conversation_id=conversation_id)
    answer = engine.ask(QUESTION_FOLLOWUP, conversation_id=conversation_id)

    analysis = answer.query_analysis
    assert analysis is not None, "追问没有返回 Query 理解结果"
    assert analysis.is_followup is True, f"『{QUESTION_FOLLOWUP}』未被识别为追问"
    assert analysis.original == QUESTION_FOLLOWUP, "改写后仍应保留原始问题"
    assert analysis.rewritten != analysis.original, "追问未被改写，检索会丢失上下文"
    assert "注册资本" in analysis.rewritten, (
        f"改写后的问句未包含追问的新议题『注册资本』：{analysis.rewritten!r}"
    )
    assert "公司" in analysis.rewritten, (
        f"改写后的问句未从历史中带回公司主体：{analysis.rewritten!r}"
    )
    assert analysis.intent == "注册资本", (
        f"改写后意图应切换到新议题『注册资本』，实际 {analysis.intent!r}（{analysis.rewritten!r}）"
    )


def test_followup_answers_the_new_topic(engine, golden_qa, requires_index) -> None:
    """追问换主题（法定代表人 -> 注册资本）时，答案必须切换到新主题。"""
    golden_capital = next(item for item in golden_qa if item.id == 543)

    conversation_id = engine.new_conversation(title="多轮-换主题")
    first = engine.ask(QUESTION_LEGAL, conversation_id=conversation_id)
    second = engine.ask(QUESTION_FOLLOWUP, conversation_id=conversation_id)

    assert first.answer != second.answer, "追问换了主题，答案却与上一轮完全相同，上下文改写未生效"
    assert second.is_unknown is False, f"追问被判为不清楚：{second.unknown_reason!r}"
    assert "5,520" in second.answer or "5520" in second.answer, (
        f"追问未回答注册资本（应含 5,520 万元），实际答案：{second.answer[:80]!r}，"
        f"标准答案：{golden_capital.answer!r}"
    )


def test_reverse_direction_followup(engine, golden_qa, requires_index) -> None:
    """反向追问（注册资本 -> 法定代表人）同样必须换到新议题。"""
    golden_legal = next(item for item in golden_qa if item.id == 531)

    conversation_id = engine.new_conversation(title="多轮-反向")
    engine.ask(QUESTION_CAPITAL, conversation_id=conversation_id)
    answer = engine.ask("那法定代表人呢？", conversation_id=conversation_id)

    assert answer.query_analysis.is_followup is True, "『那法定代表人呢？』未被识别为追问"
    assert answer.query_analysis.intent == "法定代表人", (
        f"意图应切换到法定代表人，实际 {answer.query_analysis.intent!r}"
    )
    assert "程家明" in answer.answer, (
        f"追问未回答法定代表人（应含『程家明』），实际：{answer.answer[:80]!r}，"
        f"标准答案：{golden_legal.answer!r}"
    )


def test_unrelated_question_is_not_treated_as_followup(engine, requires_index) -> None:
    """无关问题不得被误判成追问，否则会被上一轮主题污染并绕过『不清楚』兜底。"""
    conversation_id = engine.new_conversation(title="多轮-无关问题")
    engine.ask(QUESTION_CAPITAL, conversation_id=conversation_id)
    answer = engine.ask(UNRELATED_QUESTION, conversation_id=conversation_id)

    analysis = answer.query_analysis
    assert analysis is not None, "缺少 Query 理解结果"
    assert analysis.is_followup is False, f"『{UNRELATED_QUESTION}』被误判为追问"
    assert analysis.rewritten == analysis.original, "无关问题不应被改写"


def test_first_round_question_is_not_rewritten(engine, requires_index) -> None:
    """首轮提问（无历史）不得被改写，避免引入噪声。"""
    answer = engine.ask(QUESTION_CAPITAL, save=False)
    analysis = answer.query_analysis
    assert analysis is not None, "首轮问答缺少 Query 理解结果"
    assert analysis.is_followup is False, "首轮提问被误判为追问"
    assert analysis.rewritten == analysis.original, "首轮提问不应被改写"


# ==========================================================================
# 3. 会话隔离与清空
# ==========================================================================
def test_conversations_are_isolated(engine, requires_index) -> None:
    """两个会话的消息必须互不可见。"""
    first = engine.new_conversation(title="多轮-A")
    second = engine.new_conversation(title="多轮-B")

    engine.ask(QUESTION_CAPITAL, conversation_id=first)
    engine.ask(QUESTION_LEGAL, conversation_id=second)

    first_messages = engine.get_messages(first)
    second_messages = engine.get_messages(second)
    assert len(first_messages) == 2 and len(second_messages) == 2, "会话消息数不正确"
    assert first_messages[0].content == QUESTION_CAPITAL, "会话 A 的消息串到了别处"
    assert second_messages[0].content == QUESTION_LEGAL, "会话 B 的消息串到了别处"


def test_clear_conversation_removes_history(engine, requires_index) -> None:
    """清空会话后，历史必须为空且追问改写不再生效。"""
    conversation_id = engine.new_conversation(title="多轮-清空")
    engine.ask(QUESTION_LEGAL, conversation_id=conversation_id)
    assert engine.conversations.history_pairs(conversation_id), "清空前应存在历史"

    removed = engine.clear_conversation(conversation_id)
    assert removed == 2, f"应删除 2 条消息，实际 {removed}"
    assert engine.get_messages(conversation_id) == [], "清空后仍能读到消息"
    assert engine.conversations.history_pairs(conversation_id) == [], "清空后历史窗口应为空"

    answer = engine.ask(QUESTION_FOLLOWUP, conversation_id=conversation_id)
    assert answer.query_analysis.is_followup is False, "清空历史后仍把追问当作有上下文的追问"


# ==========================================================================
# 4. 会话标题与计数
# ==========================================================================
def test_title_is_taken_from_first_question(engine, requires_index) -> None:
    """会话标题应取首轮问题的前 20 字，便于用户在会话列表中辨识。"""
    conversation_id = engine.new_conversation(title="新对话")
    engine.ask(QUESTION_CAPITAL, conversation_id=conversation_id)

    conversation = engine.store.get_conversation(conversation_id)
    assert conversation is not None, "会话不存在"
    assert conversation.title == QUESTION_CAPITAL[:20], (
        f"会话标题应为首轮问题前 20 字 {QUESTION_CAPITAL[:20]!r}，实际 {conversation.title!r}"
    )


@pytest.mark.xfail(
    reason=(
        "已知缺陷：auto_title 通过 INSERT OR REPLACE 重写会话行，把 message_count 重置为 0"
        "（app/core/conversation.py auto_title + app/storage/sqlite_manager.py create_conversation）"
    ),
    strict=False,
)
def test_message_count_matches_stored_messages(engine, requires_index) -> None:
    """会话的 message_count 必须与实际消息条数一致（界面依赖该字段）。"""
    conversation_id = engine.new_conversation(title="多轮-计数")
    engine.ask(QUESTION_CAPITAL, conversation_id=conversation_id)

    stored = len(engine.get_messages(conversation_id))
    counted = engine.store.get_conversation(conversation_id).message_count
    assert counted == stored, (
        f"会话 message_count={counted}，实际消息数={stored}，计数被重置（界面会显示 0 条消息）"
    )


@pytest.mark.xfail(
    reason=(
        "已知缺陷：message_count 被重置为 0，使 auto_title 的『超过 2 条不再改名』守卫失效，"
        "标题被每一轮问题覆盖"
    ),
    strict=False,
)
def test_title_is_not_overwritten_by_later_rounds(engine, requires_index) -> None:
    """会话标题应保持首轮问题，不应被后续追问覆盖。"""
    conversation_id = engine.new_conversation(title="新对话")
    engine.ask(QUESTION_CAPITAL, conversation_id=conversation_id)
    engine.ask(QUESTION_LEGAL, conversation_id=conversation_id)

    conversation = engine.store.get_conversation(conversation_id)
    assert conversation.title == QUESTION_CAPITAL[:20], (
        f"标题被后续轮次覆盖为 {conversation.title!r}（应保持首轮问题 {QUESTION_CAPITAL[:20]!r}）"
    )


# ==========================================================================
# 5. 会话列表与切换
# ==========================================================================
def test_list_and_switch_conversations(engine, requires_index) -> None:
    """会话列表应包含新建会话；切换到不存在的会话必须报错。"""
    conversation_id = engine.new_conversation(title="多轮-列表")
    engine.ask(QUESTION_CAPITAL, conversation_id=conversation_id)

    listed = {item.conversation_id for item in engine.list_conversations()}
    assert conversation_id in listed, f"新建会话未出现在会话列表中：{listed}"

    engine.switch_conversation(conversation_id)
    with pytest.raises(KeyError):
        engine.switch_conversation("conv_不存在")
