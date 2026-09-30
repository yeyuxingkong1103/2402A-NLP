# -*- coding: utf-8 -*-
"""提示词模板模块的单元测试。"""
from app.core import prompts


def test_format_history_renders_user_and_assistant_turns_with_role_name():
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀，有什么可以帮你？"},
        {"role": "user", "content": "今天天气怎么样"},
    ]
    text = prompts.format_history(history, role_name="小助手")

    assert "用户：你好" in text
    assert "小助手：你好呀，有什么可以帮你？" in text
    assert "用户：今天天气怎么样" in text
    # 顺序保持对话顺序
    assert text.index("用户：你好") < text.index("小助手：")
    assert text.index("小助手：") < text.index("用户：今天天气怎么样")


def test_format_history_returns_empty_string_for_empty_history():
    assert prompts.format_history([], role_name="小助手") == ""


def test_build_messages_injects_persona_and_role_name_into_system_prompt():
    messages = prompts.build_messages(
        role_name="林医生",
        persona="一位有耐心的全科医生，喜欢用通俗的语言解释",
        prompt_template=prompts.DEFAULT_PROMPT_TEMPLATE,
        history=[],
        user_input="我头疼",
    )

    system = messages[0]
    assert system["role"] == "system"
    assert "林医生" in system["content"]
    assert "一位有耐心的全科医生" in system["content"]
    assert messages[-1]["role"] == "user"


def test_build_messages_includes_history_and_user_input():
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好"},
    ]
    messages = prompts.build_messages(
        role_name="林医生",
        persona="全科医生",
        prompt_template=prompts.DEFAULT_PROMPT_TEMPLATE,
        history=history,
        user_input="我头疼",
    )

    # 历史应出现在 system 提示中（短期记忆注入）
    assert "用户：你好" in messages[0]["content"]
    assert "林医生：你好" in messages[0]["content"]
    # 用户最新输入单独作为 user 消息
    assert messages[-1] == {"role": "user", "content": "我头疼"}


def test_build_messages_injects_knowledge_block_when_provided():
    messages = prompts.build_messages(
        role_name="林医生",
        persona="全科医生",
        prompt_template=prompts.DEFAULT_PROMPT_TEMPLATE,
        history=[],
        user_input="高血压怎么办",
        knowledge="高血压患者应低盐饮食。",
    )
    system = messages[0]["content"]
    assert "【知识库内容】" in system
    assert "高血压患者应低盐饮食。" in system


def test_knowledge_block_demands_strict_accuracy():
    """准确性优先：严格依据知识库、精确数值、不编造。"""
    block = prompts.format_knowledge_block("知识内容。")
    assert "严格依据" in block
    assert "精确数值" in block
    assert "知识库中没有相关信息" in block


def test_knowledge_block_demands_covering_all_subquestions():
    block = prompts.format_knowledge_block("知识内容。")
    assert "逐个子问题" in block


def test_knowledge_block_contains_few_shot_examples():
    """few-shot：枚举要列全、归类照原文。"""
    block = prompts.format_knowledge_block("知识内容。")
    assert "回答示例" in block
    assert "全部列出" in block
    assert "归类以知识库原文为准" in block


def test_knowledge_block_examples_are_domain_neutral():
    """示例用中性占位词（甲乙丙），不泄漏医学内容到其他角色。"""
    block = prompts.format_knowledge_block("知识内容。")
    assert "甲、乙、丙" in block
    assert "高血压" not in block.replace("知识内容。", "")


def test_knowledge_block_empty_for_blank_knowledge():
    assert prompts.format_knowledge_block("") == ""
    assert prompts.format_knowledge_block("   \n ") == ""


def test_build_messages_omits_knowledge_section_when_empty():
    messages = prompts.build_messages(
        role_name="林医生",
        persona="全科医生",
        prompt_template=prompts.DEFAULT_PROMPT_TEMPLATE,
        history=[],
        user_input="你好",
        knowledge="",
    )
    assert "【知识库内容】" not in messages[0]["content"]


def test_build_messages_ignores_knowledge_for_legacy_template_without_placeholder():
    legacy = "你扮演{role_name}。人设：{persona}。历史：{history}。用户：{user_input}"
    messages = prompts.build_messages(
        role_name="X",
        persona="Y",
        prompt_template=legacy,
        history=[],
        user_input="hi",
        knowledge="不应出现在模板里",
    )
    assert "不应出现在模板里" not in messages[0]["content"]


def test_build_messages_raises_when_template_misses_placeholder():
    bad_template = "你扮演{role_name}，性格：{persona}，没有用户输入占位符"
    try:
        prompts.build_messages(
            role_name="X",
            persona="Y",
            prompt_template=bad_template,
            history=[],
            user_input="hi",
        )
    except ValueError as exc:
        assert "{user_input}" in str(exc)
    else:
        raise AssertionError("缺少占位符的模板应抛出 ValueError")


# ---------- 长期记忆注入 ----------

def test_memory_block_contains_past_conversation_hint():
    block = prompts.format_memory_block(["用户：我养了一只猫\n小阳：真可爱！"])
    assert "【长期记忆】" in block
    assert "我养了一只猫" in block
    assert "过去的对话" in block


def test_memory_block_empty_for_no_memories():
    assert prompts.format_memory_block([]) == ""


def test_build_messages_injects_memories_section():
    messages = prompts.build_messages(
        role_name="小阳", persona="朋友", prompt_template=prompts.DEFAULT_PROMPT_TEMPLATE,
        history=[], user_input="我的猫叫什么？", memories=["用户：我养了一只猫"],
    )
    assert "【长期记忆】" in messages[0]["content"]
    assert "我养了一只猫" in messages[0]["content"]


def test_build_messages_omits_memory_section_when_none():
    messages = prompts.build_messages(
        role_name="小阳", persona="朋友", prompt_template=prompts.DEFAULT_PROMPT_TEMPLATE,
        history=[], user_input="你好",
    )
    assert "【长期记忆】" not in messages[0]["content"]
