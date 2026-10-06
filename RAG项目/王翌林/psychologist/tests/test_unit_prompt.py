"""单元测试：src.rag.prompt 提示词组装（三角色话术、知识片段、记忆、危机指令）。"""
import pytest

from src.core.config import settings
from src.rag import prompt as prompt_utils
from src.services.persona_seed import PERSONAS

# 角色 -> system prompt 中应出现的关键话术关键词
PERSONA_KEYWORDS = {
    "humanistic_lin": "我听到你",
    "cbt_chen": "我们一起看看这个想法",
    "mindfulness_zhou": "先做三次深呼吸",
}

PERSONA_BY_CODE = {p["persona_code"]: dict(p) for p in PERSONAS}

CONTEXT = "[片段1]（来源：情绪管理手册.pdf，相关度：0.92）\n焦虑时可以用呼吸放松帮助自己平静下来。"
HISTORY = [{"role": "user", "content": "我最近总失眠"},
           {"role": "assistant", "content": "我听到你很累，我们先慢慢说"}]
MEMORIES = [{"summary": "用户长期存在入睡困难，曾尝试睡前冥想"}]
QUESTION = "我晚上躺下就开始胡思乱想，怎么办？"


def _system(prompt):
    assert prompt[0]["role"] == "system"
    return prompt[0]["content"]


# ---------------- 三个角色 system prompt ----------------
@pytest.mark.parametrize("code", list(PERSONA_KEYWORDS))
def test_each_persona_keeps_its_own_tone(code):
    persona = PERSONA_BY_CODE[code]
    system = _system(prompt_utils.build_chat_prompt(
        persona=persona, context=CONTEXT, recent_messages=HISTORY, question=QUESTION))
    assert PERSONA_KEYWORDS[code] in system
    assert persona["name"] in system


def test_persona_system_prompts_are_distinct():
    prompts = {_system(prompt_utils.build_chat_prompt(
        persona=dict(p), context=CONTEXT, recent_messages=HISTORY, question=QUESTION))
        for p in PERSONAS}
    assert len(prompts) == 3


def test_safety_boundary_present_in_prompt():
    for persona in PERSONAS:
        system = _system(prompt_utils.build_chat_prompt(
            persona=dict(persona), context=CONTEXT, recent_messages=HISTORY, question=QUESTION))
        assert "不进行医学诊断" in system
        assert "12356" in system


# ---------------- 知识片段 / 记忆 / 问题拼接 ----------------
def test_context_is_injected():
    system = _system(prompt_utils.build_chat_prompt(
        persona=dict(PERSONA_BY_CODE["cbt_chen"]), context=CONTEXT,
        recent_messages=HISTORY, question=QUESTION))
    assert "焦虑时可以用呼吸放松帮助自己平静下来" in system
    assert "情绪管理手册.pdf" in system


def test_empty_hits_use_no_knowledge_hint():
    hits = []
    context = prompt_utils.build_context_block(hits)
    assert context == prompt_utils.NO_KNOWLEDGE_HINT
    system = _system(prompt_utils.build_chat_prompt(
        persona=dict(PERSONA_BY_CODE["humanistic_lin"]), context=context,
        recent_messages=[], question=QUESTION))
    assert prompt_utils.NO_KNOWLEDGE_HINT in system
    assert "（无历史对话）" in system
    assert "（无长期记忆）" in system


def test_short_term_history_is_injected():
    system = _system(prompt_utils.build_chat_prompt(
        persona=dict(PERSONA_BY_CODE["humanistic_lin"]), context=CONTEXT,
        recent_messages=HISTORY, question=QUESTION))
    assert "用户：我最近总失眠" in system
    assert "咨询师：我听到你很累，我们先慢慢说" in system


def test_long_term_memory_is_injected():
    system = _system(prompt_utils.build_chat_prompt(
        persona=dict(PERSONA_BY_CODE["mindfulness_zhou"]), context=CONTEXT,
        recent_messages=HISTORY, question=QUESTION, long_term_memory=MEMORIES))
    assert "用户长期存在入睡困难，曾尝试睡前冥想" in system


def test_question_goes_into_system_and_user_message():
    prompt = prompt_utils.build_chat_prompt(
        persona=dict(PERSONA_BY_CODE["cbt_chen"]), context=CONTEXT,
        recent_messages=HISTORY, question=QUESTION)
    assert len(prompt) == 2
    assert prompt[1]["role"] == "user"
    assert prompt[1]["content"] == QUESTION
    assert QUESTION in prompt[0]["content"]


# ---------------- 危机指令 ----------------
def test_crisis_injection_when_crisis_true():
    system = _system(prompt_utils.build_chat_prompt(
        persona=dict(PERSONA_BY_CODE["humanistic_lin"]), context=CONTEXT,
        recent_messages=HISTORY, question="我不想活了", crisis=True))
    assert "【危机干预优先指令】" in system
    assert "120" in system and "110" in system
    assert settings.crisis_hotline in system


def test_no_crisis_injection_when_crisis_false():
    system = _system(prompt_utils.build_chat_prompt(
        persona=dict(PERSONA_BY_CODE["humanistic_lin"]), context=CONTEXT,
        recent_messages=HISTORY, question=QUESTION, crisis=False))
    assert "【危机干预优先指令】" not in system


# ---------------- 模板分支 / 辅助函数 ----------------
def test_persona_with_placeholder_template_is_filled():
    persona = {
        "name": "测试医生", "therapy_type": "心理陪伴", "style": "温和", "methods": "倾听",
        "system_prompt": prompt_utils.SYSTEM_PROMPT_TEMPLATE,
    }
    system = _system(prompt_utils.build_chat_prompt(
        persona=persona, context=CONTEXT, recent_messages=HISTORY,
        question=QUESTION, long_term_memory=MEMORIES))
    assert "测试医生" in system
    assert CONTEXT in system and QUESTION in system
    assert "{" not in system  # 占位符已全部填充


def test_build_context_block_respects_max_chars():
    hits = [{"text": "很长的一段知识" * 50, "source": "a.pdf", "score": 0.9}]
    context = prompt_utils.build_context_block(hits, max_chars=50)
    assert context == prompt_utils.NO_KNOWLEDGE_HINT


def test_build_recent_messages_block_limits_turns():
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"内容{i}"}
                for i in range(10)]
    block = prompt_utils.build_recent_messages_block(messages, max_turns=2)
    assert "内容0" not in block
    assert "内容9" in block


def test_default_system_prompt_lookup():
    for code in PERSONA_KEYWORDS:
        assert prompt_utils.get_default_system_prompt(code) == PERSONA_BY_CODE[code]["system_prompt"]