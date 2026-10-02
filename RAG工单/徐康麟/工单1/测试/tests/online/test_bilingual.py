"""在线测试：中英文双语问答（工单追加要求）。

覆盖：
- 语言检测（中文 / 英文 / 英文缩写不误判）；
- 英文提问的语言桥接（中文问句模板 + 中文关键词）；
- 英文回答的确定性（数字与单位由代码换算，不由模型翻译）；
- 英文引用格式 ``[Page: N]``；
- 英文无关问题必须兜底为 ``I don't know``；
- 中文链路不受影响（回归保护）。
"""

from __future__ import annotations

import re

import pytest

from app.core.english_answer import get_english_builder
from app.core.language import (
    CHINESE_QUERY_TEMPLATES,
    detect_english_intent,
    detect_language,
    format_amount_en,
    localize,
    resolve_answer_language,
)
from app.core.number_utils import parse_amount


# ==========================================================================
# 语言检测
# ==========================================================================
@pytest.mark.parametrize(
    "text, expected",
    [
        ("注册资本是多少？", "zh"),
        ("What is the registered capital?", "en"),
        ("Who is the legal representative of the company?", "en"),
        ("How much revenue came from the military sector?", "en"),
        ("C4ISR", "zh"),  # 纯缩写不应误判为英文提问
        ("2016", "zh"),  # 纯数字按中文处理
    ],
)
def test_detect_language(text: str, expected: str) -> None:
    """语言检测必须区分中英文，并且不把缩写/数字误判成英文提问。"""
    assert detect_language(text) == expected, f"{text!r} 应判为 {expected}"


def test_resolve_answer_language_follows_question() -> None:
    """auto 模式下回答语言跟随提问语言。"""
    assert resolve_answer_language("注册资本是多少？", "auto") == "zh"
    assert resolve_answer_language("What is the registered capital?", "auto") == "en"


def test_resolve_answer_language_can_be_forced() -> None:
    """显式配置的语言优先于自动检测。"""
    assert resolve_answer_language("What is the registered capital?", "zh") == "zh"
    assert resolve_answer_language("注册资本是多少？", "en") == "en"


# ==========================================================================
# 英文意图识别与中文问句模板
# ==========================================================================
@pytest.mark.parametrize(
    "question, expected_intent",
    [
        ("What is the registered capital of the company?", "注册资本"),
        ("Who is the legal representative?", "法定代表人"),
        ("Which technical standard did the company formulate?", "技术标准"),
        ("How much of the raised funds will be used to supplement working capital?", "募资用途"),
        ("What percentage of main business revenue came from the military?", "收入占比"),
        ("What are the upstream and downstream of the industry?", "上下游"),
    ],
)
def test_detect_english_intent(question: str, expected_intent: str) -> None:
    """英文意图识别必须能选出正确的中文问句模板。

    这是英文问答的关键一环：中文意图规则对英文无效，若不补判，
    检索只能拿到裸关键词（余弦低于阈值），英文问题会全部答不出来。
    """
    assert detect_english_intent(question) == expected_intent
    assert expected_intent in CHINESE_QUERY_TEMPLATES, f"{expected_intent} 缺少中文问句模板"


def test_every_intent_template_is_a_chinese_question() -> None:
    """所有中文问句模板都必须是含中文疑问结构的完整句子。"""
    for intent, template in CHINESE_QUERY_TEMPLATES.items():
        assert re.search(r"[\u4e00-\u9fff]", template), f"{intent} 模板不含中文"
        assert len(template) >= 6, f"{intent} 模板过短：{template!r}"


# ==========================================================================
# 英文答案的确定性（数字与单位不能交给模型翻译）
# ==========================================================================
def test_amount_english_uses_exact_conversion() -> None:
    """金额英文表述必须由代码换算，不能出现 100 倍量级错误。

    背景：本地 0.6B 小模型会把“5,520 万元”译成 “5,520 million yuan”（差 100 倍），
    因此金额一律走确定性换算。
    """
    text = format_amount_en("5,520 万元")
    assert "55.20 million" in text, f"5,520 万元 = 5,520 万元 = 5520万元 = 55.2 百万元，实际：{text}"
    assert "5,520.00 ten-thousand yuan" in text

    # 对照：解析成元后应为 55,200,000
    assert parse_amount("5,520 万元") == pytest.approx(55_200_000)
    assert parse_amount("1.5 亿元") == pytest.approx(150_000_000)


def test_localize_translates_domain_terms() -> None:
    """领域术语必须被替换成英文。"""
    localized = localize("公司目前已经成为军队视频指挥领域的重要供应商")
    assert "major supplier" in localized, f"未替换“重要供应商”，实际：{localized}"
    assert "video command" in localized, f"未替换“视频指挥”，实际：{localized}"


# ==========================================================================
# 端到端：英文问答（需要索引）
# ==========================================================================
EN_CASES: list[tuple[str, str, bool]] = [
    ("What is the registered capital of the company?", r"55\.20 million", False),
    ("Who is the legal representative?", r"Cheng Jiaming|程家明", False),
    ("Which technical standard did the company participate in formulating?", r"technical standard", False),
    ("What percentage of main business revenue came from the military sector during the reporting period?", r"82\.10", False),
    ("What is the weather like today?", r".*", True),
]


@pytest.mark.parametrize("question, pattern, expect_unknown", EN_CASES)
def test_english_question_end_to_end(engine, requires_index, question: str, pattern: str, expect_unknown: bool) -> None:
    """英文提问必须得到英文回答（或对无关问题兜底）。"""
    answer = engine.ask(question, allow_llm=False, save=False)

    if expect_unknown:
        assert answer.is_unknown, f"无关的英文问题应兜底，实际回答：{answer.answer!r}"
        # 必须严格是**英文**兜底文案：曾经这里写成“英文或中文都可以”，
        # 结果漏掉了“英文问题被中文兜底”的真实缺陷（qa_engine 未把 language
        # 传给 _unknown）。断言必须能抓住这个回归。
        assert answer.answer == "I don't know", (
            f"英文提问的兜底文案必须是英文，实际：{answer.answer!r}"
        )
        assert answer.language == "en", f"兜底回答的语言标记应为 en，实际：{answer.language}"
        return

    assert not answer.is_unknown, f"英文问题不应兜底，问题：{question}"
    assert re.search(pattern, answer.answer), f"英文回答未命中期望 {pattern!r}，实际：{answer.answer!r}"
    # 英文回答不应包含中文兜底文案
    assert "不清楚" not in answer.answer, f"英文回答不应出现中文兜底文案：{answer.answer!r}"


def test_english_answer_uses_english_citation_label(engine, requires_index) -> None:
    """英文回答的引用必须用 ``[Page: N]`` 而不是 ``[页码: N]``。"""
    answer = engine.ask("What is the registered capital of the company?", allow_llm=False, save=False)
    assert answer.citations, "英文回答必须带引用"
    assert answer.language == "en", f"回答语言应为 en，实际 {answer.language}"
    for citation in answer.citations:
        label = citation.label("en")
        assert label.startswith("[Page:"), f"英文引用标签错误：{label}"
        assert citation.page >= 1


def test_chinese_path_unaffected(engine, requires_index) -> None:
    """英文链路的引入不能破坏中文问答（回归保护）。"""
    answer = engine.ask("武汉兴图新科电子股份有限公司注册资本是多少？", allow_llm=False, save=False)
    assert not answer.is_unknown, "中文问题不应兜底"
    assert "5,520" in answer.answer, f"中文回答应含 5,520，实际：{answer.answer!r}"
    assert answer.language == "zh", f"回答语言应为 zh，实际 {answer.language}"


def test_english_builder_returns_none_for_uncovered_question(engine) -> None:
    """模板与术语表都覆盖不到时，构造器返回 None 交由上层兜底，而不是编造。"""
    builder = get_english_builder()
    outcome = builder.build("What is the CEO's favourite colour?", [], None)
    assert outcome is None, "无证据时不应构造出答案"


def test_english_fallback_is_english_within_conversation(engine, requires_index) -> None:
    """多轮会话中英文无关问题的兜底也必须是英文。

    回归背景：``QAEngine.ask`` 的“相关度不足”分支调用
    ``generator._unknown(...)`` 时漏传了 ``language``，
    导致英文问题被拒答时返回中文「不清楚」。
    该缺陷只在**带会话上下文**、且走到置信度闸门时暴露，
    因此需要单独一条用例固定住。
    """
    conversation = engine.new_conversation()
    engine.ask("What is the registered capital of the company?", conversation_id=conversation, allow_llm=False, save=True)
    answer = engine.ask("What is the weather like today?", conversation_id=conversation, allow_llm=False, save=True)

    assert answer.is_unknown, f"无关的英文追问应兜底，实际：{answer.answer!r}"
    assert answer.answer == "I don't know", (
        f"会话内的英文兜底必须是英文，实际：{answer.answer!r}"
    )
    assert "不清楚" not in answer.answer, "英文兜底不得出现中文文案"
