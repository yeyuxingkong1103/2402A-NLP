"""测试拒答与表述拦截模块。

任务书 4-C 要求：
1. 无候选时直接拒答（不调用大模型）
2. 命中绝对化表述时拦截并重写为审慎表述
3. 固定免责声明
"""
import pytest
from app.chat.guard import (
    should_refuse,
    check_absolute_expressions,
    ensure_disclaimer,
    REFUSAL_ANSWER,
)


def test_should_refuse_empty_candidates():
    """测试无候选条文时应该拒答"""
    assert should_refuse([]) is True


def test_should_refuse_with_candidates():
    """测试有候选条文时不拒答"""
    assert should_refuse(["条文1", "条文2"]) is False


def test_absolute_expression_detection():
    """测试绝对化表述检测"""
    # 包含绝对化表述
    answer1 = "这种情况下，公司必然败诉，保证胜诉。"
    issues1 = check_absolute_expressions(answer1)
    assert len(issues1) > 0
    assert any("必然" in issue or "保证胜诉" in issue for issue in issues1)

    # 包含"确定违法"
    answer2 = "公司的行为确定违法，肯定会被处罚。"
    issues2 = check_absolute_expressions(answer2)
    assert len(issues2) > 0
    assert any("确定违法" in issue or "肯定" in issue for issue in issues2)

    # 正常表述
    answer3 = "根据法律规定，公司可能需要支付赔偿金。"
    issues3 = check_absolute_expressions(answer3)
    assert len(issues3) == 0


def test_absolute_expression_rewrite():
    """测试绝对化表述重写为审慎表述"""
    from app.chat.guard import rewrite_absolute_expressions

    answer = "这种情况下，公司必然败诉，保证胜诉。"
    rewritten = rewrite_absolute_expressions(answer)

    # 绝对化词汇应该被替换
    assert "必然" not in rewritten
    assert "保证" not in rewritten
    # 应该包含审慎表述
    assert "可能" in rewritten or "通常" in rewritten or "一般" in rewritten


def test_ensure_disclaimer_adds_when_missing():
    """测试缺少免责声明时自动添加"""
    answer = "根据劳动合同法[1]，试用期不得超过六个月。"
    result = ensure_disclaimer(answer)

    assert "内容仅供" in result or "不能替代律师" in result


def test_ensure_disclaimer_preserves_when_present():
    """测试已有免责声明时不重复添加"""
    answer = """根据劳动合同法[1]，试用期不得超过六个月。

内容仅供法律信息参考，不能替代律师出具的正式法律意见。"""

    result = ensure_disclaimer(answer)

    # 不应该重复添加免责声明
    disclaimer_count = result.count("内容仅供")
    assert disclaimer_count == 1


def test_refusal_answer_constant():
    """测试拒答文本常量存在且合理"""
    assert REFUSAL_ANSWER is not None
    assert len(REFUSAL_ANSWER) > 50
    assert "无法确认" in REFUSAL_ANSWER or "未找到" in REFUSAL_ANSWER
    assert "建议" in REFUSAL_ANSWER
    assert "内容仅供" in REFUSAL_ANSWER


def test_guard_integration():
    """测试护栏集成：无候选 → 拒答，有候选 → 检查表述"""
    # 场景 1：无候选，应该拒答
    if should_refuse([]):
        answer = REFUSAL_ANSWER
        assert "无法确认" in answer or "未找到" in answer

    # 场景 2：有候选，正常回答
    normal_answer = "根据劳动合同法[1]，试用期最长不超过六个月。"
    issues = check_absolute_expressions(normal_answer)
    assert len(issues) == 0

    # 场景 3：有候选，但包含绝对化表述
    bad_answer = "根据劳动合同法[1]，公司必然败诉。"
    issues = check_absolute_expressions(bad_answer)
    assert len(issues) > 0
