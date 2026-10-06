# 拒答规则是合规项（FR-8.3、AC-19），必须确定性——所以用规则而不是模型判断。
# 假阳性同样有害：把正常提问拒掉，用户直接流失。故负例与正例一样重要。
import pytest

from app.generation.boundaries import (
    PUBLIC_DISCLAIMER, REFUSAL_REPLY, apply_boundaries, is_refusal,
)
from app.generation.schema import Answer
from app.generation.profiles import SIDE_INTERNAL, SIDE_PUBLIC


@pytest.mark.parametrize("question", [
    "这个官司我能赢吗",
    "我能不能打赢这场官司",
    "帮我写一份起诉状",
    "替我起草一份答辩状",
    "你能不能代理我的案子",
    "帮我打官司",
    "这个案子胜诉概率多大",
])
def test_refuses_case_conclusion_and_agency_requests(question):
    assert is_refusal(question) is True


@pytest.mark.parametrize("question", [
    "租房押金不退怎么办",
    "合同解除需要什么条件",
    "民间借贷没有约定利息还能要利息吗",
    "民法典第五百八十四条规定了什么",
    "买了别人无权处分的二手车能取得所有权吗",
    "我赢了公司的劳动仲裁，接下来该怎么做",  # 含"赢"但不是在问胜诉预测
])
def test_does_not_refuse_ordinary_questions(question):
    assert is_refusal(question) is False


def test_public_answer_gets_code_injected_disclaimer():
    """AC-18 是合规项：免责由代码强制注入，不靠模型心情。"""
    ans = Answer.model_validate({"status": "ok", "answer": "正文", "citations": [],
                                 "disclaimer": "模型自己写的免责"})
    result = apply_boundaries(ans, SIDE_PUBLIC)
    assert result.disclaimer == PUBLIC_DISCLAIMER
    assert "模型自己写的免责" not in result.disclaimer
    # 钉正文而不仅钉"等于常量"：否则把常量整段改成 "x" 也全绿（变异 M6 实证），
    # 合规措辞就失去了测试层防线。与 REFUSAL_REPLY 钉「律师」的做法对称
    assert "仅供参考" in result.disclaimer
    assert "不构成正式法律意见" in result.disclaimer
    assert "咨询执业律师" in result.disclaimer


def test_internal_answer_has_no_disclaimer():
    ans = Answer.model_validate({"status": "ok", "answer": "正文", "citations": [],
                                 "disclaimer": "public 那份被模型串了"})
    assert apply_boundaries(ans, SIDE_INTERNAL).disclaimer == ""


def test_unknown_side_raises_instead_of_silently_dropping_disclaimer():
    # 合规兜底模块不许 fail-open：侧别写错时最危险的失败方向是"公众答案没有免责"。
    # profiles.system_prompt 与 llm_router.get_llm 对未知侧都抛错，此处同一口径
    ans = Answer.model_validate({"status": "ok", "answer": "正文", "citations": []})
    with pytest.raises(ValueError):
        apply_boundaries(ans, "public_typo")


def test_apply_boundaries_does_not_mutate_input():
    ans = Answer.model_validate({"status": "ok", "answer": "正文", "citations": []})
    apply_boundaries(ans, SIDE_PUBLIC)
    assert ans.disclaimer == ""


def test_refusal_reply_mentions_lawyer_consultation():
    # 拒绝之后必须给出路，否则等于把用户扔在原地
    assert "律师" in REFUSAL_REPLY
