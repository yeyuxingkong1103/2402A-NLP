"""护栏"无引用的实质性结论"规则细分测试（#4 修复）。

背景：旧实现用粗关键词表（应当/可以/不得/…/规定/超过/不满/期限）判定
"实质性结论"，导致 LLM 写"无法给出统一期限…"这类不带 [n] 的不确定
总结句也被拦。修复目标：
- 确定性结论（应当/必须/可以要求/属于违法…）且无 [n] → 拦
- 不确定/概括表述（无法确定/取决于/视情况/需结合…建议咨询…）且无 [n] → 放行
- 有 [n] → 放行（不变）；纯免责声明 → 放行（不变）
"""
import pytest

from app.chat.citation_check import (
    DETERMINISTIC_CONCLUSION_KEYWORDS,
    UNCERTAIN_EXPRESSION_KEYWORDS,
    CitationError,
    check_citations,
)


# ==================== a) 四类用例各一条 ====================


def test_deterministic_conclusion_without_citation_blocked():
    """确定性结论且无 [n] → 必须拦（没放过头）。"""
    answer = (
        "根据劳动合同法[1]，试用期最长不得超过六个月。"
        "用人单位必须自用工之日起一个月内订立书面劳动合同。"
    )
    with pytest.raises(CitationError) as exc_info:
        check_citations(answer, 1)
    assert "无引用的实质性结论" in str(exc_info.value)


def test_uncertain_summary_without_citation_passed():
    """不确定总结句且无 [n] → 放行。

    句中含旧词表会误判的主题词"期限"——修复前这里会抛 CitationError。
    """
    answer = (
        "根据劳动合同法[1]第四十七条，经济补偿按工作年限计算。"
        "具体能拿到多少补偿无法确定，赔偿的仲裁时效期限需结合具体事实判断。"
    )
    check_citations(answer, 1)


def test_sentences_with_citation_passed():
    """带 [n] 的结论 → 照旧放行（现有逻辑不变）。"""
    answer = "根据劳动合同法[1]，试用期不得超过六个月[1]。"
    check_citations(answer, 1)


def test_pure_disclaimer_passed():
    """纯免责声明 → 照旧放行。"""
    check_citations("内容仅供法律信息参考，不能替代律师出具的正式法律意见。", 1)


# ==================== b) 用户点名的对照句对 ====================


def test_you_can_demand_compensation_blocked():
    """「你可以要求公司支付补偿」是确定性断言（"可以要求"）→ 拦。"""
    answer = "根据劳动合同法[1]，试用期最长六个月。你可以要求公司支付补偿。"
    with pytest.raises(CitationError):
        check_citations(answer, 1)


def test_depends_on_circumstances_passed():
    """「能否获得补偿取决于具体情况」是不确定表述 → 放行。"""
    answer = "根据劳动合同法[1]。能否获得补偿，取决于具体情况。"
    check_citations(answer, 1)


# ==================== c) 词表契约（防止旧粗规则回潮） ====================


def test_topic_words_not_in_deterministic_list():
    """主题词不是结论动词，不得回到确定性词表（旧误拦的根因）。"""
    for banned in ("规定", "超过", "不满", "期限"):
        assert banned not in DETERMINISTIC_CONCLUSION_KEYWORDS


def test_required_deterministic_words_present():
    """用户点名的确定性结论词必须在词表中。"""
    for required in ("应当", "必须", "可以要求", "属于违法"):
        assert required in DETERMINISTIC_CONCLUSION_KEYWORDS


def test_required_uncertain_expressions_present():
    """用户点名的不确定/概括表述必须在词表中。"""
    for required in (
        "无法确定",
        "取决于",
        "视情况",
        "需结合",
        "建议咨询",
    ):
        assert required in UNCERTAIN_EXPRESSION_KEYWORDS


def test_advice_to_consult_lawyer_passed():
    """「建议咨询」类结尾句无 [n] → 放行（含旧词表误判词"期限"）。"""
    answer = (
        "根据劳动合同法[1]，用人单位应当及时足额支付劳动报酬[1]。"
        "如果情况复杂，建议咨询专业律师后再决定申诉期限。"
    )
    check_citations(answer, 1)


def test_mixed_uncertain_wins_over_deterministic():
    """同一句里既有确定性词又含不确定表述 → 放行（宁可少拦不误拦）。"""
    answer = "根据劳动合同法[1]。是否属于违法，取决于具体事实，可能需要视情况认定。"
    check_citations(answer, 1)
