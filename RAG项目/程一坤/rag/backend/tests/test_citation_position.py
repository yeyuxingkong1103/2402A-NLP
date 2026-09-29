"""引用位置归一化测试。

背景：LLM 书写习惯是「结论。[8]」（引用编号在句末标点之后），
而 citation_check 按 。！ 切句会把句号吃掉，导致引用编号落入下一句，
每个实质句都被判成「无引用」→ 好答案被整段替换。

归一化规则：句末标点后紧跟的 [n] 归属前一句，即「。[8]」视为「[8]。」。
两类书写习惯（标点前/后）都必须通过校验。
"""

import pytest

from app.chat.citation_check import check_citations, CitationError


CONTEXT = "[1] 中华人民共和国劳动合同法\n[2] 工资支付暂行规定"


def test_citation_after_period_passes() -> None:
    """「工资。[1]」句号后引用：归一化后应通过校验（修复前红灯）。"""
    answer = "用人单位应当按时足额支付工资。[1]不得克扣或者无故拖欠。[2]"
    check_citations(answer, 2, ["中华人民共和国劳动合同法", "工资支付暂行规定"])


def test_citation_before_period_still_passes() -> None:
    """「工资[1]。」句号前引用：原有习惯，回归保障。"""
    answer = "用人单位应当按时足额支付工资[1]。不得克扣或者无故拖欠[2]。"
    check_citations(answer, 2, ["中华人民共和国劳动合同法", "工资支付暂行规定"])


def test_mixed_styles_pass() -> None:
    """同一段回答里两种写法混用，都应通过。"""
    answer = (
        "工资应当以货币形式按月支付。[1]"
        "拖欠工资的，劳动者可以投诉[2]。"
    )
    check_citations(answer, 2, ["中华人民共和国劳动合同法", "工资支付暂行规定"])


def test_truly_uncited_conclusion_still_blocked() -> None:
    """没有引用的确定性结论仍要拦截——归一化不许放走真问题。"""
    answer = "用人单位必须自用工之日起一个月内订立书面劳动合同。拖欠工资应当加付赔偿金。"
    with pytest.raises(CitationError):
        check_citations(answer, 2, ["中华人民共和国劳动合同法", "工资支付暂行规定"])


def test_markdown_structure_with_trailing_citations_passes() -> None:
    """真实 LLM 输出：markdown 小节 + 列表 + 句号后引用，修复前必被误杀。"""
    answer = (
        "## 直接回答\n\n"
        "单位拖欠工资，你可以向劳动行政部门投诉，由其责令支付。[2]\n\n"
        "## 依据\n\n"
        "**一、支付义务**\n\n"
        "- 「法规原文」《工资支付暂行规定》第十八条：对克扣或者无故拖欠工资的，"
        "责令支付工资和经济补偿。[2]\n"
        "- 「法规原文」《劳动合同法》第三十条：用人单位应当按照约定，"
        "及时足额支付劳动报酬。[1]"
    )
    check_citations(answer, 2, ["中华人民共和国劳动合同法", "工资支付暂行规定"])
