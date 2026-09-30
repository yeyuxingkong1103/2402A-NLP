"""护栏分级处理测试（批次 3-2；b) 案例口径由批次 10 更新为保留+警示）。

要求：
- a) 回答有 [n] 引用、只是提到清单外法规 → 保留回答 + 追加一行警示，不替换
- b) 回答零引用（有候选条文）→ 保留回答 + 追加一行警示，不替换（批次 10）
- c) 引用越界等引用错乱 → 仍整段替换为风险提示
- d) 回答主体有引用、个别句是无引用的确定性结论 → 保留回答 + 追加警示，不替换
     （修正逻辑倒挂：该档比"整篇零引用"轻，不该受最重的处置）
"""
import pytest

from app.chat.guard import (
    CITATION_FALLBACK_NOTICE,
    UNCITED_CONCLUSION_WARNING,
    UNKNOWN_LAW_WARNING,
)
from app.chat.service import ChatService
from app.retrieval.context_builder import RetrievedArticle
from app.retrieval.service import RetrievalResult


def _make_service(llm_answer: str) -> ChatService:
    """固定检索结果（法源清单只有劳动合同法）+ 固定 LLM 回答。"""

    class FixedRetrieval:
        def retrieve(self, question: str, **kwargs):
            return RetrievalResult(
                articles=[
                    RetrievedArticle(
                        chunk_key="chunk-47",
                        content="第四十七条正文",
                        document_title="中华人民共和国劳动合同法",
                        source_url="https://example.com",
                        recall_score=0.9,
                        article_number="第四十七条",
                        paragraph_number="1",
                        document_id="doc-law",
                    )
                ],
                context_block="[1] 第四十七条正文",
                stats={"reranked_count": 1},
            )

    class FixedLlm:
        def chat(self, system_prompt: str, user_prompt: str) -> str:
            return llm_answer

    return ChatService(retrieval_service=FixedRetrieval(), llm_client=FixedLlm())


def test_unknown_law_with_citation_kept_and_warning_appended():
    """a) 有 [n] 引用但提到清单外《民法典》→ 回答保留 + 追加警示行。"""
    answer = (
        "根据《劳动合同法》第四十七条[1]，经济补偿按工作年限计算。"
        "另外，《民法典》合同编也有相关规定。"
    )
    result = _make_service(answer).chat("经济补偿怎么算？")

    # 回答原文保留，不被整段替换
    assert "经济补偿按工作年限计算" in result.answer
    assert "《民法典》" in result.answer
    # 追加了一行警示
    assert UNKNOWN_LAW_WARNING in result.answer
    # 护栏记录的是警告而非整段失败
    assert any(
        item.startswith("citation_warning_unknown_law")
        for item in result.guardrail_applied
    )


def test_zero_citation_answer_kept_and_warning_appended():
    """b) 零引用（批次 10 分级）：回答保留 + 追加警示行，不整段替换。"""
    answer = "试用期一般是三到六个月，具体要看合同约定，建议咨询律师后再决定。"
    result = _make_service(answer).chat("试用期多长？")

    # 回答原文保留
    assert "试用期一般是三到六个月" in result.answer
    # 追加了零引用警示
    from app.chat.guard import NO_CITATION_WARNING

    assert NO_CITATION_WARNING in result.answer
    assert any(
        item.startswith("citation_warning_no_citation")
        for item in result.guardrail_applied
    )


def test_out_of_range_citation_still_replaced():
    """c) 引用越界：引用错乱不可核查，仍整段替换。"""
    answer = "根据《劳动合同法》第五条[5]，试用期不得超过六个月。"
    result = _make_service(answer).chat("试用期多长？")

    assert result.answer == CITATION_FALLBACK_NOTICE
    assert any(
        item.startswith("citation_check_failed")
        for item in result.guardrail_applied
    )


def test_empty_answer_still_replaced():
    """c 之二) 回答为空：无内容可保留，仍整段替换（不可核查档）。"""
    result = _make_service("").chat("试用期多长？")

    assert result.answer == CITATION_FALLBACK_NOTICE
    assert any(
        item.startswith("citation_check_failed")
        for item in result.guardrail_applied
    )


def test_uncited_conclusion_keeps_answer_and_warns():
    """d) 主体有 [1] 引用、夹一句无引用的确定性结论 → 保留回答 + 追加警示。

    修正前的行为：抛通用 CitationError → 整段替换，整篇好回答被丢掉，
    而更严重的"整篇零引用"反而只警示 —— 严重度与处置倒挂。
    """
    answer = (
        "根据劳动合同法[1]，经济补偿按工作年限计算。"
        "用人单位必须自用工之日起一个月内订立书面劳动合同。"
    )
    result = _make_service(answer).chat("经济补偿怎么算？")

    # 回答保留（主体引用可核查），不被整段替换
    assert result.answer != CITATION_FALLBACK_NOTICE
    assert "经济补偿按工作年限计算" in result.answer
    assert "一个月内订立书面劳动合同" in result.answer
    # 追加的是本档专属警示文案
    assert UNCITED_CONCLUSION_WARNING in result.answer
    # 护栏事件名正确（前端/落库据此判定处置档位）
    assert any(
        item.startswith("citation_warning_uncited_conclusion")
        for item in result.guardrail_applied
    )
    assert not any(
        item.startswith("citation_check_failed")
        for item in result.guardrail_applied
    )


def test_valid_citation_answer_untouched():
    """引用合规的回答：不替换、不追加警示。"""
    answer = "根据劳动合同法[1]，试用期不得超过六个月。"
    result = _make_service(answer).chat("试用期多长？")

    assert "试用期不得超过六个月" in result.answer
    assert UNKNOWN_LAW_WARNING not in result.answer
    assert result.answer != CITATION_FALLBACK_NOTICE


def test_statutory_percentage_range_not_corrupted():
    """法条里的「50%以上100%以下」是法定比例区间，改写护栏不得篡改引文。"""
    from app.chat.guard import rewrite_absolute_expressions

    text = "逾期不支付的，按应付金额50%以上100%以下的标准向劳动者加付赔偿金。"
    assert rewrite_absolute_expressions(text) == text  # 原文不动

    # 口语化的绝对化承诺仍然要改写
    claim = "这个案子100%能要回工资。"
    assert "100%" not in rewrite_absolute_expressions(claim)
