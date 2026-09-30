"""批次 10 任务 3：护栏零引用分级的三场景验收测试。

场景：
1. 有候选条文但回答零引用 → 保留回答 + 追加警示（不整段替换）
2. 无候选（零候选触发拒答）→ 保持拒答行为
3. 正常带引用 → 回答不变、无警示
"""

from app.chat.guard import NO_CITATION_WARNING, REFUSAL_ANSWER
from app.chat.service import ChatService
from app.retrieval.context_builder import RetrievedArticle
from app.retrieval.service import RetrievalResult


def _make_service(llm_answer: str, *, with_candidates: bool) -> ChatService:
    """固定检索结果 + 固定 LLM 回答；with_candidates 控制是否有候选条文。"""

    class FixedRetrieval:
        def retrieve(self, question: str, **kwargs):
            articles = []
            if with_candidates:
                articles = [
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
                ]
            return RetrievalResult(
                articles=articles,
                context_block="[1] 第四十七条正文" if articles else "",
                stats={"reranked_count": len(articles)},
            )

    class FixedLlm:
        def chat(self, system_prompt: str, user_prompt: str) -> str:
            return llm_answer

    return ChatService(retrieval_service=FixedRetrieval(), llm_client=FixedLlm())


def test_zero_citation_kept_with_warning():
    """场景 1：有候选 + 零引用回答 → 保留 + 警示。"""
    answer = "根据检索到的法条，试用期通常由合同约定，建议咨询律师。"
    result = _make_service(answer, with_candidates=True).chat("试用期多长？")

    assert "试用期通常由合同约定" in result.answer
    assert NO_CITATION_WARNING in result.answer
    assert not result.refused
    assert any(
        g.startswith("citation_warning_no_citation") for g in result.guardrail_applied
    )


def test_no_candidates_allows_general_guidance():
    """无候选时保留 LLM 的一般性引导，并继续补免责声明。"""
    answer = "建议先保存解除通知，并整理发生时间和公司说明。"
    result = _make_service(answer, with_candidates=False).chat("我被公司开了怎么办")

    assert result.refused is False
    assert "建议先保存解除通知" in result.answer
    assert "内容仅供法律信息参考" in result.answer
    assert "refuse_no_candidates" not in result.guardrail_applied


def test_normal_cited_answer_unchanged():
    """场景 3：正常带引用 → 无警示、原样保留。"""
    answer = "根据劳动合同法[1]，试用期不得超过六个月。"
    result = _make_service(answer, with_candidates=True).chat("试用期多长？")

    assert "试用期不得超过六个月" in result.answer
    assert NO_CITATION_WARNING not in result.answer
    assert "citation_check_passed" in result.guardrail_applied
