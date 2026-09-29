from dataclasses import replace
"""低分候选的三档处置测试。"""

from types import SimpleNamespace
from typing import Any, Iterator

from app.chat.guard import GREETING_ANSWER, REFUSAL_ANSWER
from app.chat.service import ChatService
from app.retrieval.context_builder import RetrievedArticle
from app.retrieval.service import RetrievalResult


def _article(score: float, chunk_key: str | None = None) -> RetrievedArticle:
    return RetrievedArticle(
        chunk_key=chunk_key or f"chunk-{score}",
        content="第四十七条 经济补偿按工作年限计算。",
        document_title="中华人民共和国劳动合同法",
        source_url="https://example.com",
        recall_score=score,
        vector_score=score,
        article_number="第四十七条",
        paragraph_number="1",
        document_id="doc-law",
    )


class _ShortTermMemory:
    def __init__(self, messages: list[dict[str, str]]) -> None:
        self.messages = messages

    def read_messages(self, user_id: str, session_id: str) -> list[dict[str, str]]:
        return self.messages


class _SequentialRetrieval:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.questions: list[str] = []

    def retrieve(self, question: str, **kwargs: Any) -> RetrievalResult:
        self.questions.append(question)
        return self.results.pop(0)


class _Llm:
    def __init__(self, answer: str = "根据劳动合同法[1]，可以依法主张经济补偿。") -> None:
        self.answer = answer
        self.calls = 0

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self.answer

    def stream_chat(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        self.calls += 1
        yield self.answer


def _result(score: float, chunk_key: str | None = None) -> RetrievalResult:
    articles = [_article(score, chunk_key)]
    return RetrievalResult(
        articles=articles,
        context_block="[1] 第四十七条 经济补偿按工作年限计算。",
        stats={"reranked_count": 1},
    )


def test_greeting_returns_guidance_instead_of_refusal() -> None:
    for question in ("你好", "在吗", "谢谢"):
        retrieval = _SequentialRetrieval([_result(0.2)])
        llm = _Llm()
        service = ChatService(
            retrieval_service=retrieval,
            llm_client=llm,
            refusal_min_vector_score=0.6304,
        )

        result = service.chat(question)

        assert result.answer == GREETING_ANSWER
        assert REFUSAL_ANSWER not in result.answer
        assert result.refused is False
        assert llm.calls == 0


def test_unrelated_question_still_refuses() -> None:
    retrieval = _SequentialRetrieval([_result(0.2)])
    llm = _Llm()
    service = ChatService(
        retrieval_service=retrieval,
        llm_client=llm,
        refusal_min_vector_score=0.6304,
    )

    result = service.chat("今天天气怎么样")

    assert result.answer == REFUSAL_ANSWER
    assert result.refused is True
    assert llm.calls == 0


def test_labor_domain_low_score_uses_general_guidance_without_citations() -> None:
    class GuidanceLlm(_Llm):
        def __init__(self) -> None:
            super().__init__(
                "先保存解除通知、工资记录和聊天记录，再补充说明发生时间和公司理由。"
            )
            self.user_prompts: list[str] = []

        def chat(self, system_prompt: str, user_prompt: str) -> str:
            self.user_prompts.append(user_prompt)
            return super().chat(system_prompt, user_prompt)

    for question in ("我被辞退了", "我干了三年被无辜辞退"):
        retrieval = _SequentialRetrieval([_result(0.2)])
        llm = GuidanceLlm()
        service = ChatService(
            retrieval_service=retrieval,
            llm_client=llm,
            refusal_min_vector_score=0.6304,
        )

        result = service.chat(question)

        assert result.refused is False
        assert result.sources == []
        assert result.context_excerpts == []
        assert "没有找到可引用的法源" in llm.user_prompts[0]
        assert "第四十七条" not in llm.user_prompts[0]
        assert "先保存解除通知" in result.answer


def test_labor_domain_follow_up_uses_previous_domain_context() -> None:
    retrieval = _SequentialRetrieval([_result(0.2), _result(0.2)])
    retrieval.short_term_memory = _ShortTermMemory(
        [{"role": "user", "content": "我被公司辞退了"}]
    )
    llm = _Llm("可以先整理工作年限、工资和解除经过，再核对具体材料。")
    service = ChatService(
        retrieval_service=retrieval,
        llm_client=llm,
        refusal_min_vector_score=0.6304,
    )

    result = service.chat(
        "那我能拿多少",
        user_id="user-1",
        session_id="session-1",
    )

    assert result.refused is False
    assert result.sources == []
    assert result.context_excerpts == []
    assert llm.calls == 1


def test_labor_domain_guidance_matches_streaming_without_citation_event() -> None:
    sync_retrieval = _SequentialRetrieval([_result(0.2)])
    sync_service = ChatService(
        retrieval_service=sync_retrieval,
        llm_client=_Llm("先保存解除通知和工资记录，再补充发生时间。"),
        refusal_min_vector_score=0.6304,
    )
    sync_result = sync_service.chat("我被辞退了")

    stream_retrieval = _SequentialRetrieval([_result(0.2)])
    stream_service = ChatService(
        retrieval_service=stream_retrieval,
        llm_client=_Llm("先保存解除通知和工资记录，再补充发生时间。"),
        refusal_min_vector_score=0.6304,
    )
    stream_context: dict[str, Any] = {}
    stream_tokens = list(
        stream_service.chat_stream("我被辞退了", stream_context=stream_context)
    )
    stream_result = stream_context["result"]

    assert sync_result.refused is False
    assert stream_result.refused is False
    assert sync_result.sources == stream_result.sources == []
    assert stream_result.context_excerpts == []
    assert "第四十七条" not in "".join(stream_tokens)
    assert "low_score_domain_guidance" in sync_result.guardrail_applied
    assert "low_score_domain_guidance" in stream_result.guardrail_applied


def test_unrelated_question_after_labor_context_still_refuses() -> None:
    retrieval = _SequentialRetrieval([_result(0.2), _result(0.2)])
    retrieval.short_term_memory = _ShortTermMemory(
        [{"role": "user", "content": "我被公司辞退了"}]
    )
    llm = _Llm()
    service = ChatService(
        retrieval_service=retrieval,
        llm_client=llm,
        refusal_min_vector_score=0.6304,
    )

    result = service.chat(
        "今天天气怎么样",
        user_id="user-1",
        session_id="session-1",
    )

    assert result.refused is True
    assert llm.calls == 0


def test_low_score_follow_up_retries_with_previous_question() -> None:
    retrieval = _SequentialRetrieval([_result(0.2), _result(0.9)])
    llm = _Llm()
    retrieval.short_term_memory = _ShortTermMemory(
        [{"role": "user", "content": "我近期被公司开除了，我想要补偿"}]
    )
    service = ChatService(
        retrieval_service=retrieval,
        llm_client=llm,
        refusal_min_vector_score=0.6304,
    )

    result = service.chat(
        "没有协商直接告诉我明天不用来了",
        user_id="user-1",
        session_id="session-1",
    )

    assert result.refused is False
    assert llm.calls == 1
    assert len(retrieval.questions) == 2
    assert "我近期被公司开除了，我想要补偿" in retrieval.questions[1]
    assert "没有协商直接告诉我明天不用来了" in retrieval.questions[1]
    assert "low_score_retry_hit" in result.guardrail_applied


def test_low_score_retry_skips_previous_greeting() -> None:
    retrieval = _SequentialRetrieval([_result(0.2), _result(0.9)])
    retrieval.short_term_memory = _ShortTermMemory(
        [
            {"role": "user", "content": "我近期被公司开除了，我想要补偿"},
            {"role": "assistant", "content": "已说明补偿判断方向。"},
            {"role": "user", "content": "你好"},
        ]
    )
    service = ChatService(
        retrieval_service=retrieval,
        llm_client=_Llm(),
        refusal_min_vector_score=0.6304,
    )

    result = service.chat(
        "没有协商直接告诉我明天不用来了",
        user_id="user-1",
        session_id="session-1",
    )

    assert result.refused is False
    assert "我近期被公司开除了，我想要补偿" in retrieval.questions[1]
    assert "你好" not in retrieval.questions[1]


def test_sync_and_streaming_use_same_three_tier_decision() -> None:
    sync_retrieval = _SequentialRetrieval([_result(0.2)])
    sync_service = ChatService(
        retrieval_service=sync_retrieval,
        llm_client=_Llm(),
        refusal_min_vector_score=0.6304,
    )
    sync_result = sync_service.chat("你好")

    stream_retrieval = _SequentialRetrieval([_result(0.2)])
    stream_service = ChatService(
        retrieval_service=stream_retrieval,
        llm_client=_Llm(),
        refusal_min_vector_score=0.6304,
    )
    stream_context: dict[str, Any] = {}
    stream_tokens = list(stream_service.chat_stream("你好", stream_context=stream_context))

    assert sync_result.answer == GREETING_ANSWER
    assert "".join(stream_tokens) == GREETING_ANSWER
    assert stream_context["result"].answer == sync_result.answer
    assert stream_context["result"].refused == sync_result.refused


def test_low_score_retry_keeps_original_golden_candidate() -> None:
    original = _result(0.2, "golden-chunk")
    retry_result = _result(0.9, "retry-chunk")
    original.articles[0] = replace(
        original.articles[0], content="原候选 golden 法条"
    )
    retry_result.articles[0] = replace(
        retry_result.articles[0], content="重试候选非 golden 法条"
    )
    original.context_block = "原候选 golden 法条"
    retry_result.context_block = "重试候选非 golden 法条"
    retrieval = _SequentialRetrieval([original, retry_result])
    retrieval.short_term_memory = _ShortTermMemory(
        [{"role": "user", "content": "我近期被公司开除了，我想要补偿"}]
    )
    service = ChatService(
        retrieval_service=retrieval,
        llm_client=_Llm(),
        refusal_min_vector_score=0.6304,
    )

    result = service.chat("没有协商直接告诉我明天不用来了", user_id="user-1", session_id="session-1")

    assert result.refused is False
    assert "low_score_retry_hit" in result.guardrail_applied
    assert {excerpt["content"] for excerpt in result.context_excerpts} == {"原候选 golden 法条", "重试候选非 golden 法条"}
