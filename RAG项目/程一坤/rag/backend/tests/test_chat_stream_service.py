"""ChatService 流式问答测试（检索/LLM 全替身，不联网）。

事件语义（与 app/api/chat.py 的 SSE 对应）：
- 正常：逐块 yield token 文本 → context["result"] 为最终 ChatResult
- 零引用等不可信回答：context["replace"] 置为兜底文案（前端整段替换，方案 A）
- 清单外法规：保留回答，末尾额外 yield 警示行
"""

from collections.abc import Iterator
from typing import Any

import pytest

from app.chat.guard import (
    CITATION_FALLBACK_NOTICE,
    NO_CITATION_WARNING,
    REFUSAL_ANSWER,
    UNCITED_CONCLUSION_WARNING,
    UNKNOWN_LAW_WARNING,
)
from app.chat.service import ChatService
from app.retrieval.context_builder import RetrievedArticle
from app.retrieval.service import RetrievalResult


def make_retrieval(articles: list[RetrievedArticle] | None = None) -> Any:
    if articles is None:
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
    return _FixedRetrieval(articles)


class _FixedRetrieval:
    def __init__(self, articles: list[RetrievedArticle]) -> None:
        self.articles = articles

    def retrieve(self, question: str, **kwargs: Any) -> RetrievalResult:
        return RetrievalResult(
            articles=self.articles,
            context_block="[1] 第四十七条正文" if self.articles else "",
            stats={"reranked_count": len(self.articles)},
        )


class StreamingLlm:
    """按预设块流式输出的 LLM 替身。"""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, str]] = []

    def stream_chat(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        self.calls.append((system_prompt, user_prompt))
        yield from self.chunks


GOOD_ANSWER = ["根据", "劳动合同法[1]，", "经济补偿按工作年限计算。"]
DISCLAIMER = "内容仅供法律信息参考，不能替代律师出具的正式法律意见。"
ZERO_CITE_ANSWER = ["试用期", "一般是三到六个月，", "具体看合同。"]
UNKNOWN_LAW_ANSWER = ["根据劳动合同法[1]，补偿按年限算。", "另外《民法典》也有规定。"]
ABSOLUTE_ANSWER = ["只要仲裁就", "必然胜诉[1]。"]
# 主体有引用、夹一句无引用的确定性结论（"必须…"）
UNCITED_CONCLUSION_ANSWER = [
    "根据劳动合同法[1]，",
    "补偿按年限算。",
    "用人单位必须提前三十日通知。",
]


def run_stream(chunks: list[str]) -> tuple[list[str], dict]:
    service = ChatService(retrieval_service=make_retrieval(), llm_client=StreamingLlm(chunks))
    context: dict[str, Any] = {}
    tokens = list(service.chat_stream("测试问题", stream_context=context))
    return tokens, context


class TestChatStream:
    def test_good_answer_streams_tokens_and_result(self) -> None:
        tokens, context = run_stream(GOOD_ANSWER)
        # 模型没自带免责声明 → 服务端补一个末段 token（含空行分隔）
        assert tokens == GOOD_ANSWER + [f"\n\n{DISCLAIMER}"]
        result = context["result"]
        assert result.refused is False
        assert result.answer.startswith("根据劳动合同法[1]，经济补偿按工作年限计算。")
        assert "citation_check_passed" in result.guardrail_applied
        assert result.sources  # 法源列表照常返回

    def test_zero_citation_appends_warning_token(self) -> None:
        """零引用（批次 10 分级）：回答保留，末尾追加警示行，不再整段 replace。"""
        tokens, context = run_stream(ZERO_CITE_ANSWER)
        assert tokens == ZERO_CITE_ANSWER + [
            f"\n\n{NO_CITATION_WARNING}",
            f"\n\n{DISCLAIMER}",
        ]
        assert "replace" not in context
        assert ZERO_CITE_ANSWER[0] in context["result"].answer
        assert NO_CITATION_WARNING in context["result"].answer
        assert any(
            g.startswith("citation_warning_no_citation")
            for g in context["result"].guardrail_applied
        )

    def test_unknown_law_appends_warning_token(self) -> None:
        tokens, context = run_stream(UNKNOWN_LAW_ANSWER)
        assert tokens == UNKNOWN_LAW_ANSWER + [
            f"\n\n{UNKNOWN_LAW_WARNING}",
            f"\n\n{DISCLAIMER}",
        ]
        assert UNKNOWN_LAW_WARNING in context["result"].answer
        assert any(g.startswith("citation_warning_unknown_law") for g in context["result"].guardrail_applied)

    def test_uncited_conclusion_appends_warning_not_replace(self) -> None:
        """个别句无引用的确定性结论：保留回答 + 追加警示，不发 replace。

        修正前该档落入 CitationError 分支 → 前端整段替换，好回答被丢弃。
        """
        tokens, context = run_stream(UNCITED_CONCLUSION_ANSWER)
        assert tokens == UNCITED_CONCLUSION_ANSWER + [
            f"\n\n{UNCITED_CONCLUSION_WARNING}",
            f"\n\n{DISCLAIMER}",
        ]
        assert "replace" not in context  # 关键：不再走整段替换
        answer = context["result"].answer
        assert "补偿按年限算" in answer
        assert "用人单位必须提前三十日通知" in answer
        assert UNCITED_CONCLUSION_WARNING in answer
        assert any(
            g.startswith("citation_warning_uncited_conclusion")
            for g in context["result"].guardrail_applied
        )

    def test_absolute_expression_triggers_replace_with_rewritten(self) -> None:
        """绝对化表述在流式下无法就地改写 → 走 replace，替换为改写后全文。"""
        tokens, context = run_stream(ABSOLUTE_ANSWER)
        assert context["replace"] is not None
        assert "必然胜诉" not in context["replace"]
        assert "较大胜算" in context["replace"]  # 改写后的审慎表述
        assert context["result"].answer == context["replace"]

    def test_no_candidates_still_streams_guidance(self) -> None:
        """无候选法源时调用 LLM，输出事实梳理与一般性引导。"""
        llm = StreamingLlm(["先保存劳动合同和解除通知。"])
        service = ChatService(
            retrieval_service=make_retrieval(articles=[]),
            llm_client=llm,
        )
        context: dict[str, Any] = {}
        tokens = list(service.chat_stream("我被公司开了怎么办？", stream_context=context))

        assert tokens[0] == "先保存劳动合同和解除通知。"
        assert llm.calls
        assert "没有找到可引用的法源" in llm.calls[0][1]
        assert context["result"].refused is False
        assert "citation_warning_no_citation" not in context["result"].guardrail_applied

    def test_refusal_matches_sync_and_streaming_sources(self) -> None:
        """业务拒答不携带法源，防止 SSE 继续发送互相矛盾的 citation。"""
        low_score_article = RetrievedArticle(
            chunk_key="chunk-weather",
            content="第四十七条正文",
            document_title="中华人民共和国劳动合同法",
            source_url="https://example.com",
            recall_score=0.2,
            vector_score=0.2,
            article_number="第四十七条",
            document_id="doc-law",
        )
        retrieval = make_retrieval([low_score_article])
        service = ChatService(
            retrieval_service=retrieval,
            llm_client=StreamingLlm(["不应调用模型"]),
            refusal_min_vector_score=0.6304,
        )

        sync_result = service.chat("今天天气怎么样")
        stream_context: dict[str, Any] = {}
        stream_tokens = list(
            service.chat_stream("今天天气怎么样", stream_context=stream_context)
        )
        stream_result = stream_context["result"]

        assert sync_result.refused is True
        assert sync_result.sources == []
        assert stream_result.refused is True
        assert stream_result.sources == []
        assert stream_result.context_excerpts == []
        assert stream_tokens == [REFUSAL_ANSWER]
        assert service.llm_client.calls == []

    def test_greeting_matches_sync_and_streaming_sources(self) -> None:
        """业务寒暄与同步路径一致，不发送低分检索 citation。"""
        service = ChatService(
            retrieval_service=make_retrieval(
                [
                    RetrievedArticle(
                        chunk_key="chunk-greeting",
                        content="第四十七条正文",
                        document_title="中华人民共和国劳动合同法",
                        source_url="https://example.com",
                        recall_score=0.2,
                        vector_score=0.2,
                        article_number="第四十七条",
                        document_id="doc-law",
                    )
                ]
            ),
            llm_client=StreamingLlm(["不应调用模型"]),
            refusal_min_vector_score=0.6304,
        )

        sync_result = service.chat("你好")
        stream_context: dict[str, Any] = {}
        stream_tokens = list(service.chat_stream("你好", stream_context=stream_context))
        stream_result = stream_context["result"]

        assert sync_result.refused is False
        assert sync_result.sources == []
        assert stream_result.refused is False
        assert stream_result.sources == []
        assert stream_result.context_excerpts == []
        assert stream_tokens
        assert service.llm_client.calls == []

    def test_missing_llm_raises(self) -> None:
        service = ChatService(retrieval_service=make_retrieval(), llm_client=None)
        with pytest.raises(ValueError):
            list(service.chat_stream("问题", stream_context={}))
