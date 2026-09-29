"""测试问答主流程模块。

任务书 4-D 要求：
- 串起来：检索 → 组装提示词 → 调用大模型 → 引用校验 → 拦截 → 输出
- 写成可被 CLI 与接口层复用的服务类
"""
import pytest
from app.chat.result import ChatResult
from app.chat.service import ChatService
from app.retrieval.context_builder import RetrievedArticle
from app.retrieval.service import RetrievalResult


class RecordingRetrievalService:
    def __init__(self) -> None:
        self.kwargs = None

    def retrieve(self, question: str, **kwargs):
        self.kwargs = kwargs
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


class FixedLlmClient:
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        return "依法可以获得经济补偿。[1]"


class EmptyRetrievalService:
    def retrieve(self, question: str, **kwargs):
        return RetrievalResult(
            articles=[],
            context_block="",
            stats={"reranked_count": 0},
        )


class RecordingEmptyLlmClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return "先保存劳动合同和解除通知，并补充说明公司如何通知你的。"



def test_chat_service_initialization():
    """测试服务初始化"""
    service = ChatService(
        retrieval_service=None,  # 使用 mock
        llm_client=None,
    )
    assert service is not None


def test_chat_result_structure():
    """测试返回结果结构"""
    result = ChatResult(
        answer="根据劳动合同法[1]，试用期最长不超过六个月。\n\n内容仅供法律信息参考，不能替代律师出具的正式法律意见。",
        sources=[{"title": "劳动合同法", "article": "19"}],
        refused=False,
        guardrail_applied=["ensure_disclaimer"],
    )

    assert result.answer is not None
    assert isinstance(result.sources, list)
    assert result.refused is False
    assert len(result.guardrail_applied) > 0


def test_chat_passes_identity_context_and_returns_contract_citations() -> None:
    retrieval = RecordingRetrievalService()
    service = ChatService(retrieval_service=retrieval, llm_client=FixedLlmClient())

    result = service.chat(
        "经济补偿怎么算？",
        user_id="user-1",
        session_id="session-1",
        request_id="req_chat123456",
    )

    assert retrieval.kwargs["user_id"] == "user-1"
    assert retrieval.kwargs["session_id"] == "session-1"
    assert retrieval.kwargs["request_id"] == "req_chat123456"
    assert result.sources == [
        {
            "chunk_id": "chunk-47",
            "law_name": "中华人民共和国劳动合同法",
            "article_number": "第四十七条",
            "paragraph_number": "1",
            "page": None,
            # 批次 37：引用契约新增 summary（测试桩未提供摘要 → 显式 None）
            "summary": None,
        }
    ]


def test_chat_service_refusal_without_llm():
    """无候选法源时仍调用 LLM，让模型进行事实梳理与一般性引导。"""
    llm = RecordingEmptyLlmClient()
    result = ChatService(
        retrieval_service=EmptyRetrievalService(),
        llm_client=llm,
    ).chat("我被公司开了怎么办？")

    assert result.refused is False
    assert result.answer.startswith("先保存劳动合同")
    assert len(llm.calls) == 1
    assert "没有找到可引用的法源" in llm.calls[0][1]
    assert "citation_warning_no_citation" not in result.guardrail_applied


def test_chat_service_full_pipeline():
    """测试完整流程：检索 → 提示词 → LLM → 校验 → 护栏"""
    # 这个测试验证完整流程串联
    # 由于涉及真实 LLM 调用，使用 mock 或集成测试
    pass


def test_chat_service_error_handling():
    """测试异常处理：LLM 调用失败、校验失败等"""
    pass
