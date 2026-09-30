import time
from collections import deque

import pytest

from backend.app.rag.pipeline import RetrievalDecision
from backend.app.schemas.chat import FactCorrection
from backend.app.services.chat_service import (
    ChatService,
    MessageTooLongError,
    RegenerationLimitError,
    classify_scope,
    validate_message_length,
)


class FakeLlm:
    def __init__(self, answer: str = "可依据材料处理。"):
        self.answer = answer
        self.requests = []

    async def stream_chat(self, request):
        self.requests.append(request)
        yield self.answer


def allowed_decision(context: str = "民法典婚姻家庭编相关依据摘要。"):
    return RetrievalDecision(
        can_answer=True,
        reason="ok",
        top_documents=[],
        citations=[],
        context=context,
    )


def blocked_decision():
    return RetrievalDecision(can_answer=False, reason="insufficient_legal_basis", top_documents=[], citations=[], context="")


def test_message_over_5000_chars_is_rejected():
    with pytest.raises(MessageTooLongError):
        validate_message_length("问" * 5001)


def test_contract_question_is_out_of_scope():
    result = classify_scope("租房合同押金不退怎么办？")

    assert result.in_scope is False
    assert result.reason == "unsupported_legal_domain"


@pytest.mark.asyncio
async def test_out_of_scope_is_checked_on_every_message():
    service = ChatService(rag_decider=lambda _text: blocked_decision(), llm_client=FakeLlm())

    first = await service.send_message("u1", "c-scope", "我想离婚")
    second = await service.send_message("u1", "c-scope", "租房合同押金不退怎么办？")

    assert first.status == "follow_up_required"
    assert second.status == "out_of_scope"
    assert "官方渠道" in second.answer


@pytest.mark.asyncio
async def test_emergency_risk_bypasses_rag_and_llm():
    service = ChatService(rag_decider=lambda _text: pytest.fail("不应调用 RAG"), llm_client=FakeLlm())

    result = await service.send_message("u1", "c1", "我正在被家暴威胁，孩子也有危险")

    assert result.status == "emergency_guidance"
    assert "110" in result.answer
    assert "法律援助" in result.answer


@pytest.mark.asyncio
async def test_ongoing_danger_alone_is_emergency():
    service = ChatService(rag_decider=lambda _text: pytest.fail("不应调用 RAG"), llm_client=FakeLlm())

    result = await service.send_message("u1", "c-danger", "我现在很危险")

    assert result.status == "emergency_guidance"
    assert "110" in result.answer


@pytest.mark.asyncio
async def test_minor_word_without_risk_is_not_emergency():
    service = ChatService(rag_decider=lambda _text: allowed_decision(), llm_client=FakeLlm())

    result = await service.send_message("u1", "c-minor", "我在北京已婚有孩子，想离婚并分割财产，无危险")

    assert result.status == "answered"


@pytest.mark.asyncio
async def test_minor_with_risk_is_emergency():
    service = ChatService(rag_decider=lambda _text: pytest.fail("不应调用 RAG"), llm_client=FakeLlm())

    result = await service.send_message("u1", "c-minor-risk", "孩子被威胁受伤了")

    assert result.status == "emergency_guidance"


@pytest.mark.asyncio
async def test_follow_up_is_limited_to_three_rounds():
    service = ChatService(rag_decider=lambda _text: allowed_decision(), llm_client=FakeLlm())

    first = await service.send_message("u1", "c2", "我想离婚")
    second = await service.send_message("u1", "c2", "离婚")
    third = await service.send_message("u1", "c2", "离婚")
    fourth = await service.send_message("u1", "c2", "离婚")

    assert [first.status, second.status, third.status] == ["follow_up_required"] * 3
    assert fourth.status != "follow_up_required"
    assert "已知事实" in fourth.answer


@pytest.mark.asyncio
async def test_rag_insufficient_blocks_free_answer():
    llm = FakeLlm()
    service = ChatService(rag_decider=lambda _text: blocked_decision(), llm_client=llm)

    result = await service.send_message("u1", "c3", "离婚财产怎么分？")

    assert result.status == "insufficient_basis"
    assert llm.requests == []


@pytest.mark.asyncio
async def test_default_rag_safely_falls_back_when_unconfigured(monkeypatch):
    monkeypatch.setattr("backend.app.services.chat_service.settings.MILVUS_URI", "")
    llm = FakeLlm()
    service = ChatService(llm_client=llm)

    result = await service.send_message("u1", "c-default", "离婚财产怎么分？")

    assert result.status == "insufficient_basis"
    assert result.reason == "rag_not_configured"
    assert llm.requests == []


@pytest.mark.asyncio
async def test_injected_rag_success_path_calls_llm():
    llm = FakeLlm("注入RAG回答")
    service = ChatService(rag_decider=lambda _text: allowed_decision(), llm_client=llm)

    result = await service.send_message("u1", "c-injected", "离婚财产怎么分？")

    assert result.status == "answered"
    assert result.answer == "注入RAG回答"
    assert len(llm.requests) == 1


@pytest.mark.asyncio
async def test_account_minute_rate_limit_is_enforced():
    service = ChatService(rag_decider=lambda _text: blocked_decision(), llm_client=FakeLlm())

    results = [await service.send_message("u-rate", f"c-rate-{index}", "离婚财产怎么分？") for index in range(6)]

    assert results[-1].status == "rate_limited"
    assert results[-1].reason == "account_minute_rate_limit"


@pytest.mark.asyncio
async def test_account_day_rate_limit_is_enforced_when_minute_window_has_room():
    service = ChatService(rag_decider=lambda _text: blocked_decision(), llm_client=FakeLlm())
    now = time.time()
    service._rate_limiter.minute_windows["u-day"] = deque()
    service._rate_limiter.day_windows["u-day"] = deque([now - index for index in range(100)])

    result = await service.send_message("u-day", "c-day", "离婚财产怎么分？")

    assert result.status == "rate_limited"
    assert result.reason == "account_day_rate_limit"


@pytest.mark.asyncio
async def test_regenerate_reruns_checks_and_has_limit():
    calls = {"count": 0}

    def rag(_text):
        calls["count"] += 1
        return allowed_decision()

    service = ChatService(rag_decider=rag, llm_client=FakeLlm("新回答"))
    original = await service.send_message("u1", "c4", "我在北京已婚有孩子，想离婚并分割财产，无危险")

    for _ in range(3):
        regenerated = await service.regenerate_answer("u1", original.message_id)

    assert regenerated.answer == "新回答"
    assert calls["count"] == 4
    with pytest.raises(RegenerationLimitError):
        await service.regenerate_answer("u1", original.message_id)


@pytest.mark.asyncio
async def test_fact_correction_marks_old_answer_and_creates_auditable_new_answer():
    service = ChatService(rag_decider=lambda _text: allowed_decision(), llm_client=FakeLlm("已按更正事实回答"))
    old = await service.send_message("u1", "c5", "我在北京已婚有孩子，想离婚并分割财产，无危险")
    old_answer = service.messages[old.message_id].answer

    result = await service.correct_facts("u1", "c5", FactCorrection(message_id=old.message_id, correction="更正：没有孩子", confirmed=True))

    assert result.status == "answered"
    assert result.answer == "已按更正事实回答"
    assert result.message_id != old.message_id
    assert service.messages[old.message_id].corrected is True
    assert service.messages[old.message_id].answer == old_answer
    assert service.messages[result.message_id].source_message_id == old.message_id


@pytest.mark.asyncio
async def test_llm_context_is_truncated_per_fragment_and_total():
    long_context = "A" * 1200 + "\n\n" + "B" * 1200 + "\n\n" + "C" * 1200 + "\n\n" + "D" * 1200
    llm = FakeLlm()
    service = ChatService(rag_decider=lambda _text: allowed_decision(long_context), llm_client=llm)

    result = await service.send_message("u1", "c-context", "离婚财产怎么分？")
    content = llm.requests[0].messages[1]["content"]

    assert result.status == "answered"
    assert len(content.split("可用资料摘要：", 1)[1]) <= 2400
    assert "A" * 900 not in content


@pytest.mark.asyncio
async def test_restart_clears_conversation_state():
    service = ChatService(rag_decider=lambda _text: allowed_decision(), llm_client=FakeLlm())
    await service.send_message("u1", "c6", "我想离婚")

    service.restart_consultation("u1", "c6")
    result = await service.send_message("u1", "c6", "我想离婚")

    assert result.status == "follow_up_required"
    assert service.conversations["c6"].follow_up_count == 1
