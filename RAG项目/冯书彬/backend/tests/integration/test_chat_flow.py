import pytest

from backend.app.rag.pipeline import RetrievalDecision
from backend.app.schemas.chat import FactCorrection
from backend.app.services.chat_service import ChatService


class FakeLlm:
    def __init__(self):
        self.count = 0

    async def stream_chat(self, request):
        self.count += 1
        yield f"回答{self.count}"


@pytest.mark.asyncio
async def test_chat_flow_followup_answer_correction_and_restart():
    service = ChatService(rag_decider=lambda _text: RetrievalDecision(True, "ok", [], [], "婚姻家庭依据"), llm_client=FakeLlm())

    follow_up = await service.send_message("u-flow", "c-flow", "我想离婚")
    answered = await service.send_message("u-flow", "c-flow", "我在上海，已婚，有孩子，想争取抚养权，无危险")
    corrected = await service.correct_facts("u-flow", "c-flow", FactCorrection(message_id=answered.message_id, correction="更正：没有共同债务", confirmed=True))

    service.restart_consultation("u-flow", "c-flow")
    restarted = await service.send_message("u-flow", "c-flow", "我想离婚")

    assert follow_up.status == "follow_up_required"
    assert answered.status == "answered"
    assert corrected.status == "answered"
    assert restarted.status == "follow_up_required"
