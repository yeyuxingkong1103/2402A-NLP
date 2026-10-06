from backend.app.api.v1.chat import chat_service
from backend.app.core.security import create_access_token
from backend.app.rag.pipeline import RetrievalDecision


def _auth_headers(user_id: str = "u-api") -> dict[str, str]:
    token = create_access_token(user_id, "session-chat-test")
    return {"Authorization": f"Bearer {token}"}


class FakeLlm:
    async def stream_chat(self, request):
        yield "接口回答"


def test_chat_message_endpoint_returns_answer(client):
    chat_service.reset_for_tests(lambda _text: RetrievalDecision(True, "ok", [], [], "婚姻家庭依据"), FakeLlm())

    response = client.post(
        "/api/v1/chat/conversations/api-c1/messages",
        headers=_auth_headers(),
        json={"user_id": "u-api", "text": "我在北京已婚有孩子，想离婚并分割财产，无危险"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "answered"
    assert response.json()["answer"] == "接口回答"


def test_chat_scope_endpoint_rejects_contract_case(client):
    response = client.post(
        "/api/v1/chat/conversations/api-c2/messages",
        headers=_auth_headers(),
        json={"user_id": "u-api", "text": "租房合同押金不退怎么办？"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "out_of_scope"
    assert "官方渠道" in response.json()["answer"]


def test_chat_regenerate_endpoint_returns_answer(client):
    chat_service.reset_for_tests(lambda _text: RetrievalDecision(True, "ok", [], [], "婚姻家庭依据"), FakeLlm())
    created = client.post(
        "/api/v1/chat/conversations/api-c3/messages",
        headers=_auth_headers(),
        json={"user_id": "u-api", "text": "我在北京已婚有孩子，想离婚并分割财产，无危险"},
    ).json()

    response = client.post(
        f"/api/v1/chat/messages/{created['message_id']}/regenerate",
        headers=_auth_headers(),
        json={"user_id": "u-api"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "answered"
