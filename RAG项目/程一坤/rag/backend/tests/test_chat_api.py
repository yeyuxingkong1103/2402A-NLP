"""问答 SSE 接口契约测试。"""

import asyncio
import json
import time

import httpx
import pytest

from app.auth.current_user import get_current_user
from app.auth.session_store import SessionUser
from app.chat.result import ChatResult
from app.main import app


ANSWER = "第一句说明经济补偿的计算规则。[1]\n第二句提示结合实际工作年限。[2]\n\n内容仅供法律信息参考，不能替代律师出具的正式法律意见。"


class RecordingChatService:
    def __init__(self) -> None:
        self.calls = []

    def chat(self, question: str, **kwargs) -> ChatResult:
        self.calls.append((question, kwargs))
        return ChatResult(
            answer=ANSWER,
            sources=[
                {
                    "chunk_id": "chunk-47",
                    "law_name": "中华人民共和国劳动合同法",
                    "article_number": "第四十七条",
                    "paragraph_number": None,
                    "page": None,
                },
                {
                    "chunk_id": "chunk-27",
                    "law_name": "中华人民共和国劳动合同法实施条例",
                    "article_number": "第二十七条",
                    "paragraph_number": None,
                    "page": None,
                },
            ],
            refused=False,
            guardrail_applied=["citation_check_passed", "guardrails_applied"],
            retrieval_stats={"reranked_count": 2},
        )


class FailingChatService:
    def chat(self, question: str, **kwargs) -> ChatResult:
        raise RuntimeError("mysql+pymysql://user:secret@host/db system prompt")


class FakeChatStore:
    """内存版会话存储替身：/stream 端点依赖的两个方法的最小实现。"""

    def __init__(self) -> None:
        # 记录每次调用参数，供断言检查归属与落库行为
        self.created = []
        self.appended = []

    def get_or_create_session(
        self,
        user_id: str,
        session_key: str,
        character_id: str = "legal-assistant",
        title: str | None = None,
    ):
        # 与真实实现同签名；默认全部放行（跨用户拒绝场景在会话测试文件覆盖）
        self.created.append((user_id, session_key, character_id, title))

    def append_message_pair(
        self,
        session_key: str,
        user_id: str,
        question: str,
        answer: str,
        sources: list[dict],
        model: str | None = None,
        message_id: str | None = None,
    ) -> None:
        # 记录落库调用（一问一答成对写入）
        self.appended.append((session_key, user_id, question, answer, sources))


class FakeShortTermMemory:
    """内存版 Redis 短期记忆替身：只实现 append_message。"""

    def __init__(self) -> None:
        # 记录写回的 (user_id, session_id, message) 三元组
        self.messages = []

    def append_message(self, user_id: str, session_id: str, message: dict) -> None:
        self.messages.append((user_id, session_id, message))


def parse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        name = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        events.append((name, data))
    return events


async def post_stream(payload: dict, *, authorization: str | None = "Bearer token"):
    headers = {"Authorization": authorization} if authorization else {}
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/api/v1/chat/stream",
            json=payload,
            headers=headers,
        ) as response:
            return response.status_code, response.headers, await response.aread()


@pytest.fixture
def authenticated_user():
    app.dependency_overrides[get_current_user] = lambda: SessionUser(
        user_id="authenticated-user",
        is_admin=False,
    )
    # 任务 5.4：注入内存版存储替身，避免测试触达真实 MySQL / Redis
    app.state.chat_session_store = FakeChatStore()
    app.state.short_term_memory = FakeShortTermMemory()
    yield
    app.dependency_overrides.pop(get_current_user, None)
    if hasattr(app.state, "chat_service"):
        del app.state.chat_service
    if hasattr(app.state, "chat_session_store"):
        del app.state.chat_session_store
    if hasattr(app.state, "short_term_memory"):
        del app.state.short_term_memory


def test_chat_emits_contract_events_and_ignores_body_user_id(authenticated_user) -> None:
    service = RecordingChatService()
    app.state.chat_service = service
    payload = {
        "session_id": "session-1",
        "character_id": "legal-assistant",
        "message": "经济补偿怎么算？",
        "user_id": "forged-user",
        "options": {
            "top_k": 2,
            "jurisdiction": "中国大陆",
            "as_of_date": "2026-09-14",
            "document_types": ["law", "administrative_regulation"],
        },
    }

    status, headers, content = asyncio.run(post_stream(payload))

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert headers["cache-control"] == "no-cache"
    events = parse_events(content.decode("utf-8"))
    names = [name for name, _ in events]
    assert names[0] == "message_start"
    assert names[-1] == "message_end"
    assert names == [
        "message_start",
        *(["token"] * names.count("token")),
        "citation",
        "citation",
        "message_end",
    ]
    message_id = events[0][1]["message_id"]
    assert message_id.startswith("message_")
    assert "request_id" in events[0][1]
    assert "".join(data["text"] for name, data in events if name == "token") == ANSWER
    citations = [data for name, data in events if name == "citation"]
    assert set(citations[0]) == {
        "chunk_id",
        "law_name",
        "article_number",
        "paragraph_number",
        "page",
    }
    assert citations[0]["page"] is None
    end = events[-1][1]
    assert end["finish_reason"] == "stop"
    assert end["usage"] is None
    assert isinstance(end["elapsed_seconds"], (int, float))
    assert end["elapsed_seconds"] >= 0

    question, kwargs = service.calls[0]
    assert question == "经济补偿怎么算？"
    assert kwargs["user_id"] == "authenticated-user"
    assert kwargs["session_id"] == "session-1"
    assert kwargs["request_id"].startswith("req_")


def test_chat_disables_query_rewrite_by_omitting_session_context(
    authenticated_user,
) -> None:
    service = RecordingChatService()
    app.state.chat_service = service
    payload = {
        "session_id": "session-1",
        "message": "那 12 期呢",
        "options": {"enable_query_rewrite": False},
    }

    status, _, _ = asyncio.run(post_stream(payload))

    assert status == 200
    assert service.calls[0][1]["session_id"] is None


def test_chat_emits_only_start_then_error_on_failure(authenticated_user) -> None:
    app.state.chat_service = FailingChatService()
    payload = {"session_id": "session-1", "message": "问题"}

    status, _, content = asyncio.run(post_stream(payload))

    assert status == 200
    events = parse_events(content.decode("utf-8"))
    assert [name for name, _ in events] == ["message_start", "error"]
    error = events[-1][1]
    assert error["code"] == 50001
    assert error["message"] == "模型服务暂时不可用"
    assert error["retryable"] is True
    serialized = content.decode("utf-8").lower()
    assert "message_end" not in serialized
    assert "mysql+pymysql" not in serialized
    assert "secret" not in serialized
    assert "system prompt" not in serialized


def test_chat_requires_authentication() -> None:
    status, _, content = asyncio.run(
        post_stream({"session_id": "session-1", "message": "问题"}, authorization=None)
    )

    assert status == 401
    assert json.loads(content)["code"] == 40100


def test_chat_service_runs_off_event_loop() -> None:
    from app.api.chat import ChatStreamRequest, _event_stream

    class BlockingService(RecordingChatService):
        def chat(self, question: str, **kwargs) -> ChatResult:
            time.sleep(0.15)
            return super().chat(question, **kwargs)

    async def measure_probe_delay() -> float:
        stream = _event_stream(
            service=BlockingService(),
            payload=ChatStreamRequest(session_id="session-1", message="问题"),
            user_id="user-1",
            message_id="message-test",
            request_id="req-test",
        )
        await anext(stream)
        started = time.perf_counter()
        consumer = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.01)
        delay = time.perf_counter() - started
        await consumer
        await stream.aclose()
        return delay

    assert asyncio.run(measure_probe_delay()) < 0.08


def test_stable_chunks_reconstruct_answer_without_splitting_characters() -> None:
    from app.api.chat import split_answer_chunks

    chunks = split_answer_chunks(ANSWER, max_chars=18)

    assert "".join(chunks) == ANSWER
    assert all(chunk.encode("utf-8").decode("utf-8") == chunk for chunk in chunks)
    assert all(
        chunk.endswith(("。", "！", "？", "!", "?", "\n"))
        for chunk in chunks[:-1]
    )


# ---------- 真流式路径（service 提供 chat_stream 时的契约） ----------


class StreamingChatService:
    """流式替身：逐块 yield，最终 result 经 stream_context 回传（与真实 ChatService 同约定）。"""

    def __init__(self, chunks: list[str], replace: str | None = None) -> None:
        self.chunks = chunks
        self.replace = replace
        self.streamed_calls = []

    def chat_stream(self, question: str, *, stream_context, **kwargs):
        self.streamed_calls.append((question, kwargs))
        answer = "".join(self.chunks)
        sources = [
            {
                "chunk_id": "chunk-47",
                "law_name": "中华人民共和国劳动合同法",
                "article_number": "第四十七条",
                "paragraph_number": None,
                "page": None,
            }
        ]
        if self.replace is not None:
            answer = self.replace
        from app.chat.result import ChatResult as _R

        stream_context["result"] = _R(
            answer=answer,
            sources=sources,
            refused=False,
            guardrail_applied=["guardrails_applied"],
            retrieval_stats={"reranked_count": 1},
        )
        yield from self.chunks
        if self.replace is not None:
            stream_context["replace"] = self.replace


def test_chat_stream_emits_tokens_incrementally(authenticated_user) -> None:
    """流式服务：token 帧逐块下发，citation/message_end 照常收尾，落库用全文。"""
    chunks = ["第一块，", "第二块[1]。\n\n内容仅供法律信息参考，不能替代律师出具的正式法律意见。"]
    service = StreamingChatService(chunks)
    app.state.chat_service = service
    status, _, body = asyncio.run(post_stream({
        "session_id": "sess-1", "message": "问题",
    }))
    assert status == 200
    events = parse_events(body.decode("utf-8"))
    names = [name for name, _ in events]
    assert names[0] == "message_start"
    assert names[1:3] == ["token", "token"]
    assert events[1][1]["text"] == chunks[0]  # 第一个 token 只含第一块，不是全文
    assert "citation" in names and names[-1] == "message_end"


def test_chat_stream_emits_replace_on_untrusted_answer(authenticated_user) -> None:
    """方案 A：护栏判定不可信 → token 已发但随后发 replace，前端整段替换。"""
    service = StreamingChatService(["看起来", "像回答的内容"], replace="兜底风险提示全文")
    app.state.chat_service = service
    store: FakeChatStore = app.state.chat_session_store
    status, _, body = asyncio.run(post_stream({
        "session_id": "sess-1", "message": "问题",
    }))
    assert status == 200
    events = parse_events(body.decode("utf-8"))
    names = [name for name, _ in events]
    assert "replace" in names
    assert names.index("replace") < names.index("message_end")
    replace_text = next(data for name, data in events if name == "replace")
    assert replace_text["text"] == "兜底风险提示全文"
    # 落库的是替换后的最终答案，不是已作废的流式文本
    assert store.appended[0][3] == "兜底风险提示全文"
