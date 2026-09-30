"""会话管理 REST 接口契约测试（任务 5.4）。

覆盖范围（与实施清单一一对应）：
1. 创建会话返回 201 + 服务端生成的会话 ID + 标题清洗
2. /chat/stream 成功流触发问答成对落库与短期记忆回写
3. 历史消息接口按正序返回且携带 citations
4. 跨用户访问他人会话统一 404（不暴露存在性）
5. 未登录访问会话接口统一 401
6. 失败的问答流不触发落库与记忆回写

所有存储依赖均为内存替身，测试不触达真实 MySQL / Redis。
"""

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest

from app.auth.current_user import get_current_user
from app.auth.session_store import SessionUser
from app.chat.chat_store import SessionAccessDenied
from app.chat.result import ChatResult
from app.main import app


# 与 test_chat_api.py 相同的回答样例（含免责声明与引用标记）
ANSWER = "第一句说明经济补偿的计算规则。[1]\n第二句提示结合实际工作年限。[2]\n\n内容仅供法律信息参考，不能替代律师出具的正式法律意见。"

# 模拟读取层（chat_history.list_messages）返回的正序历史消息
HISTORY = [
    {
        "message_id": None,
        "role": "user",
        "content": "经济补偿怎么算？",
        "citations": None,
        "model": None,
        "created_at": datetime(2026, 9, 18, 10, 0, 0),
    },
    {
        "message_id": "message_abc123def456",
        "role": "assistant",
        "content": ANSWER,
        "citations": [
            {
                "chunk_id": "chunk-47",
                "law_name": "中华人民共和国劳动合同法",
                "article_number": "第四十七条",
            }
        ],
        "model": None,
        "created_at": datetime(2026, 9, 18, 10, 0, 5),
    },
]


class RecordingChatService:
    """成功路径的问答服务替身：返回固定的回答与引用。"""

    def __init__(self) -> None:
        # 记录每次调用参数（本文件只关心调用发生与否）
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
                }
            ],
            refused=False,
            guardrail_applied=["guardrails_applied"],
            retrieval_stats={"reranked_count": 1},
        )


class FailingChatService:
    """失败路径的问答服务替身：模拟模型调用抛异常。"""

    def chat(self, question: str, **kwargs) -> ChatResult:
        raise RuntimeError("模型调用失败")


class FakeChatStore:
    """内存版会话写入存储替身（与 app.chat.chat_store 接口对齐）。"""

    def __init__(self) -> None:
        # 建档调用记录：(user_id, session_key, character_id, title)
        self.created = []
        # 落库调用记录：(session_key, user_id, question, answer, sources)
        self.appended = []
        # 删除调用记录：(user_id, session_key)
        self.deleted = []
        # 模拟归属失败：非 None 时 delete_session 抛出该异常
        self.delete_error: Exception | None = None

    def get_or_create_session(
        self,
        user_id: str,
        session_key: str,
        character_id: str = "legal-assistant",
        title: str | None = None,
    ):
        # 默认全部放行；跨用户拒绝场景由 FakeChatHistoryReader 单独模拟
        self.created.append((user_id, session_key, character_id, title))
        # 返回与 ChatSession 字段同名的简单记录（创建接口要读 session_key/title/created_at）
        return SimpleNamespace(
            session_key=session_key,
            character_id=character_id,
            title=title or "新会话",
            created_at=datetime.now(),
        )

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
        self.appended.append(
            (session_key, user_id, question, answer, sources, message_id)
        )

    def delete_session(self, user_id: str, session_key: str) -> None:
        # 与真实实现同签名；delete_error 非空时模拟"会话不存在或不属于本人"
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append((user_id, session_key))


class FakeShortTermMemory:
    """内存版 Redis 短期记忆替身（与 app.memory.short_term 接口对齐）。"""

    def __init__(self) -> None:
        # 写回记录：(user_id, session_id, message)
        self.messages = []
        # 会话记忆清理记录：(user_id, session_id)
        self.deleted_sessions = []

    def append_message(self, user_id: str, session_id: str, message: dict) -> None:
        self.messages.append((user_id, session_id, message))

    def delete_session_memory(self, user_id: str, session_id: str) -> None:
        self.deleted_sessions.append((user_id, session_id))


class FakeChatHistoryReader:
    """内存版会话读取存储替身（与 app.chat.chat_history 接口对齐）。"""

    def __init__(self) -> None:
        # list_sessions 的返回值（默认空列表）
        self.sessions_result = []
        # list_messages 的返回值（默认空列表）
        self.messages_result = []
        # 模拟归属失败：非 None 时 list_messages 抛出该异常
        self.messages_error: Exception | None = None
        # 记录读取层实际收到的分页参数（验证 API 层透传）
        self.sessions_page_args = []
        self.messages_page_args = []

    def list_sessions(
        self, user_id: str, page: int = 1, page_size: int = 20
    ) -> list[dict]:
        self.sessions_page_args.append((page, page_size))
        return self.sessions_result

    def list_messages(
        self, user_id: str, session_key: str, page: int = 1, page_size: int = 20
    ) -> list[dict]:
        self.messages_page_args.append((page, page_size))
        if self.messages_error is not None:
            raise self.messages_error
        return self.messages_result


async def request_api(method: str, path: str, *, payload: dict | None = None, authorization: str | None = "Bearer token"):
    """以 httpx.ASGITransport 直打 app 的轻量请求辅助。"""
    headers = {"Authorization": authorization} if authorization else {}
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, json=payload, headers=headers)


@pytest.fixture
def session_env():
    """注入认证替身与全部内存存储替身；测试结束统一清理。"""
    app.dependency_overrides[get_current_user] = lambda: SessionUser(
        user_id="user-a",
        is_admin=False,
    )
    store = FakeChatStore()
    reader = FakeChatHistoryReader()
    memory = FakeShortTermMemory()
    app.state.chat_session_store = store
    app.state.chat_history_reader = reader
    app.state.short_term_memory = memory
    yield {"store": store, "reader": reader, "memory": memory}
    app.dependency_overrides.pop(get_current_user, None)
    for attr in ("chat_session_store", "chat_history_reader", "short_term_memory", "chat_service"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


def test_create_session_returns_201_with_clean_title(session_env) -> None:
    # 标题带首尾空白与连续空格，验证服务端做统一清洗
    response = asyncio.run(
        request_api(
            "POST",
            "/api/v1/sessions",
            payload={"title": "  劳动合同   审查  "},
        )
    )
    status = response.status_code
    body = response.json()

    assert status == 201
    # 通用响应结构
    assert body["code"] == 0
    assert body["message"] == "success"
    # 会话 ID 由服务端生成，客户端后续原样回传
    assert body["data"]["session_id"].startswith("session_")
    # 标题清洗：去首尾空白 + 连续空白压成一个空格
    assert body["data"]["title"] == "劳动合同 审查"
    # character_id 缺省为法律助手
    assert body["data"]["character_id"] == "legal-assistant"
    # 建档调用以认证上下文身份发生（请求体无法伪造归属）
    user_id, session_key, character_id, title = session_env["store"].created[0]
    assert user_id == "user-a"
    assert session_key == body["data"]["session_id"]
    assert character_id == "legal-assistant"
    assert title == "劳动合同 审查"


def test_stream_persists_qa_pair_and_memory(session_env) -> None:
    app.state.chat_service = RecordingChatService()
    payload = {"session_id": "session-1", "message": "  经济补偿怎么算？  "}

    response = asyncio.run(request_api("POST", "/api/v1/chat/stream", payload=payload))

    # 流本身成功完成
    assert response.status_code == 200
    # 问答成对落库一次：提问带清洗（与发给服务的一致）、回答为最终文本、引用非空
    assert len(session_env["store"].appended) == 1
    (
        session_key,
        user_id,
        question,
        answer,
        sources,
        persisted_message_id,
    ) = session_env["store"].appended[0]
    assert (session_key, user_id) == ("session-1", "user-a")
    assert question == "经济补偿怎么算？"
    assert answer == ANSWER
    assert sources[0]["chunk_id"] == "chunk-47"
    # 契约 7.6 一致性：落库 assistant 消息携带对外 message_id（message_ 前缀）
    assert persisted_message_id is not None
    assert persisted_message_id.startswith("message_")
    # SSE message_start 事件携带的 message_id 与落库值完全一致（客户端据此关联整轮）
    lines = response.text.splitlines()
    start_payload = lines[lines.index("event: message_start") + 1]
    message_id_in_stream = json.loads(
        start_payload.removeprefix("data: ")
    )["message_id"]
    assert message_id_in_stream == persisted_message_id
    # 短期记忆按 user → assistant 顺序回写两条，结构与 query_rewrite 读取对齐
    assert session_env["memory"].messages == [
        ("user-a", "session-1", {"role": "user", "content": "经济补偿怎么算？"}),
        ("user-a", "session-1", {"role": "assistant", "content": ANSWER}),
    ]


def test_list_messages_returns_history_with_citations(session_env) -> None:
    # 读取层返回正序历史（citations 已反序列化），API 层原样透传
    session_env["reader"].messages_result = list(HISTORY)

    response = asyncio.run(
        request_api("GET", "/api/v1/sessions/session-1/messages")
    )
    status, body = response.status_code, response.json()

    assert status == 200
    assert body["code"] == 0
    # 外层键 items（接口文档 7.6 契约），不再回传 session_id
    messages = body["data"]["items"]
    # 正序：先提问后回答
    assert [item["role"] for item in messages] == ["user", "assistant"]
    # user 消息无引用，assistant 消息带法源列表
    assert messages[0]["citations"] is None
    assert messages[1]["citations"][0]["law_name"] == "中华人民共和国劳动合同法"


def test_messages_of_other_user_returns_404(session_env) -> None:
    # 归属校验失败（不存在或不属于本人）统一映射为 404
    session_env["reader"].messages_error = SessionAccessDenied("会话不存在")

    response = asyncio.run(
        request_api("GET", "/api/v1/sessions/someone-else/messages")
    )
    status, body = response.status_code, response.json()

    assert status == 404
    # 错误码 40001（资源不存在），不区分"不存在"与"存在但不属于你"
    assert body["code"] == 40001
    assert body["message"] == "会话不存在"
    assert body["data"] is None


def test_sessions_require_authentication() -> None:
    # 未提供令牌：创建会话 401
    create_response = asyncio.run(
        request_api("POST", "/api/v1/sessions", payload={"title": "x"}, authorization=None)
    )
    assert create_response.status_code == 401
    assert create_response.json()["code"] == 40100
    # 未提供令牌：拉取历史 401
    history_response = asyncio.run(
        request_api("GET", "/api/v1/sessions/session-1/messages", authorization=None)
    )
    assert history_response.status_code == 401
    assert history_response.json()["code"] == 40100


def test_failed_stream_skips_persist(session_env) -> None:
    app.state.chat_service = FailingChatService()
    payload = {"session_id": "session-1", "message": "问题"}

    response = asyncio.run(request_api("POST", "/api/v1/chat/stream", payload=payload))
    content = response.text

    # 流以 200 返回，但只到 error 事件为止
    assert response.status_code == 200
    assert "event: error" in content
    assert "event: message_end" not in content
    # 失败流不允许触发落库与短期记忆回写
    assert session_env["store"].appended == []
    assert session_env["memory"].messages == []


def test_delete_session_removes_store_and_memory(session_env) -> None:
    # 本人删除自己的会话：MySQL 删除 + Redis 短期记忆清理都被触发
    response = asyncio.run(
        request_api("DELETE", "/api/v1/sessions/session-1")
    )
    status, body = response.status_code, response.json()

    assert status == 200
    assert body["code"] == 0
    assert body["data"] == {"session_id": "session-1", "status": "deleted"}
    # 归属人以认证上下文为准（SQL 层双条件过滤由真实实现负责）
    assert session_env["store"].deleted == [("user-a", "session-1")]
    # 短期记忆（消息列表 + 摘要）同步清理
    assert session_env["memory"].deleted_sessions == [("user-a", "session-1")]


def test_delete_session_of_other_user_returns_404(session_env) -> None:
    # 归属失败（不存在或不属于本人）统一 404，且不触发 Redis 清理
    session_env["store"].delete_error = SessionAccessDenied("会话不存在")

    response = asyncio.run(
        request_api("DELETE", "/api/v1/sessions/someone-else")
    )
    status, body = response.status_code, response.json()

    assert status == 404
    assert body["code"] == 40001
    assert body["message"] == "会话不存在"
    # 归属失败时不允许有任何删除副作用
    assert session_env["store"].deleted == []
    assert session_env["memory"].deleted_sessions == []


def test_delete_session_requires_authentication() -> None:
    # 未提供令牌：删除接口 401
    response = asyncio.run(
        request_api("DELETE", "/api/v1/sessions/session-1", authorization=None)
    )
    assert response.status_code == 401
    assert response.json()["code"] == 40100


def test_list_sessions_pagination_params_pass_through(session_env) -> None:
    # 分页参数原样透传读取层（SQL 分页在读取层完成）；外层键 items（契约 7.5）
    response = asyncio.run(
        request_api("GET", "/api/v1/sessions?page=2&page_size=5")
    )
    status, body = response.status_code, response.json()

    assert status == 200
    assert body["code"] == 0
    assert body["data"]["items"] == []
    assert session_env["reader"].sessions_page_args == [(2, 5)]


def test_list_messages_pagination_params_pass_through(session_env) -> None:
    # 历史消息接口同样透传分页参数（契约 7.6）
    response = asyncio.run(
        request_api("GET", "/api/v1/sessions/session-1/messages?page=3&page_size=10")
    )
    status, body = response.status_code, response.json()

    assert status == 200
    assert body["code"] == 0
    assert body["data"]["items"] == []
    assert session_env["reader"].messages_page_args == [(3, 10)]


@pytest.mark.parametrize(
    "query",
    ["?page=0", "?page=-1", "?page_size=0", "?page_size=101"],
)
def test_pagination_invalid_params_return_422(session_env, query) -> None:
    # 非法分页参数在 Query 校验层直接 422，不触达存储层
    # （注入认证替身：否则认证依赖先于参数校验返回 401）
    response = asyncio.run(request_api("GET", f"/api/v1/sessions{query}"))
    assert response.status_code == 422
