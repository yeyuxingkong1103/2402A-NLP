"""流式 LLM 客户端测试（全部用注入替身，不联网）。

OpenAI 兼容接口的流式协议：请求体带 "stream": true，
响应体是逐行 SSE：data: {"choices":[{"delta":{"content":"..."}}]}，以 data: [DONE] 结束。
"""

from typing import Any

import pytest

from app.models.llm import LlmApiError, OpenAiCompatibleChatClient


def make_sse_transport(chunks: list[str]) -> Any:
    """构造假的流式传输函数：返回可逐行迭代的 SSE 字节响应。"""

    lines = []
    for text in chunks:
        payload = '{"choices":[{"delta":{"content":"%s"}}]}' % text
        lines.append(f"data: {payload}\n\n".encode("utf-8"))
    lines.append(b"data: [DONE]\n\n")

    class FakeResponse:
        def __iter__(self):
            return iter(lines)

        def close(self) -> None:
            pass

    def transport(url: str, headers: dict, payload: bytes, timeout: float) -> Any:
        FakeResponse.captured_url = url
        FakeResponse.captured_payload = payload
        return FakeResponse()

    return transport


class TestStreamChat:
    def test_stream_yields_deltas_in_order(self) -> None:
        """逐块产出增量内容，拼接后等于完整回答。"""
        client = OpenAiCompatibleChatClient(
            api_base_url="https://llm.example/v1",
            api_key="k",
            model="m",
            transport=lambda *a, **k: {},  # chat() 不用；stream 用注入的 stream_transport
        )
        client.stream_transport = make_sse_transport(["用人单位", "应当按月", "支付工资[1]。"])
        parts = list(client.stream_chat("系统", "问题"))
        assert parts == ["用人单位", "应当按月", "支付工资[1]。"]

    def test_stream_request_has_stream_true(self) -> None:
        """请求体必须带 stream:true；走 /chat/completions 端点。"""
        captured: dict[str, Any] = {}

        def transport(url: str, headers: dict, payload: bytes, timeout: float) -> Any:
            captured["url"] = url
            captured["payload"] = payload
            return iter([])

        client = OpenAiCompatibleChatClient(
            api_base_url="https://llm.example/v1",
            api_key="k",
            model="m",
        )
        client.stream_transport = transport
        list(client.stream_chat("s", "u"))
        assert captured["url"].endswith("/chat/completions")
        assert b'"stream":true' in captured["payload"] or b'"stream": true' in captured["payload"]

    def test_stream_skips_empty_deltas_and_role_deltas(self) -> None:
        """delta 无 content（如 role 帧）时不产出空串。"""
        lines = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'.encode("utf-8"),
            'data: {"choices":[{"delta":{"content":"你好"}}]}\n\n'.encode("utf-8"),
            b'data: {"choices":[{"delta":{}}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        class FakeResponse:
            def __iter__(self):
                return iter(lines)

        client = OpenAiCompatibleChatClient(
            api_base_url="https://llm.example/v1", api_key="k", model="m"
        )
        client.stream_transport = lambda *a, **k: FakeResponse()
        assert list(client.stream_chat("s", "u")) == ["你好"]

    def test_stream_missing_config_raises(self) -> None:
        client = OpenAiCompatibleChatClient(api_base_url="", api_key="", model="")
        with pytest.raises(LlmApiError):
            list(client.stream_chat("s", "u"))

    def test_stream_http_error_raises(self) -> None:
        import urllib.error

        def transport(url: str, headers: dict, payload: bytes, timeout: float) -> Any:
            raise urllib.error.HTTPError(url, 502, "Bad Gateway", {}, None)

        client = OpenAiCompatibleChatClient(
            api_base_url="https://llm.example/v1", api_key="k", model="m"
        )
        client.stream_transport = transport
        with pytest.raises(LlmApiError):
            list(client.stream_chat("s", "u"))

    def test_stream_invalid_json_line_is_skipped(self) -> None:
        """服务端偶发脏行不应中断整个流。"""
        lines = [
            'data: {"choices":[{"delta":{"content":"片段"}}]}\n\n'.encode("utf-8"),
            b"data: not-json\n\n",
            b"data: [DONE]\n\n",
        ]

        class FakeResponse:
            def __iter__(self):
                return iter(lines)

        client = OpenAiCompatibleChatClient(
            api_base_url="https://llm.example/v1", api_key="k", model="m"
        )
        client.stream_transport = lambda *a, **k: FakeResponse()
        assert list(client.stream_chat("s", "u")) == ["片段"]
