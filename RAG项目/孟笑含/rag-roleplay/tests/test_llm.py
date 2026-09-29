# -*- coding: utf-8 -*-
"""LLM 兼容层：用 MockTransport 模拟 OpenAI 兼容服务端，验证请求格式与解析。"""
import json

import httpx
import pytest

from app.core.llm import OpenAILLM


def make_llm(handler):
    """用自定义 transport 构造 LLM，不发真实网络请求。"""
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return OpenAILLM(
        base_url="http://fake.local/v1",
        api_key="test-key",
        model="test-model",
        http_client=client,
    )


def json_response(payload):
    return httpx.Response(200, json=payload, request=httpx.Request("POST", "http://fake.local/v1/chat/completions"))


async def test_chat_sends_messages_and_returns_content():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        return json_response(
            {"choices": [{"message": {"role": "assistant", "content": "你好呀"}}]}
        )

    llm = make_llm(handler)
    reply = await llm.chat(
        [{"role": "system", "content": "人设"}, {"role": "user", "content": "你好"}]
    )

    assert reply == "你好呀"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["messages"][-1] == {"role": "user", "content": "你好"}
    assert seen["body"]["stream"] is False


async def test_chat_returns_empty_string_when_content_missing():
    def handler(request: httpx.Request):
        return json_response({"choices": [{"message": {"role": "assistant"}}]})

    llm = make_llm(handler)
    assert await llm.chat([{"role": "user", "content": "hi"}]) == ""


async def test_chat_stream_yields_chunks_in_order():
    def handler(request: httpx.Request):
        assert json.loads(request.content)["stream"] is True
        body = (
            'data: {"choices":[{"delta":{"content":"你"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"好"}}]}\n\n'
            'data: {"choices":[{"delta":{}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    llm = make_llm(handler)
    chunks = [c async for c in llm.chat_stream([{"role": "user", "content": "hi"}])]

    assert chunks == ["你", "好"]


async def test_chat_stream_propagates_api_error():
    def handler(request: httpx.Request):
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    llm = make_llm(handler)
    with pytest.raises(Exception):
        async for _ in llm.chat_stream([{"role": "user", "content": "hi"}]):
            pass


# ---------- MockLLM（压测开关，与 OpenAILLM 同模块） ----------

async def test_mock_llm_chat_returns_preset_reply():
    from app.core.llm import MockLLM

    llm = MockLLM()
    reply = await llm.chat([{"role": "user", "content": "你好"}])
    assert isinstance(reply, str) and reply


async def test_mock_llm_stream_yields_chunks_then_completes():
    from app.core.llm import MockLLM

    llm = MockLLM()
    chunks = [c async for c in llm.chat_stream([{"role": "user", "content": "你好"}])]
    assert "".join(chunks) == llm.reply
    assert len(chunks) >= 1


def test_get_llm_returns_mock_when_enabled(monkeypatch):
    import asyncio

    from app.api.deps import get_llm
    from app.config import settings
    from app.core.llm import MockLLM

    monkeypatch.setattr(settings, "llm_mock", True)
    llm = asyncio.run(get_llm())
    assert isinstance(llm, MockLLM)

    monkeypatch.setattr(settings, "llm_mock", False)
    llm = asyncio.run(get_llm())
    assert not isinstance(llm, MockLLM)


# ---------- QueryRewriter（查询改写扩写） ----------

async def test_rewriter_parses_lines_and_caps_two_variants():
    from app.core.llm import QueryRewriter

    class LineLLM:
        async def chat(self, messages):
            return "变体一\n变体二\n变体三"

    rewriter = QueryRewriter(LineLLM())
    variants = await rewriter.rewrite("原始问题")
    assert variants == ["变体一", "变体二"]


async def test_rewriter_skips_blank_lines():
    from app.core.llm import QueryRewriter

    class LineLLM:
        async def chat(self, messages):
            return "\n\n变体一\n\n"

    rewriter = QueryRewriter(LineLLM())
    assert await rewriter.rewrite("q") == ["变体一"]


async def test_rewriter_returns_empty_on_blank_reply():
    from app.core.llm import QueryRewriter

    class LineLLM:
        async def chat(self, messages):
            return "   \n  "

    rewriter = QueryRewriter(LineLLM())
    assert await rewriter.rewrite("q") == []


def test_rewriter_sends_query_instruction_prompt():
    from app.core.llm import QueryRewriter
    import asyncio

    class EchoLLM:
        def __init__(self):
            self.last = None

        async def chat(self, messages):
            self.last = messages
            return "改写的查询"

    llm = EchoLLM()
    asyncio.run(QueryRewriter(llm).rewrite("高血压怎么办"))
    assert llm.last[0]["role"] == "system"
    assert "检索" in llm.last[0]["content"]
    assert llm.last[-1] == {"role": "user", "content": "高血压怎么办"}
