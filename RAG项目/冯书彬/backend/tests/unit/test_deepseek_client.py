import json
from types import TracebackType

import httpx
import pytest

from backend.app.llm.base import LlmRequest
from backend.app.llm.deepseek_client import DeepSeekClient, DeepSeekUnavailableError


class _FakeStreamResponse:
    def __init__(self, status_code: int = 200, lines: list[str] | None = None, json_body: dict | None = None):
        self.status_code = status_code
        self._lines = lines or []
        self._json_body = json_body or {}
        self._request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    @property
    def request(self):
        return self._request

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = httpx.Response(self.status_code, json=self._json_body, request=self._request)
            raise httpx.HTTPStatusError("bad status", request=self._request, response=response)

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeAsyncClient:
    response: _FakeStreamResponse | None = None
    raised: BaseException | None = None
    captured_json: dict | None = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def stream(self, method: str, url: str, json: dict, headers: dict):
        type(self).captured_json = json
        if type(self).raised:
            raise type(self).raised
        return type(self).response


def _patch_http_client(monkeypatch, response: _FakeStreamResponse | None = None, raised: BaseException | None = None):
    _FakeAsyncClient.response = response
    _FakeAsyncClient.raised = raised
    _FakeAsyncClient.captured_json = None
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)


@pytest.mark.asyncio
async def test_deepseek_failure_raises_unavailable(monkeypatch):
    _patch_http_client(monkeypatch, _FakeStreamResponse(status_code=500, json_body={"error": "server"}))
    client = DeepSeekClient(api_key="test", base_url="https://api.deepseek.com")

    with pytest.raises(DeepSeekUnavailableError):
        async for _chunk in client.stream_chat(request=LlmRequest(messages=[])):
            pass


@pytest.mark.asyncio
async def test_deepseek_network_error_raises_unavailable(monkeypatch):
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    _patch_http_client(monkeypatch, raised=httpx.ConnectError("network down", request=request))
    client = DeepSeekClient(api_key="test", base_url="https://api.deepseek.com")

    with pytest.raises(DeepSeekUnavailableError):
        async for _chunk in client.stream_chat(request=LlmRequest(messages=[], trace_id="trace-network")):
            pass


@pytest.mark.asyncio
async def test_deepseek_streams_content_after_accepted(monkeypatch):
    payload = {"choices": [{"delta": {"content": "你好"}}]}
    lines = [f"data: {json.dumps(payload, ensure_ascii=False)}", "data: [DONE]"]
    _patch_http_client(monkeypatch, _FakeStreamResponse(status_code=200, lines=lines))
    client = DeepSeekClient(api_key="test", base_url="https://api.deepseek.com", model="deepseek-chat")

    chunks = []
    async for chunk in client.stream_chat(request=LlmRequest(messages=[{"role": "user", "content": "你好"}], trace_id="trace-ok")):
        chunks.append(chunk)

    assert chunks == ["你好"]
    assert _FakeAsyncClient.captured_json == {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": True,
    }


@pytest.mark.asyncio
async def test_deepseek_malformed_stream_raises_unavailable(monkeypatch):
    _patch_http_client(monkeypatch, _FakeStreamResponse(status_code=200, lines=["data: {bad json}"]))
    client = DeepSeekClient(api_key="test", base_url="https://api.deepseek.com")

    with pytest.raises(DeepSeekUnavailableError):
        async for _chunk in client.stream_chat(request=LlmRequest(messages=[])):
            pass


@pytest.mark.asyncio
async def test_deepseek_does_not_yield_partial_content_when_stream_fails(monkeypatch):
    payload = {"choices": [{"delta": {"content": "半截回答"}}]}
    lines = [f"data: {json.dumps(payload, ensure_ascii=False)}", "data: {bad json}"]
    _patch_http_client(monkeypatch, _FakeStreamResponse(status_code=200, lines=lines))
    client = DeepSeekClient(api_key="test", base_url="https://api.deepseek.com")

    chunks = []
    with pytest.raises(DeepSeekUnavailableError):
        async for chunk in client.stream_chat(request=LlmRequest(messages=[], trace_id="trace-partial")):
            chunks.append(chunk)

    assert chunks == []
