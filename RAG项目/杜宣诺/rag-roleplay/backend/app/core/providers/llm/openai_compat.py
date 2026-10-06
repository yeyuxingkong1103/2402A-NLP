import json
from typing import AsyncIterator

import httpx

from ....config import get_settings


class OpenAICompatLLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        s = get_settings()
        self.base_url = (base_url or s.llm_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else s.llm_api_key
        self.model = model or s.llm_model

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def chat(self, messages: list[dict], **opts) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False, **opts}
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers())
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def chat_stream(self, messages: list[dict], **opts) -> AsyncIterator[str]:
        payload = {"model": self.model, "messages": messages, "stream": True, **opts}
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        delta = json.loads(data)["choices"][0]["delta"]
                        if "content" in delta and delta["content"]:
                            yield delta["content"]
