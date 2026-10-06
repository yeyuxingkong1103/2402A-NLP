from collections.abc import AsyncIterator

import httpx
from openai import AsyncOpenAI

from app.config import get_settings


class LLMClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def build_llm_from_character(character) -> LLMClient:
    settings = get_settings()
    base_url = character.base_url or settings.llm_base_url
    return LLMClient(
        model=character.model_name,
        base_url=base_url,
        api_key=settings.llm_api_key,
        temperature=character.temperature,
        top_p=character.top_p,
        max_tokens=character.max_tokens,
    )
