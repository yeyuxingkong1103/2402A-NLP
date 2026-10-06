"""统一模型调用层。

这里只处理模型连接、消息格式以及同步/异步/流式输出，不包含 RAG、
知识图谱或医疗业务提示词。所有配置和密钥均来自 :mod:`src.config`。
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Iterator
from typing import Literal, Optional, TypedDict

from openai import AsyncOpenAI, OpenAI

from src import config

DEFAULT_SYSTEM_PROMPT = "你是专业的用药咨询助手。"


class LLMChunk(TypedDict):
    type: Literal["reasoning", "content"]
    text: str


def _openai_base_url(base_url: str) -> str:
    """把 DashScope compatible-mode 地址规范到 OpenAI SDK 所需的 v1。"""
    base = (base_url or "").rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if "dashscope.aliyuncs.com/compatible-mode" in base and not base.endswith("/v1"):
        base += "/v1"
    return base


def _user_content(user: str, images: Optional[list[dict]] = None):
    """构造 OpenAI 兼容的文本/图片消息内容。"""
    if not images:
        return user
    content: list[dict] = []
    for image in images:
        url = image.get("url")
        if not url:
            media_type = image.get("media_type", "image/jpeg")
            data = str(image.get("data", ""))
            url = data if data.startswith("data:") else f"data:{media_type};base64,{data}"
        content.append({"type": "image_url", "image_url": {"url": url}})
    if user:
        content.append({"type": "text", "text": user})
    return content


class ModelClient:
    """项目共享的 OpenAI-compatible 模型客户端。"""

    protocol = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.base_url = _openai_base_url(base_url or config.LLM_BASE_URL)
        self.model = model or config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        self.max_tokens = config.LLM_MAX_TOKENS if max_tokens is None else max_tokens
        self.timeout = config.LLM_TIMEOUT if timeout is None else timeout
        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None
        self._client_lock = threading.Lock()

    def is_available(self) -> bool:
        return bool(config.get_llm_enabled() and self.api_key and self.base_url and self.model)

    def _get_client(self) -> OpenAI:
        if not self.is_available():
            raise RuntimeError("LLM 不可用：请检查 LLM_ENABLED、LLM_API_KEY 和模型配置")
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        timeout=self.timeout,
                        max_retries=config.LLM_MAX_RETRIES,
                    )
        return self._client

    def _get_async_client(self) -> AsyncOpenAI:
        if not self.is_available():
            raise RuntimeError("LLM 不可用：请检查 LLM_ENABLED、LLM_API_KEY 和模型配置")
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=config.LLM_MAX_RETRIES,
            )
        return self._async_client

    @staticmethod
    def _messages(system: str, user: str, images: Optional[list[dict]]) -> list[dict]:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": _user_content(user, images)})
        return messages

    def stream(
        self,
        prompt: str = "",
        *,
        system: str = DEFAULT_SYSTEM_PROMPT,
        images: Optional[list[dict]] = None,
        use_thinking: bool = False,
    ) -> Iterator[LLMChunk]:
        """流式调用模型，分别产出思考片段和最终内容片段。"""
        completion = self._get_client().chat.completions.create(
            model=self.model,
            messages=self._messages(system, prompt, images),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"enable_thinking": use_thinking},
            stream=True,
        )
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield {"type": "reasoning", "text": reasoning}
            content = getattr(delta, "content", None)
            if content:
                yield {"type": "content", "text": content}

    def chat(
        self,
        system: str = DEFAULT_SYSTEM_PROMPT,
        user: str = "",
        images: Optional[list[dict]] = None,
        use_thinking: bool = False,
    ) -> str:
        """同步调用；内部复用流式接口并汇总最终内容。"""
        answer = "".join(
            chunk["text"]
            for chunk in self.stream(user, system=system, images=images, use_thinking=use_thinking)
            if chunk["type"] == "content"
        ).strip()
        if not answer:
            raise RuntimeError("大模型没有返回最终答案")
        return answer

    def generate(self, prompt: str, images=None, use_thinking: bool = False) -> str:
        return self.chat(
            system="", user=prompt, images=images, use_thinking=use_thinking
        )

    async def stream_async(
        self,
        prompt: str = "",
        *,
        system: str = DEFAULT_SYSTEM_PROMPT,
        images: Optional[list[dict]] = None,
        use_thinking: bool = False,
    ) -> AsyncIterator[LLMChunk]:
        """异步流式调用模型。"""
        completion = await self._get_async_client().chat.completions.create(
            model=self.model,
            messages=self._messages(system, prompt, images),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"enable_thinking": use_thinking},
            stream=True,
        )
        async for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield {"type": "reasoning", "text": reasoning}
            content = getattr(delta, "content", None)
            if content:
                yield {"type": "content", "text": content}

    async def chat_async(
        self,
        system: str = DEFAULT_SYSTEM_PROMPT,
        user: str = "",
        images: Optional[list[dict]] = None,
        use_thinking: bool = False,
    ) -> str:
        parts: list[str] = []
        async for chunk in self.stream_async(
            user, system=system, images=images, use_thinking=use_thinking
        ):
            if chunk["type"] == "content":
                parts.append(chunk["text"])
        answer = "".join(parts).strip()
        if not answer:
            raise RuntimeError("大模型没有返回最终答案")
        return answer

    async def generate_async(self, prompt: str, images=None, use_thinking: bool = False) -> str:
        return await self.chat_async(
            system="", user=prompt, images=images, use_thinking=use_thinking
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None


_default_model: ModelClient | None = None
_default_model_lock = threading.Lock()


def get_default_model() -> ModelClient:
    """返回整个项目共享的默认模型实例。"""
    global _default_model
    if _default_model is None:
        with _default_model_lock:
            if _default_model is None:
                _default_model = ModelClient()
    return _default_model


def LLM_stream(prompt: str, use_thinking: bool = False) -> Iterator[LLMChunk]:
    """兼容知识图谱原有的简洁流式接口。"""
    yield from get_default_model().stream(prompt, use_thinking=use_thinking)


def LLM(prompt: str, use_thinking: bool = False) -> str:
    """兼容知识图谱原有的简洁同步接口。"""
    return get_default_model().generate(prompt, use_thinking=use_thinking)
