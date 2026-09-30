"""LLM client implementations supporting multiple providers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when LLM generation fails."""


@dataclass(slots=True, frozen=True)
class Message:
    """A single message in a conversation."""

    role: str
    content: str


@dataclass(slots=True, frozen=True)
class GenerationResult:
    """Result from LLM generation."""

    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


class LLMClient(Protocol):
    """Common interface for LLM providers."""

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate a response given conversation messages."""


@dataclass(slots=True)
class MockLLMClient:
    """Deterministic LLM client for tests and no-API-key development."""

    model: str = "mock-model"

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        last_message = messages[-1].content if messages else ""
        response = f"Mock response to: {last_message[:100]}"
        return GenerationResult(
            content=response,
            model=self.model,
            prompt_tokens=len(last_message) // 4,
            completion_tokens=len(response) // 4,
            finish_reason="stop",
        )


@dataclass(slots=True)
class OpenAILLMClient:
    """OpenAI-compatible LLM client."""

    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout_seconds: int = 60

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise LLMError("openai package is required for OpenAILLMClient") from exc

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": msg.role, "content": msg.content} for msg in messages],
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            )
        except Exception as exc:
            raise LLMError(f"OpenAI generation failed: {exc}") from exc

        choice = response.choices[0]
        usage = response.usage
        return GenerationResult(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason,
        )


@dataclass(slots=True)
class AnthropicLLMClient:
    """Anthropic Claude LLM client."""

    api_key: str
    model: str = "claude-sonnet-4-6"
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout_seconds: int = 60

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        client = self._client()
        request = self._build_request(messages, max_tokens)

        try:
            response = client.messages.create(**request)
        except Exception as exc:
            raise LLMError(f"Anthropic generation failed: {exc}") from exc

        # content 里可能混有 thinking 块（Sonnet 5 默认开 adaptive thinking），
        # 不能直接取 content[0]，要挑第一个 text 块。
        content_text = next(
            (block.text for block in response.content if getattr(block, "type", None) == "text"),
            "",
        )
        return GenerationResult(
            content=content_text,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason,
        )

    def stream_generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield answer text incrementally as the model produces it.

        Without this, StreamingLLMWrapper falls back to calling generate() and
        slicing the finished string — the caller waits for the whole answer and
        only then sees fake "streaming".
        """
        client = self._client()
        request = self._build_request(messages, max_tokens)

        try:
            with client.messages.stream(**request) as stream:
                for event in stream:
                    # 只转发正文增量；thinking 块的 delta 不往前端发。
                    if getattr(event, "type", None) != "content_block_delta":
                        continue
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", None) == "text_delta":
                        yield delta.text
        except Exception as exc:
            raise LLMError(f"Anthropic streaming failed: {exc}") from exc

    def _client(self) -> Any:
        try:
            from anthropic import Anthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise LLMError("anthropic package is required for AnthropicLLMClient") from exc

        return Anthropic(
            api_key=self.api_key,
            base_url=self.base_url or None,
            timeout=self.timeout_seconds,
        )

    def _build_request(
        self, messages: list[Message], max_tokens: int | None
    ) -> dict[str, Any]:
        system_messages = [msg.content for msg in messages if msg.role == "system"]
        conversation = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role in {"user", "assistant"}
        ]

        # 新版 Messages API 已移除 temperature（Sonnet 5 / Opus 5 传了会 400），
        # 采样强度改由 output_config.effort 控制，所以这里不再透传 temperature。
        request: dict[str, Any] = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if system_messages:
            request["system"] = "\n\n".join(system_messages)

        return request


def build_llm_client(
    *,
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    timeout_seconds: int = 60,
) -> LLMClient:
    """Build an LLM client from configuration."""
    provider_lower = provider.lower()

    if provider_lower == "mock":
        return MockLLMClient(model=model)

    if provider_lower == "openai":
        if not api_key:
            raise LLMError("API key is required for OpenAI provider")
        return OpenAILLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    if provider_lower == "anthropic":
        if not api_key:
            raise LLMError("API key is required for Anthropic provider")
        return AnthropicLLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    raise LLMError(f"Unsupported LLM provider: {provider}")
