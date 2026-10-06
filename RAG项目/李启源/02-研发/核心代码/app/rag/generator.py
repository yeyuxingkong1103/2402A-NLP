"""RAG generation with context assembly and prompt templates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.llm.client import LLMClient, Message
from app.rag.retriever import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """Raised when RAG generation fails."""


@dataclass(slots=True, frozen=True)
class RAGResponse:
    """Complete RAG response with answer and sources."""

    answer: str
    query: str
    sources: tuple[RetrievedChunk, ...]
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


DEFAULT_SYSTEM_PROMPT = """你是一个专业的知识问答助手。你的任务是根据提供的参考文档回答用户的问题。

请遵循以下规则：
1. 仅基于提供的参考文档回答问题
2. 如果文档中没有相关信息，请明确告知用户
3. 回答要准确、清晰、有条理
4. 适当引用文档来源支持你的回答
5. 使用专业但易懂的语言"""


DEFAULT_USER_PROMPT_TEMPLATE = """参考文档：
{context}

用户问题：{query}

请根据上述参考文档回答用户的问题。如果文档中没有足够的信息，请如实说明。"""


class RAGGenerator:
    """RAG generator combining retrieval results with LLM generation."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        max_context_tokens: int = 4000,
        include_sources: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE
        self.max_context_tokens = max_context_tokens
        self.include_sources = include_sources

    def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        conversation_history: list[Message] | None = None,
    ) -> RAGResponse:
        """Generate an answer using retrieved chunks as context."""
        if not retrieval_result.chunks:
            return self._generate_no_context_response(query)

        context = self._build_context(retrieval_result.chunks)
        user_content = self.user_prompt_template.format(context=context, query=query)

        messages = [Message(role="system", content=self.system_prompt)]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append(Message(role="user", content=user_content))

        try:
            result = self.llm_client.generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise GenerationError(f"LLM generation failed: {exc}") from exc

        logger.info(
            "RAG response generated",
            extra={
                "query_length": len(query),
                "chunks_used": len(retrieval_result.chunks),
                "answer_length": len(result.content),
                "model": result.model,
            },
        )

        return RAGResponse(
            answer=result.content,
            query=query,
            sources=retrieval_result.chunks if self.include_sources else tuple(),
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    def _build_context(self, chunks: tuple[RetrievedChunk, ...]) -> str:
        """Build context string from retrieved chunks."""
        context_parts: list[str] = []
        total_chars = 0
        max_chars = self.max_context_tokens * 4

        for index, chunk in enumerate(chunks, start=1):
            chunk_text = f"[文档{index}] 来源: {chunk.source}\n{chunk.text}"
            chunk_chars = len(chunk_text)

            if total_chars + chunk_chars > max_chars:
                remaining = max_chars - total_chars
                if remaining > 200:
                    chunk_text = chunk_text[:remaining] + "..."
                    context_parts.append(chunk_text)
                break

            context_parts.append(chunk_text)
            total_chars += chunk_chars

        return "\n\n".join(context_parts)

    def _generate_no_context_response(self, query: str) -> RAGResponse:
        """Generate a response when no relevant chunks are found."""
        no_context_answer = "抱歉，我在知识库中没有找到与您的问题相关的信息。请尝试换个方式提问，或者确认知识库中包含相关内容。"

        return RAGResponse(
            answer=no_context_answer,
            query=query,
            sources=tuple(),
            model="no-context",
            prompt_tokens=0,
            completion_tokens=0,
        )


class ConversationalRAGGenerator(RAGGenerator):
    """RAG generator with conversation history support."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        max_context_tokens: int = 4000,
        include_sources: bool = True,
        max_history_turns: int = 5,
    ) -> None:
        super().__init__(
            llm_client,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            max_context_tokens=max_context_tokens,
            include_sources=include_sources,
        )
        self.max_history_turns = max_history_turns
        self.conversation_store: dict[str, list[Message]] = {}

    def generate_with_history(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        *,
        conversation_id: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> RAGResponse:
        """Generate with conversation history tracking."""
        history = self.conversation_store.get(conversation_id, [])
        trimmed_history = history[-(self.max_history_turns * 2):]

        response = self.generate(
            query,
            retrieval_result,
            temperature=temperature,
            max_tokens=max_tokens,
            conversation_history=trimmed_history,
        )

        updated_history = trimmed_history + [
            Message(role="user", content=query),
            Message(role="assistant", content=response.answer),
        ]
        self.conversation_store[conversation_id] = updated_history

        return response

    def clear_history(self, conversation_id: str) -> None:
        """Clear conversation history for a given ID."""
        self.conversation_store.pop(conversation_id, None)
