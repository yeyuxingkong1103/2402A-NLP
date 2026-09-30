"""Streaming response support for RAG using Server-Sent Events (SSE).

This module implements:
1. SSE streaming generator
2. LLM streaming wrapper
3. Chunk-by-chunk post-processing
4. Error handling for streaming
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class StreamingError(RuntimeError):
    """Raised when streaming fails."""


def format_sse(data: dict[str, Any], event: str | None = None) -> str:
    """Format data as Server-Sent Event."""
    lines = []

    if event:
        lines.append(f"event: {event}")

    lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
    lines.append("")  # Empty line to end the event

    return "\n".join(lines) + "\n"


class StreamingLLMWrapper:
    """Wrapper to make LLM client support streaming."""

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    def stream_generate(self, messages: list[Any], **kwargs: Any) -> Iterator[str]:
        """Generate streaming response."""
        # Check if client supports streaming natively
        if hasattr(self.llm_client, "stream_generate"):
            yield from self.llm_client.stream_generate(messages, **kwargs)
            return

        # Fallback: simulate streaming from non-streaming client
        try:
            result = self.llm_client.generate(messages, **kwargs)
            content = result.content

            # Simulate streaming by yielding in chunks
            chunk_size = 20  # Characters per chunk
            for i in range(0, len(content), chunk_size):
                chunk = content[i : i + chunk_size]
                yield chunk

                # Simulate delay
                import time

                time.sleep(0.05)

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise StreamingError(f"Failed to generate stream: {e}") from e


class RAGStreamingPipeline:
    """Stream responses from the same prepared RAG request as non-streaming chat."""

    def __init__(self, chat_service: Any) -> None:
        self.chat_service = chat_service
        self.streaming_llm = StreamingLLMWrapper(chat_service.llm_client)

    def stream_request(self, request: Any) -> Iterator[str]:
        """Execute a prepared chat request and emit Server-Sent Events."""
        try:
            yield from self._stream_request(request)
        except Exception as exc:
            logger.exception("RAG streaming pipeline failed")
            yield format_sse({"type": "error", "error": str(exc)}, event="error")

    def _stream_request(self, request: Any) -> Iterator[str]:
        prepared = self.chat_service.prepare(request)
        yield format_sse(
            {
                "type": "start",
                "conversation_id": prepared.conversation_id,
                "session_id": request.session_id,
            },
            event="start",
        )
        yield format_sse(
            {
                "type": "rewrite",
                "original": request.query,
                "rewritten": prepared.rewritten_query,
            },
            event="metadata",
        )
        yield format_sse(
            {
                "type": "retrieval",
                "count": len(prepared.retrieved_chunks),
                "sources": (
                    self.chat_service.source_payloads(prepared.final_chunks)
                    if request.enable_citation
                    else []
                ),
            },
            event="metadata",
        )
        yield format_sse(
            {
                "type": "rerank",
                "enabled": request.enable_rerank,
                "final_count": len(prepared.final_chunks),
            },
            event="metadata",
        )

        if prepared.refused:
            answer = self.chat_service.refusal_message()
            yield format_sse(
                {"type": "content", "content": answer, "index": 0},
                event="content",
            )
            self.chat_service.persist_turn(request, answer)
            yield format_sse(
                {"type": "done", "refused": True, "total_length": len(answer)},
                event="done",
            )
            return

        prompt = self.chat_service.build_prompt(prepared)
        messages = self.chat_service.build_messages(prompt, prepared.history)
        stream = self.streaming_llm.stream_generate(
            messages,
            temperature=request.temperature,
            max_tokens=None,
        )

        answer_parts: list[str] = []
        for index, chunk in enumerate(stream):
            if not chunk:
                continue
            chunk = chunk.replace("\x00", "")
            answer_parts.append(chunk)
            yield format_sse(
                {"type": "content", "content": chunk, "index": index},
                event="content",
            )

        raw_answer = "".join(answer_parts)
        final_answer, warnings, replacements = self.chat_service.process_answer(
            raw_answer,
            enabled=request.enable_postprocess,
        )
        if final_answer != raw_answer:
            yield format_sse(
                {
                    "type": "postprocess",
                    "corrected_text": final_answer,
                    "replacements": replacements,
                    "warnings": warnings,
                },
                event="metadata",
            )

        self.chat_service.persist_turn(request, final_answer)
        yield format_sse(
            {
                "type": "done",
                "refused": False,
                "total_tokens": len(final_answer) // 4,
                "total_length": len(final_answer),
                "warnings": warnings,
            },
            event="done",
        )


def create_sse_stream(content: str) -> Iterator[str]:
    """Create a simple SSE stream from static content."""
    chunk_size = 20
    for i in range(0, len(content), chunk_size):
        chunk = content[i : i + chunk_size]
        yield format_sse({"type": "content", "content": chunk}, event="content")

    yield format_sse({"type": "done"}, event="done")
