"""RAG 问答主流程编排。

`ChatService` 不直接创建数据库或模型客户端，而是通过构造参数注入依赖。
这样标准 JSON 接口和 SSE 流式接口可以复用同一套检索、重排、Prompt 与记忆逻辑，
测试时也能替换为 Fake/Mock 实现。
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from app.api.v1.chat_schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    SourceChunkResponse,
)
from app.rag.chat_models import PreparedChat
from app.rag.prompt import PromptBuilder
from app.rag.query_rewriter import HybridQueryBuilder, QueryRewriter

logger = logging.getLogger(__name__)


class ChatService:
    """Run one chat request with injected infrastructure dependencies."""

    def __init__(
        self,
        *,
        query_rewriter: QueryRewriter,
        retriever: Any,
        reranker: Any,
        prompt_builder: PromptBuilder,
        llm_client: Any,
        postprocessor: Any,
        load_history: Callable[[str | None, int], list[dict[str, Any]]],
        save_turn: Callable[[str | None, str, str], None],
        refusal_message: Callable[[], str],
        score_threshold: float,
        load_long_term: Callable[[str | None, str], dict[str, Any]] | None = None,
        save_long_term: Callable[[str | None, str, str], None] | None = None,
    ) -> None:
        self.query_rewriter = query_rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.postprocessor = postprocessor
        self.load_history = load_history
        self.save_turn = save_turn
        self.refusal_message = refusal_message
        self.score_threshold = score_threshold
        self.load_long_term = load_long_term
        self.save_long_term = save_long_term

    def prepare(self, request: ChatRequest) -> PreparedChat:
        """完成生成答案前的公共准备阶段。

        流程：读取短期历史 → 改写问题 → 构造混合检索词 → 召回 → 重排 →
        相关性阈值过滤。返回的 ``PreparedChat`` 同时供普通问答和流式问答使用。
        """
        # 1. 调用方未提供 conversation_id 时生成一个，便于串联本次问答结果。
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # 2. session_id 存在且启用记忆时，从 Redis 读取最近若干轮对话。
        history = self._history(request)

        # 3. Query 改写用于修正口语、省略或错别字；检索查询还会补充上一轮用户问题。
        rewritten = self._rewrite(request.query, request.enable_query_rewrite)
        retrieval_query = self._retrieval_query(rewritten, history)
        hybrid_query = HybridQueryBuilder(self.query_rewriter).build(retrieval_query)

        # 4. 租户和知识库过滤条件必须一路传入检索层，防止跨范围召回。
        filters = self._filters(request)

        # 5. 先扩大候选集，再交给重排器收敛到用户要求的 top_k。
        retrieval = self.retriever.search(
            query=hybrid_query["vector_query"],
            top_k=request.top_k * 2,
            final_top_k=request.top_k * 2,
            filters=filters or None,
            bm25_keywords=(
                hybrid_query["bm25_keywords"] if request.enable_hybrid_search else None
            ),
        )
        chunks = retrieval.results
        final_chunks = self._rerank(request, rewritten, chunks)

        # 6. 没有候选或全部低于阈值时清空上下文，后续统一进入拒答流程。
        # 不把低相关文本交给 LLM，是降低“看似合理但无知识依据”回答的关键边界。
        if not final_chunks or (
            request.enable_rerank
            and all(self._score(chunk) < self.score_threshold for chunk in final_chunks)
        ):
            final_chunks = []
        return PreparedChat(
            request=request,
            conversation_id=conversation_id,
            rewritten_query=rewritten,
            history=history,
            retrieved_chunks=chunks,
            final_chunks=final_chunks,
        )

    def answer(self, request: ChatRequest) -> ChatResponse:
        """生成一次完整回答；记忆组件失败不会中断主问答链路。"""
        started = time.time()
        prepared = self.prepare(request)

        # 检索阶段已经决定是否拒答，生成阶段不再自行猜测知识库外的信息。
        if prepared.refused:
            answer = self.refusal_message()
            self.persist_turn(request, answer)
            return self._response(
                request,
                answer,
                prepared.conversation_id,
                prepared.rewritten_query,
                len(prepared.history),
                len(prepared.retrieved_chunks),
                [],
                "refusal",
                ["No relevant content found in knowledge base"],
            )

        # 将最终证据、长期记忆和回答长度要求组装成 Prompt，再调用统一 LLM 客户端。
        prompt = self.build_prompt(prepared)
        generated = self.llm_client.generate(
            self.build_messages(prompt, prepared.history),
            temperature=request.temperature,
            max_tokens=None,
        )
        answer, warnings, _ = self.process_answer(
            generated.content,
            enabled=request.enable_postprocess,
        )

        # 响应中保留来源、token 和端到端耗时，便于前端展示和后续质量评估。
        response = self._response(
            request,
            answer,
            prepared.conversation_id,
            prepared.rewritten_query,
            len(prepared.history),
            len(prepared.retrieved_chunks),
            prepared.final_chunks,
            generated.model,
            warnings,
            prompt_tokens=generated.prompt_tokens,
            completion_tokens=generated.completion_tokens,
            latency_ms=int((time.time() - started) * 1000),
        )
        self.persist_turn(request, answer)
        return response

    def _long_term_context(self, request: ChatRequest) -> dict[str, Any]:
        if not request.enable_memory or self.load_long_term is None:
            return {}
        try:
            return self.load_long_term(request.session_id, request.query)
        except Exception as exc:
            logger.warning("长期记忆召回失败，按无长期记忆处理: %s", exc)
            return {}

    def build_prompt(
        self,
        prepared: PreparedChat,
    ) -> Any:
        return self.prompt_builder.build(
            prepared.rewritten_query,
            list(prepared.final_chunks),
            max_length=prepared.request.max_length,
            additional_context=self._long_term_context(prepared.request),
        )

    def _history(self, request: ChatRequest) -> list[dict[str, Any]]:
        if not request.enable_memory:
            return []
        return self.load_history(request.session_id, request.max_memory_turns)

    def _rewrite(self, query: str, enabled: bool) -> str:
        if not enabled:
            return query
        analysis = self.query_rewriter.analyze(query)
        return analysis.rewritten_query or query

    @staticmethod
    def _retrieval_query(query: str, history: list[dict[str, Any]]) -> str:
        if not history or len(query) > 15:
            return query
        previous = next(
            (
                str(item.get("content", ""))
                for item in reversed(history)
                if item.get("role") == "user"
            ),
            "",
        )
        return f"{previous} {query}" if previous else query

    @staticmethod
    def _filters(request: ChatRequest) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if request.tenant_id is not None:
            filters["tenant_id"] = request.tenant_id
        if request.knowledge_base_id is not None:
            filters["kb_id"] = request.knowledge_base_id
        return filters

    def _rerank(self, request: ChatRequest, query: str, chunks: Any) -> Any:
        if not request.enable_rerank:
            return chunks[: request.top_k]
        return self.reranker.rerank_results(query, chunks, top_k=request.top_k).results

    @staticmethod
    def _score(chunk: Any) -> float:
        rerank_score = getattr(chunk, "rerank_score", None)
        if rerank_score is not None:
            return float(rerank_score)
        return float(chunk.score)

    def build_messages(self, prompt: Any, history: list[dict[str, Any]]) -> list[Any]:
        """Build provider-neutral messages from a prompt and conversation history."""
        from app.llm.client import Message

        messages = [Message(role="system", content=prompt.system_prompt)]
        for item in history:
            role = str(item.get("role", ""))
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                messages.append(Message(role=role, content=content))
        messages.append(Message(role="user", content=prompt.user_prompt_template))
        return messages

    def process_answer(
        self,
        answer: str,
        *,
        enabled: bool,
    ) -> tuple[str, list[str], int]:
        """Apply the same final post-processing for all response modes."""
        if not enabled:
            return answer, [], 0
        result = self.postprocessor.process(answer)
        return result.text, result.warnings, result.replacements_made

    def persist_turn(self, request: ChatRequest, answer: str) -> None:
        """完整问答只持久化一次，避免普通/流式适配器重复写入。"""
        # 短期记忆是会话体验的一部分；长期记忆属于增强能力，失败时只记录日志。
        self.save_turn(request.session_id, request.query, answer)
        if self.save_long_term is not None:
            try:
                self.save_long_term(request.session_id, request.query, answer)
            except Exception as exc:
                logger.warning("长期记忆保存失败: %s", exc)

    def source_payloads(self, chunks: Any) -> list[dict[str, Any]]:
        """Serialize source metadata shared by HTTP and SSE adapters."""
        return [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "score": self._score(chunk),
                "summary": getattr(chunk, "summary", ""),
            }
            for chunk in chunks
        ]

    def _response(
        self,
        request: ChatRequest,
        answer: str,
        conversation_id: str,
        rewritten: str,
        history_count: int,
        retrieval_count: int,
        chunks: Any,
        model: str,
        warnings: list[str],
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: int | None = None,
    ) -> ChatResponse:
        return ChatResponse(
            answer=answer,
            query=request.query,
            conversation_id=conversation_id,
            session_id=request.session_id,
            sources=(
                [
                    SourceChunkResponse(**source)
                    for source in self.source_payloads(chunks)
                ]
                if request.enable_citation
                else []
            ),
            metadata=ChatMetadata(
                query_rewritten=rewritten if request.enable_query_rewrite else None,
                retrieval_count=retrieval_count,
                rerank_enabled=request.enable_rerank,
                final_chunk_count=len(chunks),
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                history_turns=history_count // 2,
                memory_enabled=bool(request.session_id and request.enable_memory),
            ),
            warnings=warnings,
        )
