"""HTTP adapters for the shared chat service."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.v1 import access, chat_memory, deps
from app.api.v1.chat_schemas import ChatRequest, ChatResponse
from app.rag.chat_service import ChatService
from app.rag.hybrid_retriever import build_hybrid_retriever
from app.rag.postprocess import ResponsePostProcessor
from app.rag.prompt import PromptBuilder
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import RerankingRetriever, build_reranker_from_env
from app.rag.streaming import RAGStreamingPipeline
logger = logging.getLogger(__name__)
router = APIRouter()
_query_rewriter: QueryRewriter | None = None
_hybrid_retriever: Any = None
_reranker: Any = None
_prompt_builder: PromptBuilder | None = None
_llm_client: Any = None
_postprocessor: ResponsePostProcessor | None = None
_chat_service: ChatService | None = None
def _get_query_rewriter() -> QueryRewriter:
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter()
    return _query_rewriter


def _get_hybrid_retriever() -> Any:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from app.ingestion.embedding import build_embedding_client_from_env

        embedding = build_embedding_client_from_env()
        if os.getenv("EMBEDDING_PROVIDER", "mock") == "mock":
            from app.rag.mock_retriever import MockHybridRetriever

            _hybrid_retriever = MockHybridRetriever()
        else:
            from app.ingestion.indexer import SqlAlchemyMetadataStore
            from app.rag.retriever import MilvusRetriever

            vector_retriever = MilvusRetriever(
                collection_name=os.getenv("MILVUS_COLLECTION", "kf_chunks"),
                uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
                token=os.getenv("MILVUS_TOKEN"),
            )
            metadata_store = None
            database_url = os.getenv("DATABASE_URL", "").strip()
            if database_url:
                try:
                    metadata_store = SqlAlchemyMetadataStore(database_url)
                except Exception as exc:
                    logger.warning("MySQL sparse search unavailable: %s", exc)
            _hybrid_retriever = build_hybrid_retriever(
                embedding_client=embedding,
                vector_retriever=vector_retriever,
                enable_bm25=True,
                candidate_provider=(
                    metadata_store.search_chunks if metadata_store else None
                ),
            )
    return _hybrid_retriever


def _get_reranker() -> Any:
    global _reranker
    if _reranker is None:
        _reranker = RerankingRetriever(
            build_reranker_from_env(),
            rerank_top_k=10,
            score_threshold=float(os.getenv("RAG_RERANK_SCORE_THRESHOLD", "0.3")),
        )
    return _reranker


def _get_prompt_builder() -> PromptBuilder:
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder()
    return _prompt_builder


def _get_llm_client() -> Any:
    global _llm_client
    if _llm_client is None:
        from app.core.config import load_config_from_env
        from app.llm.client import build_llm_client

        _, _, _, _, config, _ = load_config_from_env()
        _llm_client = build_llm_client(
            provider=config.provider,
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
        )
    return _llm_client


def _get_postprocessor() -> ResponsePostProcessor:
    global _postprocessor
    if _postprocessor is None:
        _postprocessor = ResponsePostProcessor()
    return _postprocessor


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        from app.rag.refusal import build_refusal_message

        _chat_service = ChatService(
            query_rewriter=_get_query_rewriter(),
            retriever=_get_hybrid_retriever(),
            reranker=_get_reranker(),
            prompt_builder=_get_prompt_builder(),
            llm_client=_get_llm_client(),
            postprocessor=_get_postprocessor(),
            load_history=chat_memory.load_history,
            save_turn=chat_memory.save_turn,
            load_long_term=chat_memory.load_long_term,
            save_long_term=chat_memory.save_long_term,
            refusal_message=build_refusal_message,
            score_threshold=float(os.getenv("RAG_RERANK_SCORE_THRESHOLD", "0.3")),
        )
    return _chat_service


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run the shared non-streaming RAG pipeline."""
    request = access.scope_chat_request(request)
    try:
        return get_chat_service().answer(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat processing failed",
        ) from exc


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Return the existing SSE adapter for streaming responses."""
    request = access.scope_chat_request(request)
    try:
        pipeline = RAGStreamingPipeline(get_chat_service())
        return StreamingResponse(
            pipeline.stream_request(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Streaming chat failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Streaming failed",
        ) from exc


@router.get("/chat/health")
async def chat_health() -> dict[str, Any]:
    """Report pipeline components and optional memory dependencies."""
    return {
        "status": "healthy",
        "components": {
            "query_rewriter": "ok",
            "retriever": "ok",
            "reranker": "ok",
            "prompt_builder": "ok",
            "llm": "ok",
            "postprocessor": "ok",
        },
        "memory": deps.memory_status(),
    }
