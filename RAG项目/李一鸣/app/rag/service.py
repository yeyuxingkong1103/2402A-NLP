import hashlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.rag.chunking import ChunkingConfig, DocumentChunker
from app.rag.embeddings import EmbeddingService
from app.rag.llm import LLMService
from app.rag.memory import MemoryService
from app.rag.parsers import DocumentParser
from app.rag.postprocess import postprocess_answer
from app.rag.prompt import build_messages
from app.rag.reranker import RerankerService
from app.rag.retrievers import HybridRetriever
from app.rag.types import ChatResult
from app.rag.vector_store import build_vector_store
from app.storage.repositories import (
    create_document,
    get_role,
    save_message,
    update_document,
)

logger = logging.getLogger(__name__)


class RAGService:
    """串联离线入库与在线问答的应用服务。

    文档入库路径是：解析 -> 分块 -> 向量化 -> 向量库。
    聊天路径是：记忆 -> 查询改写 -> 混合检索 -> 重排 -> 提示词 -> 大模型。
    """

    def __init__(self, settings: Settings):
        # 这些组件在服务启动时组装一次，单次请求只负责调用，不重复加载模型或连接。
        self.settings = settings
        self.parser = DocumentParser()
        self.chunker = DocumentChunker(
            ChunkingConfig(
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
        )
        self.embeddings = EmbeddingService(settings)
        self.vector_store = build_vector_store(settings)
        self.retriever = HybridRetriever(self.vector_store, self.embeddings, settings)
        self.reranker = RerankerService(settings)
        self.memory = MemoryService(settings)
        self.llm = LLMService(settings)

    async def ingest_file(
        self,
        session: AsyncSession,
        file_path: str | Path,
        filename: str,
        source: str = "upload",
        role_id: str | None = None,
    ):
        # 用文件哈希记录内容指纹，便于后续实现去重和增量更新。
        path = Path(file_path)
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        document = await create_document(session, filename, source, content_hash)
        started = time.perf_counter()
        try:
            parsed = self.parser.parse(path)
            # role_id 写入每个 chunk 的 metadata，Milvus 和本地检索都可据此做权限/角色过滤。
            metadata = {
                **parsed.metadata,
                "filename": filename,
                "role_id": role_id or "global",
            }
            chunks = self.chunker.split(
                parsed,
                document_id=document.id,
                source=filename,
                base_metadata=metadata,
            )
            vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
            # 将每个知识块和对应向量绑定，再一次性写入向量库。
            for chunk, vector in zip(chunks, vectors):
                chunk.embedding = vector
            self.vector_store.upsert(chunks)
            self.retriever.refresh()
            document = await update_document(
                session, document.id, "ready", chunk_count=len(chunks)
            )
            logger.info(
                "document ingestion complete",
                extra={
                    "document_id": document.id if document else "-",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            return document
        except Exception as exc:
            logger.exception("document ingestion failed: %s", filename)
            document = await update_document(
                session, document.id, "failed", error_message=str(exc)
            )
            raise RuntimeError(f"document ingestion failed: {exc}") from exc

    async def chat(
        self,
        session: AsyncSession,
        user_id: str,
        role_id: str,
        conversation_id: str,
        message: str,
        top_k: int | None = None,
    ) -> ChatResult:
        # 非流式接口等待完整答案后一次性返回。
        prepared = await self._prepare_chat(
            session, user_id, role_id, conversation_id, message, top_k
        )
        trace_id = prepared["trace_id"]
        timings = prepared["timings"]
        citations = prepared["citations"]
        messages = prepared["messages"]
        llm_started = time.perf_counter()
        raw_answer = await self.llm.complete(messages)
        timings["llm_ms"] = round((time.perf_counter() - llm_started) * 1000, 2)
        answer = postprocess_answer(raw_answer)

        citation_payload = self._citation_payload(citations)
        await self._save_chat_turn(
            session,
            user_id,
            role_id,
            conversation_id,
            message,
            answer,
            citation_payload,
        )
        timings["total_ms"] = round(
            (time.perf_counter() - prepared["total_started"]) * 1000, 2
        )
        logger.info(
            "chat completed",
            extra={
                "conversation_id": conversation_id,
                "role_id": role_id,
                "user_id": user_id,
                "latency_ms": timings["total_ms"],
            },
        )
        logger.debug("chat trace=%s timings=%s citations=%s", trace_id, timings, len(citations))
        return ChatResult(answer=answer, citations=citations, trace_id=trace_id, timings=timings)

    async def stream_chat(
        self,
        session: AsyncSession,
        user_id: str,
        role_id: str,
        conversation_id: str,
        message: str,
        top_k: int | None = None,
    ) -> AsyncIterator[dict]:
        # 流式接口先发送引用元数据，再逐片发送 token，最后保存完整答案。
        prepared = await self._prepare_chat(
            session, user_id, role_id, conversation_id, message, top_k
        )
        trace_id = prepared["trace_id"]
        citations = prepared["citations"]
        messages = prepared["messages"]
        citation_payload = self._citation_payload(citations)
        yield {"type": "meta", "trace_id": trace_id, "citations": citation_payload}
        pieces: list[str] = []
        try:
            async for piece in self.llm.stream_complete(messages):
                pieces.append(piece)
                yield {"type": "token", "content": piece}
            answer = postprocess_answer("".join(pieces))
            await self._save_chat_turn(
                session,
                user_id,
                role_id,
                conversation_id,
                message,
                answer,
                citation_payload,
            )
            timings = prepared["timings"]
            timings["total_ms"] = round(
                (time.perf_counter() - prepared["total_started"]) * 1000, 2
            )
            logger.info(
                "stream chat completed",
                extra={
                    "conversation_id": conversation_id,
                    "role_id": role_id,
                    "user_id": user_id,
                    "latency_ms": timings["total_ms"],
                },
            )
            yield {"type": "done", "answer": answer, "timings": timings}
        except Exception:
            logger.exception("stream chat failed", extra={"conversation_id": conversation_id})
            raise

    async def _prepare_chat(
        self,
        session: AsyncSession,
        user_id: str,
        role_id: str,
        conversation_id: str,
        message: str,
        top_k: int | None,
    ) -> dict:
        # 这个阶段只准备上下文，不调用大模型，便于分别统计检索和生成耗时。
        trace_id = uuid.uuid4().hex
        timings: dict[str, float] = {}
        total_started = time.perf_counter()
        role = await get_role(session, role_id)
        if role is None:
            raise ValueError(f"role not found: {role_id}")

        history_started = time.perf_counter()
        history = self.memory.get_history(user_id, role_id, conversation_id)
        timings["memory_ms"] = round((time.perf_counter() - history_started) * 1000, 2)

        query_started = time.perf_counter()
        rewritten_query = self.rewrite_query(message, history)
        timings["query_rewrite_ms"] = round((time.perf_counter() - query_started) * 1000, 2)

        retrieval_started = time.perf_counter()
        filters = None
        # knowledge_scope 是显式知识范围；非 doctor 角色默认屏蔽 global 医疗演示资料。
        if role.knowledge_scope not in {"", "global"}:
            filters = {"role_id": role.knowledge_scope}
        elif role.category.lower() not in self.settings.rag_global_categories:
            # Global demo documents currently contain medical content. Keep them
            # away from unrelated roles until role-specific documents are ingested.
            filters = {"role_id": role.id}
        candidates = self.retriever.retrieve(rewritten_query, top_k=top_k, filters=filters)
        timings["retrieval_ms"] = round((time.perf_counter() - retrieval_started) * 1000, 2)

        rerank_started = time.perf_counter()
        citations = self.reranker.rerank(
            rewritten_query, candidates, top_n=top_k or self.settings.top_k_final
        )
        timings["rerank_ms"] = round((time.perf_counter() - rerank_started) * 1000, 2)

        logger.info(
            "chat context prepared",
            extra={"conversation_id": conversation_id, "role_id": role_id, "user_id": user_id},
        )
        return {
            "trace_id": trace_id,
            "timings": timings,
            "total_started": total_started,
            "citations": citations,
            "messages": build_messages(role.system_prompt, message, history, citations),
        }

    async def _save_chat_turn(
        self,
        session: AsyncSession,
        user_id: str,
        role_id: str,
        conversation_id: str,
        message: str,
        answer: str,
        citation_payload: list[dict],
    ) -> None:
        # SQLite/MySQL 保存完整审计记录，Redis/内存只保存短期上下文。
        await save_message(session, user_id, role_id, conversation_id, "user", message)
        await save_message(
            session,
            user_id,
            role_id,
            conversation_id,
            "assistant",
            answer,
            citation_payload,
        )
        self.memory.append_turn(user_id, role_id, conversation_id, message, answer)

    @staticmethod
    def _citation_payload(citations) -> list[dict]:
        # 对外只返回有限长度的文本片段，避免引用内容过大。
        return [
            {
                "chunk_id": item.chunk.id,
                "document_id": item.chunk.document_id,
                "filename": item.chunk.source,
                "content": item.chunk.text[:1000],
                "score": round(item.score, 6),
                "retrieval_method": item.retrieval_method,
            }
            for item in citations
        ]

    @staticmethod
    def rewrite_query(message: str, history: list[dict]) -> str:
        # 短追问补上上一轮问题，让“那需要多久？”也能被独立检索。
        message = message.strip()
        if not history or len(message) >= 12:
            return message
        previous_user = next(
            (item["content"] for item in reversed(history) if item.get("role") == "user"),
            "",
        )
        return f"{previous_user}；补充问题：{message}" if previous_user else message
