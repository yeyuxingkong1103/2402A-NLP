"""编排层：把 Query 理解 → 混合检索 → 重排 → 生成 → 引用 → 落库 串成一条链路。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 编排（对应 设计/接口设计.md §2.17）

事件契约（UI 与测试共用，``stream()``）::

    status      {"stage": rewrite|retrieve|rerank|generate|persist, "msg": str}
    first_token {"first_token_ms": float, "trace_id": str}
    delta       {"text": str}
    citations   {"citations": [...]}
    done        {"answer": Answer, "trace_id": str}
    error       {"code": str, "message": str, "trace_id": str}

异常策略：``ask()/stream()`` 对外**不抛业务异常**，内部失败一律转 ``error`` 事件与兜底文案；
``build_index()`` 会抛 ``PDFParseError``（调用方是脚本/界面，需要显式失败）。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.core.chunker import Chunker, get_chunker
from app.core.citation import CitationManager, get_citation_manager
from app.core.config import get_settings
from app.core.conversation import ConversationManager, get_conversation_manager
from app.core.embedder import get_embedder
from app.core.errors import IndexNotReadyError, PDFParseError, RAGError, RetrievalError
from app.core.generator import Generator, get_generator
from app.core.logging_conf import log_event, logger, new_trace_id, trace, trace_context
from app.core.pdf_parser import PDFParser, get_pdf_parser
from app.core.query_understanding import QueryUnderstanding, get_query_understanding
from app.core.retriever import Retriever, get_retriever
from app.models.schemas import (
    Answer,
    Chunk,
    Conversation,
    DocumentMeta,
    Feedback,
    Message,
    QueryAnalysis,
    RetrievalTrace,
    RetrievedChunk,
)
from app.storage.sqlite_manager import SQLiteManager, get_sqlite_manager


class QAEngine:
    """RAG 问答引擎（唯一编排入口）。"""

    def __init__(self, doc_id: str | None = None, force_extractive: bool = False) -> None:
        self._settings = get_settings()
        self._store: SQLiteManager = get_sqlite_manager()
        self._store.init_schema()
        self._conversations: ConversationManager = get_conversation_manager(self._store)
        self._parser: PDFParser = get_pdf_parser()
        self._chunker: Chunker = get_chunker()
        self._retriever: Retriever = get_retriever()
        self._generator: Generator = get_generator(force_extractive=force_extractive)
        self._understanding: QueryUnderstanding = get_query_understanding()
        self._citations: CitationManager = get_citation_manager()
        self._doc_id: str = doc_id or ""
        self._chunks: list[Chunk] = []
        self._last_retrieved: list[RetrievedChunk] = []
        self._ready = False

    # ------------------------------------------------------------------
    # 索引 / 健康
    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        """索引是否就绪。"""
        return self._ready and bool(self._chunks)

    def default_doc_id(self) -> str:
        """默认文档 ID（无显式指定时用 PDF 文件名）。"""
        if self._doc_id:
            return self._doc_id
        meta = self._store.get_default_document()
        if meta is not None:
            return meta.doc_id
        return self._settings.paths.default_pdf.stem

    @trace
    def load_index(self, doc_id: str | None = None) -> bool:
        """从 SQLite + 索引目录加载分块与向量/BM25 索引。"""
        try:
            target = doc_id or self.default_doc_id()
            chunks = self._store.get_chunks(target)
            if not chunks:
                logger.warning("app.core.qa_engine", "库中无该文档分块，索引未就绪", doc_id=target)
                self._ready = False
                return False
            ok = self._retriever.load_index(chunks)
            self._doc_id = target
            self._chunks = chunks if ok else []
            self._ready = ok
            if ok:
                low, high = self._store.page_range(target)
                self._citations.set_valid_pages(set(range(low, high + 1)))
                self._citations.set_valid_chunk_ids({chunk.chunk_id for chunk in chunks})
                logger.info(
                    "app.core.qa_engine",
                    "索引加载完成",
                    doc_id=target,
                    chunks=len(chunks),
                    page_range=[low, high],
                )
            return ok
        except Exception:
            logger.exception("app.core.qa_engine", "加载索引异常", doc_id=doc_id)
            self._ready = False
            return False

    @trace
    def build_index(self, pdf_path: Path | None = None, reset: bool = True) -> dict[str, Any]:
        """解析 PDF → 分块 → 建索引 → 落库（失败显式抛出，由脚本/界面处理）。"""
        path = Path(pdf_path) if pdf_path else self._settings.paths.default_pdf
        document = self._parser.parse(path)
        self._parser.save(document)
        chunks = self._chunker.split(document)
        self._chunker.save(chunks)
        stats = self._retriever.build_index(chunks, reset=reset)
        self._retriever.save_index()
        self._doc_id = document.doc_id
        self._chunks = chunks
        # 文档元数据 + 分块落库
        low = min((chunk.page for chunk in chunks), default=1)
        high = max((chunk.page for chunk in chunks), default=document.page_count)
        self._store.upsert_document(
            DocumentMeta(
                doc_id=document.doc_id,
                title=document.title or path.stem,
                source_path=str(path),
                page_count=document.page_count,
                chunk_count=len(chunks),
                table_count=len(document.tables),
                status="indexed",
                is_default=True,
            )
        )
        written = self._store.insert_chunks(chunks, replace_doc=True)
        self._citations.set_valid_pages(set(range(min(low, 1), max(high, document.page_count) + 1)))
        self._citations.set_valid_chunk_ids({chunk.chunk_id for chunk in chunks})
        self._ready = True
        result = {
            **stats,
            "doc_id": document.doc_id,
            "pages": document.page_count,
            "tables": len(document.tables),
            "blocks": len(document.blocks),
            "merged_tables": document.merged_table_count,
            "table_errors": document.table_errors,
            "chunks_written": written,
            "chunk_stats": self._chunker.stats(chunks),
            "index_dir": str(self._retriever._store.index_dir),  # noqa: SLF001（脚本取证用）
        }
        logger.info("app.core.qa_engine", "索引构建完成", **{k: v for k, v in result.items() if k != "chunk_stats"})
        return result

    def warmup(self) -> dict[str, Any]:
        """预热嵌入与 LLM（把冷启动移出首问，首字预算治理的关键一步）。"""
        started = time.perf_counter()
        embed_info = get_embedder().warmup()
        llm_info: dict[str, Any] = {}
        try:
            if self._settings.llm.warmup and self._generator.backend_name() != "extractive":
                # 用极短 prompt 触发模型加载（不计入业务首字）
                for _event, _payload in self._generator._client.chat_stream(  # noqa: SLF001
                    [{"role": "user", "content": "你好"}], max_tokens=1
                ):
                    break
                llm_info = self._generator._client.probe().as_dict()  # noqa: SLF001
        except Exception:
            logger.exception("app.core.qa_engine", "LLM 预热失败（不影响提问，首问会稍慢）")
        elapsed = (time.perf_counter() - started) * 1000
        result = {
            "embedder": embed_info,
            "llm": llm_info,
            "elapsed_ms": round(elapsed, 2),
            "ready": self.ready,
        }
        logger.info("app.core.qa_engine", "预热完成", elapsed_ms=round(elapsed, 2), ready=self.ready)
        return result

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------
    def new_conversation(self, title: str = "新对话") -> str:
        """新建会话（绑定当前文档）。"""
        return self._conversations.new_conversation(title=title, doc_id=self._doc_id or None)

    def list_conversations(self) -> list[Conversation]:
        """列出会话。"""
        return self._conversations.list_conversations()

    def switch_conversation(self, conversation_id: str) -> None:
        """切换当前会话（仅记录日志，UI 侧维护状态）。"""
        logger.info("app.core.qa_engine", "切换会话", conversation_id=conversation_id)

    def clear_conversation(self, conversation_id: str) -> int:
        """清空会话消息。"""
        return self._conversations.clear(conversation_id)

    def get_messages(self, conversation_id: str) -> list[Message]:
        """读取会话消息。"""
        return self._conversations.messages(conversation_id)

    def last_retrieved(self) -> list[RetrievedChunk]:
        """最近一次检索结果（UI 折叠区取证用）。"""
        return self._last_retrieved

    def submit_feedback(
        self, conversation_id: str, message_id: int | None, rating: str, comment: str = "", question: str = ""
    ) -> int:
        """提交用户反馈。"""
        try:
            return self._store.add_feedback(
                Feedback(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    rating=rating,  # type: ignore[arg-type]
                    comment=comment,
                    question=question,
                )
            )
        except Exception:
            logger.exception("app.core.qa_engine", "提交反馈失败", conversation_id=conversation_id)
            return 0

    # ------------------------------------------------------------------
    # 问答
    # ------------------------------------------------------------------
    def ask(self, question: str, conversation_id: str | None = None) -> Answer:
        """非流式问答（聚合 ``stream()`` 事件；对外不抛业务异常）。"""
        final: Answer | None = None
        for event, payload in self.stream(question, conversation_id=conversation_id):
            if event == "done":
                final = payload.get("answer")
        if final is None:
            final = Answer(
                answer=self._settings.app.llm_unavailable_answer,
                is_unknown=True,
                mode="fallback",
                unknown_reason="stream_no_done_event",
            )
        return final

    def stream(self, question: str, conversation_id: str | None = None) -> Iterator[tuple[str, Any]]:
        """流式问答（事件契约见模块 docstring）。"""
        trace_id = new_trace_id()
        started = time.perf_counter()
        with trace_context(trace_id):
            yield from self._stream_inner(question, conversation_id, trace_id, started)

    def _stream_inner(
        self, question: str, conversation_id: str | None, trace_id: str, started: float
    ) -> Iterator[tuple[str, Any]]:
        """内部流式实现（异常统一转 error 事件 + 兜底文案）。"""
        analysis: QueryAnalysis | None = None
        contexts: list[RetrievedChunk] = []
        answer: Answer | None = None
        try:
            # 0) 索引校验（显式失败，不静默就地重建）
            if not self.ready and not self.load_index():
                message = f"索引未就绪，请先执行：pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py"
                logger.error("app.core.qa_engine", "索引未就绪", doc_id=self._doc_id, trace_id=trace_id)
                yield ("error", {"code": "INDEX_NOT_READY", "message": message, "trace_id": trace_id})
                answer = Answer(
                    answer=message,
                    is_unknown=True,
                    unknown_reason="index_not_ready",
                    mode="fallback",
                    trace_id=trace_id,
                )
                yield ("done", {"answer": answer, "trace_id": trace_id})
                return

            cid = self._conversations.get_or_create(conversation_id, doc_id=self._doc_id or None)
            history = self._conversations.history_pairs(cid)
            # 1) Query 理解与改写
            yield ("status", {"stage": "rewrite", "msg": "正在理解问题…"})
            analysis = self._understanding.analyze(question, history)
            self._conversations.add_user_message(cid, question)
            self._conversations.auto_title(cid, question)
            # 2) 检索（多路变体 + 混合 + 加权 + 重排）
            yield ("status", {"stage": "retrieve", "msg": "正在检索文档…"})
            variants = self._understanding.search_queries(analysis)
            retrieve_started = time.perf_counter()
            contexts = self._retriever.retrieve_multi(
                variants,
                analysis=analysis,
                top_k=self._settings.retrieval.rerank_top_n,
            )
            debug = self._retriever.debug_snapshot()
            self._last_retrieved = contexts
            yield ("status", {"stage": "rerank", "msg": f"已重排（{debug.rerank_mode}）"})
            # 3) 拒答判定（三条件联合）
            page_filter = analysis.page_filter or []
            if not contexts:
                if page_filter:
                    answer = self._fallback_answer(
                        self._settings.app.page_filter_empty_answer,
                        "PAGE_FILTER_EMPTY",
                        analysis,
                        trace_id,
                        started,
                    )
                else:
                    answer = self._fallback_answer(
                        self._settings.app.unknown_answer, "NO_EVIDENCE", analysis, trace_id, started
                    )
            elif not self._retriever.is_confident(contexts) or not self._retriever.is_answerable(
                # 可答性校验作用在**实际检索串**上：英文提问经语言桥接后检索串是中文，
                # 若拿英文原句去校验中文片段，实义词永远不重合 → 误判「无依据」（t11 回归根因之一）。
                # 注意：这不是放宽校验——同一条实义词覆盖规则仍在生效，只是作用在真正的查询上。
                analysis.search_query or question,
                contexts,
            ):
                answer = self._fallback_answer(
                    self._settings.app.unknown_answer, "NO_EVIDENCE", analysis, trace_id, started
                )
            elif not self._topic_gate(contexts, analysis, question, trace_id):
                answer = self._fallback_answer(
                    self._settings.app.unknown_answer, "NO_EVIDENCE", analysis, trace_id, started
                )
            else:
                # 4) 生成（流式）
                yield ("status", {"stage": "generate", "msg": "正在生成回答…"})
                for event, payload in self._generator.stream(question, contexts, history=history, analysis=analysis):
                    if event == "first_token":
                        yield (
                            "first_token",
                            {"first_token_ms": payload.get("first_token_ms", 0.0), "trace_id": trace_id},
                        )
                    elif event == "delta":
                        yield ("delta", payload)
                    elif event == "done":
                        answer = payload
            if answer is None:
                answer = self._fallback_answer(
                    self._settings.app.llm_unavailable_answer, "LLM_UNAVAILABLE", analysis, trace_id, started
                )
            # 5) 引用与落库
            yield ("status", {"stage": "persist", "msg": "正在整理引用…"})
            answer.trace_id = trace_id
            answer.query_analysis = analysis
            answer.retrieved_count = len(contexts)
            answer.retrieved = contexts
            if contexts and not answer.is_unknown:
                answer.citations = self._citations.build(
                    answer.answer, contexts, language=answer.language, primary_chunk_id=answer.primary_chunk_id
                )
            answer.total_ms = round((time.perf_counter() - started) * 1000, 2)
            yield ("citations", {"citations": [item.model_dump(mode="json") for item in answer.citations]})
            message_id = self._persist(cid, question, answer, analysis, contexts, debug, trace_id)
            log_event(
                "generation",
                "app.core.qa_engine",
                "QAEngine.stream",
                trace_id=trace_id,
                backend=self._generator.backend_name(),
                model=self._settings.llm.model,
                mode=answer.mode,
                prompt_chars=0,
                context_pages=[item.chunk.page for item in contexts],
                context_chunk_ids=[item.chunk.chunk_id for item in contexts],
                first_token_ms=answer.first_token_ms,
                total_ms=answer.total_ms,
                answer_chars=len(answer.answer),
                citations=[{"page": c.page, "chunk_id": c.chunk_id} for c in answer.citations],
                message_id=message_id,
            )
            yield ("done", {"answer": answer, "trace_id": trace_id})
        except RAGError as exc:
            logger.exception("app.core.qa_engine", "问答业务异常", code=exc.code, trace_id=trace_id)
            yield ("error", {"code": exc.code, "message": exc.user_message, "trace_id": trace_id})
            fallback = self._fallback_answer(exc.user_message, exc.code, analysis, trace_id, started)
            yield ("done", {"answer": fallback, "trace_id": trace_id})
        except Exception as exc:
            logger.exception("app.core.qa_engine", "问答未预期异常", trace_id=trace_id)
            yield ("error", {"code": "RAG_ERROR", "message": str(exc), "trace_id": trace_id})
            fallback = self._fallback_answer(
                self._settings.app.llm_unavailable_answer, "RAG_ERROR", analysis, trace_id, started
            )
            yield ("done", {"answer": fallback, "trace_id": trace_id})

    # ------------------------------------------------------------------
    def _topic_gate(
        self,
        contexts: list[RetrievedChunk],
        analysis: QueryAnalysis | None,
        question: str,
        trace_id: str,
    ) -> bool:
        """独立闸门（t13）：主题覆盖（rarest-first）+ 意图↔取值类型一致性。

        为什么独立成一层：含公司名/文档名的无关问题在"实义词覆盖"上天然很高
        （实测「…的食堂今天中午吃什么？」被答成"发行人基本情况 [页码: 52]"），
        必须单独判断"证据是否真的回答了问的那类信息"。
        """
        try:
            query = (analysis.search_query or analysis.original or question) if analysis else question
            intent = analysis.intent if analysis is not None else ""
            ok, reason = self._retriever.is_topic_covered(query, contexts, intent=intent)
            if not ok:
                logger.info(
                    "app.core.qa_engine",
                    "主题/答案类型一致性不通过，返回不清楚",
                    trace_id=trace_id,
                    query=query[:60],
                    intent=intent,
                    reason=reason,
                    top_pages=[item.chunk.page for item in contexts[:3]],
                )
            return ok
        except Exception:
            logger.exception("app.core.qa_engine", "主题一致性闸门异常，按可答处理（不误拒正常问题）")
            return True

    # ------------------------------------------------------------------
    def _fallback_answer(
        self,
        text: str,
        reason: str,
        analysis: QueryAnalysis | None,
        trace_id: str,
        started: float,
    ) -> Answer:
        """构造兜底回答。"""
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        return Answer(
            answer=text,
            is_unknown=True,
            unknown_reason=reason,
            first_token_ms=elapsed,
            total_ms=elapsed,
            mode="fallback",
            language=analysis.language if analysis is not None else "zh",
            trace_id=trace_id,
            query_analysis=analysis,
        )

    def _persist(
        self,
        conversation_id: str,
        question: str,
        answer: Answer,
        analysis: QueryAnalysis | None,
        contexts: list[RetrievedChunk],
        debug: Any,
        trace_id: str,
    ) -> int:
        """落库：助手消息 + 检索追踪（失败不影响已生成的答案，但必须记日志）。"""
        message_id = 0
        try:
            message_id = self._conversations.add_assistant_message(
                conversation_id, answer.answer, citations=answer.citations, first_token_ms=answer.first_token_ms
            )
        except Exception:
            logger.exception("app.core.qa_engine", "助手消息落库失败", trace_id=trace_id)
        try:
            self._store.add_retrieval_trace(
                RetrievalTrace(
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    message_id=message_id or None,
                    question=question,
                    rewritten_query=analysis.rewritten if analysis is not None else "",
                    variants=list(getattr(debug, "variants", []) or []),
                    candidates=[
                        {
                            "chunk_id": item.chunk.chunk_id,
                            "page": item.chunk.page,
                            "type": item.chunk.type,
                            "score": item.score,
                            "vector_score": item.vector_score,
                            "bm25_score": item.bm25_score,
                            "rerank_score": item.rerank_score,
                            "boosts": item.boosts,
                        }
                        for item in contexts
                    ],
                    boosts={
                        "boosted_table": getattr(debug, "boosted_table", 0),
                        "boosted_numeric": getattr(debug, "boosted_numeric", 0),
                        "boosted_keyword": getattr(debug, "boosted_keyword", 0),
                        "penalized_boilerplate": getattr(debug, "penalized_boilerplate", 0),
                        "penalized_fragment": getattr(debug, "penalized_fragment", 0),
                        "rerank_mode": getattr(debug, "rerank_mode", "off"),
                    },
                    final_pages=[item.chunk.page for item in contexts],
                    rerank_mode=getattr(debug, "rerank_mode", "off"),
                    top_cosine=float(getattr(debug, "top_cosine", 0.0)),
                    first_token_ms=answer.first_token_ms,
                    total_ms=answer.total_ms,
                    mode=answer.mode,
                )
            )
        except Exception:
            logger.exception("app.core.qa_engine", "检索追踪落库失败", trace_id=trace_id)
        return message_id

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """返回库表统计与当前状态。"""
        try:
            return {**self._store.stats(), "chunks_loaded": len(self._chunks), "doc_id": self._doc_id}
        except Exception:
            logger.exception("app.core.qa_engine", "统计失败")
            return {}

    def health(self) -> dict[str, Any]:
        """健康信息（界面与部署自检共用）。"""
        embedder = get_embedder()
        return {
            "app": {"name": self._settings.app.app_name, "version": self._settings.app.version},
            "embedder": embedder.health(),
            "index": self._retriever._store.health(),  # noqa: SLF001（健康检查需要内部状态）
            "bm25_docs": self._retriever._bm25.size,  # noqa: SLF001
            "llm": self._generator.health(),
            "reranker": self._retriever._reranker.health() if self._retriever._reranker else {"mode": "off"},  # noqa: SLF001
            "ready": self.ready,
            "doc_id": self._doc_id,
            "first_token_budget_seconds": self._settings.app.first_token_budget_seconds,
        }


_engine: QAEngine | None = None
_lock = threading.Lock()


def get_qa_engine(doc_id: str | None = None, force_extractive: bool = False) -> QAEngine:
    """获取进程级引擎单例。"""
    global _engine
    with _lock:
        if _engine is None or force_extractive:
            _engine = QAEngine(doc_id=doc_id, force_extractive=force_extractive)
    return _engine


def reset_qa_engine() -> None:
    """重置单例（测试用）。"""
    global _engine
    with _lock:
        _engine = None
