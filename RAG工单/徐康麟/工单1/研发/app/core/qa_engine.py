"""问答引擎：把解析、检索、生成、引用、对话、存储装配成一条完整链路。

对外只暴露一个类 ``QAEngine``，界面与测试脚本都通过它与系统交互。

单次问答链路（与工单第 5 节一一对应）::

    用户问题
      └─ Query 理解（意图/消歧/分解/多轮改写）        query_understanding.py
          └─ 混合检索（向量 + BM25 + 领域加权 + 页码过滤） retriever.py
              └─ 相关性判定 ──不足──> “不清楚” 兜底
                  └─ 生成（LLM 流式 / 抽取式降级）      generator.py
                      └─ 引用构造与校验                citation.py
                          └─ 落库（消息 / 对话）        conversation.py + sqlite

对外承诺：
- 任何异常都不会让调用方拿到半成品：要么给出答案，要么给出“不清楚”并记录原因；
- 首字返回时间可观测（``first_token_ms``）；
- 中间步骤只写日志，不返回给用户。
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from pathlib import Path

from app.core.bm25_index import BM25Index
from app.core.chunker import Chunker
from app.core.citation import CitationManager
from app.core.config import get_settings
from app.core.conversation import ConversationManager, get_conversation_manager
from app.core.embedder import get_embedder
from app.core.generator import Generator, get_generator
from app.core.logging_conf import logger, trace
from app.core.language import resolve_answer_language
from app.core.pdf_parser import PDFParser
from app.core.query_understanding import QueryUnderstanding
from app.core.retriever import Retriever
from app.core.vector_store import get_vector_store
from app.models.schemas import Answer, Chunk, Feedback, ParsedDocument, QueryAnalysis, RetrievedChunk
from app.storage.sqlite_manager import SQLiteManager, get_sqlite_manager

# 中文片段提取：英文语句里若混有中文线索（术语表桥接的结果），
# 只取中文部分参与“实义词覆盖”判定。
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def _chinese_part(text: str) -> str:
    """返回文本中的中文片段（以空格连接）；没有中文时返回空串。"""
    return " ".join(_CJK_RUN_RE.findall(text or ""))


def _variant_text(variants: list) -> str:
    """把查询变体列表拼成一段文本（变体可能是 ``str`` 或 ``(str, 类型)``）。"""
    parts: list[str] = []
    for item in variants:
        query = item[0] if isinstance(item, tuple) else item
        if query:
            parts.append(str(query))
    return " ".join(parts)


class QAEngine:
    """RAG 问答引擎。"""

    def __init__(
        self,
        doc_id: str | None = None,
        store: SQLiteManager | None = None,
        force_extractive: bool = False,
        auto_load_index: bool = True,
    ) -> None:
        self.settings = get_settings()
        self.store = store or get_sqlite_manager()
        self.doc_id = doc_id
        self._last_retrieved: list[RetrievedChunk] = []
        self._chunks: dict[str, Chunk] = {}

        # ---- 组件装配 ----
        self.embedder = get_embedder()
        self.vector_store = get_vector_store()
        self.bm25 = BM25Index()
        self.retriever = Retriever(embedder=self.embedder, vector_store=self.vector_store, bm25_index=self.bm25)
        self.generator: Generator = get_generator(force_extractive=force_extractive)
        self.understander = QueryUnderstanding()
        self.conversations = get_conversation_manager(self.store)
        self.citation = CitationManager()
        self.parser = PDFParser()
        self.chunker = Chunker()

        if auto_load_index:
            self.doc_id = self.doc_id or self._resolve_default_doc_id()
            if self.doc_id:
                self.load_index(self.doc_id)

    # ==================================================================
    # 索引
    # ==================================================================
    def _resolve_default_doc_id(self) -> str | None:
        """取默认文档 ID：优先标记为 default 的，其次最近一个已索引的。"""
        meta = self.store.get_default_document()
        if meta is None:
            documents = [doc for doc in self.store.list_documents() if doc.status == "indexed"]
            meta = documents[0] if documents else None
        if meta is None:
            logger.warning(
                "app.core.qa_engine",
                "数据库中没有可用文档，请先运行 scripts/build_index.py 建索引",
            )
            return None
        return meta.doc_id

    @trace
    def load_index(self, doc_id: str | None = None) -> bool:
        """从 SQLite + 磁盘索引装载检索所需数据。"""
        target = doc_id or self.doc_id
        if not target:
            logger.warning("app.core.qa_engine", "load_index 未指定 doc_id")
            return False

        chunks = self.store.get_chunks(target)
        if not chunks:
            logger.warning(
                "app.core.qa_engine",
                "该文档尚无分块，请先建索引",
                doc_id=target,
                hint="python scripts/build_index.py",
            )
            return False

        self.doc_id = target
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        meta = self.store.get_document(target)
        page_count = meta.page_count if meta else 0
        self.citation.set_valid_pages(range(1, page_count + 1) if page_count else set())

        loaded = self.retriever.load_index(chunks)
        logger.info(
            "app.core.qa_engine",
            "索引装载完成",
            doc_id=target,
            chunks=len(chunks),
            vector_store=self.vector_store.name,
            vector_count=self.vector_store.count(),
            bm25_documents=self.retriever.bm25.size,
            loaded_from_disk=loaded,
        )
        return True

    @trace
    def build_index(
        self,
        pdf_path: Path | str | None = None,
        doc_id: str | None = None,
        max_pages: int = 0,
        save_processed: bool = True,
    ) -> dict[str, object]:
        """解析 PDF -> 分块 -> 向量化 -> 建 BM25 -> 落库落盘。"""
        started = time.perf_counter()
        pdf = Path(pdf_path) if pdf_path else self.settings.paths.default_pdf
        document: ParsedDocument = self.parser.parse(pdf, doc_id=doc_id, max_pages=max_pages)
        if save_processed:
            self.parser.save(document)

        chunks = self.chunker.split(document)
        if not chunks:
            raise RuntimeError("分块结果为空，PDF 可能没有可提取文本（扫描件需先做 OCR）")
        self.chunker.save(chunks)

        # 向量 + BM25
        index_info = self.retriever.build_index(chunks)
        self.retriever.save_index()

        # SQLite
        from app.models.schemas import DocumentMeta

        self.store.upsert_document(
            DocumentMeta(
                doc_id=document.doc_id,
                title=document.title,
                source_path=document.source_path,
                page_count=document.page_count,
                chunk_count=len(chunks),
                table_count=len(document.tables),
                status="indexed",
                is_default=True,
            )
        )
        self.store.insert_chunks(chunks, replace_doc=True)

        self.doc_id = document.doc_id
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.citation.set_valid_pages(range(1, document.page_count + 1))

        elapsed = time.perf_counter() - started
        summary = {
            "doc_id": document.doc_id,
            "title": document.title,
            "pages": document.page_count,
            "tables": len(document.tables),
            "chunks": len(chunks),
            "vector_count": index_info.get("vector", 0),
            "dimension": index_info.get("dimension", 0),
            "embedder": index_info.get("embedder", ""),
            "elapsed_s": round(elapsed, 2),
        }
        logger.info("app.core.qa_engine", "建索引完成", **summary)
        return summary

    def warmup(self) -> dict[str, object]:
        """预热：把一次性开销从“用户第一次提问”挪到“服务启动时”。

        为什么必须做：本地嵌入模型（BGE）首次加载要数秒，分词器也要建缓存，
        若发生在第一次提问里，界面上的**首字响应时间**会显示 5~10 秒，
        直接违反工单“首字 < 3 秒”的验收标准。

        本方法执行一次编码，触发模型加载与缓存建立；幂等，可重复调用。

        Returns:
            预热耗时与状态，供健康检查与界面展示。
        """
        started = time.perf_counter()
        detail: dict[str, object] = {"ok": False}
        try:
            if not self._chunks:
                detail["reason"] = "索引未就绪"
                return detail
            # 触发嵌入模型加载（若尚未加载）
            self.embedder.encode(["预热"])
            # 触发 jieba 词典构建
            self.retriever.bm25.search("预热", top_k=1)
            # 触发向量库首次检索（numpy 后端会读盘）
            self.vector_store.search(self.embedder.encode_one("预热"), top_k=1)
            # 触发一次完整检索：把多路召回、加权合并等一次性开销也算进来
            self.retriever.retrieve_multi([("预热", "clean")])
            # 探测 LLM 服务并缓存结果。
            # 放在这里而不是第一次提问里：探测即使快速失败也要几百毫秒，
            # 若发生在用户提问路径上会直接体现在首字延迟里。
            detail["llm_available"] = self.generator.check_llm_available()
            detail["ok"] = True
            detail["embedder"] = self.embedder.name
            detail["semantic"] = self.embedder.is_semantic
        except Exception as exc:
            detail["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("app.core.qa_engine", "预热失败（不影响后续问答，但首字延迟会偏高）")
        detail["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        logger.info("app.core.qa_engine", "预热完成", **detail)
        return detail

    # ==================================================================
    # 问答
    # ==================================================================
    @trace
    def ask(
        self,
        question: str,
        conversation_id: str | None = None,
        allow_llm: bool = True,
        save: bool = True,
    ) -> Answer:
        """完整问答：检索 + 生成 + 引用 + 落库。"""
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        if not self._chunks:
            logger.warning("app.core.qa_engine", "索引为空，无法回答", question=question)
            return self._empty_index_answer(started, question)

        # ---- 1. 会话与历史 ----
        cid = self.conversations.get_or_create(conversation_id, doc_id=self.doc_id)
        history = self.conversations.history_pairs(cid)
        if save:
            self.conversations.add_user_message(cid, question)

        # ---- 2. Query 理解 ----
        analysis: QueryAnalysis = self.understander.analyze(question, history)
        search_query = self.understander.search_query(analysis)
        query_variants = self.understander.search_queries(analysis)

        # ---- 3. 检索（多查询变体，提升措辞鲁棒性）----
        retrieved = self.retriever.retrieve_multi(query_variants, analysis=analysis)
        self._last_retrieved = retrieved

        # ---- 3.1 复杂问题：对子问题补充检索并合并 ----
        if analysis.sub_questions:
            for sub_question in analysis.sub_questions:
                sub_analysis = self.understander.analyze(sub_question, history)
                extra = self.retriever.retrieve_multi(
                    self.understander.search_queries(sub_analysis), analysis=sub_analysis
                )
                retrieved = self._merge_retrieved(retrieved, extra)
            self._last_retrieved = retrieved
            logger.info(
                "app.core.qa_engine",
                "子问题检索合并完成",
                sub_questions=analysis.sub_questions,
                merged=len(retrieved),
            )

        # ---- 4. 相关性判定：不足即“不清楚” ----
        # 两道闸门：语义相似度（整体是否相关）+ 实义词覆盖（问的东西是否真在文里）
        #
        # 两个细节：
        # 1. 实义词覆盖必须用**检索查询串**而不是原问题：英文提问时原问题是英文，
        #    证据是中文，直接比对会把“已正确桥接并命中”的问题误判为无法回答；
        # 2. 英文链路里查询串会混入英文原问题（术语表未命中时），
        #    此时只校验其中的中文实义词，英文虚词不应参与判定。
        overlap_probe = _chinese_part(_variant_text(query_variants)) or search_query
        if not self.retriever.is_confident(retrieved) or not self.retriever.is_answerable(
            overlap_probe, retrieved
        ):
            answer = self.generator._unknown(
                "检索结果相关度不足", started, retrieved, language=analysis.language
            )
            # 兜底路径同样回填 Query 理解结果：便于排查“为什么这个问题被判为无关”，
            # 也让调用方（界面/测试）拿到的 Answer 结构保持一致。
            answer.query_analysis = analysis
            answer.retrieved_count = len(retrieved)
            answer.pages = sorted({item.chunk.page for item in retrieved})
            answer.total_ms = round((time.perf_counter() - started) * 1000, 2)
            self._persist_answer(cid, answer, save)
            return answer

        # ---- 5. 生成 ----
        answer = self.generator.generate(
            question=analysis.original,
            contexts=retrieved,
            analysis=analysis,
            history=history,
            allow_llm=allow_llm,
        )

        # ---- 6. 引用 ----
        # 生成器已按“显式引用 > 主证据 > 高分补充”的优先级构造过引用；
        # 这里只在缺失时重建，避免把精准引用覆盖成“全部检索片段”。
        if not answer.is_unknown and not answer.citations:
            primary = next(
                (item for item in retrieved if item.chunk.chunk_id == answer.primary_chunk_id), None
            )
            rebuilt = self.citation.build(answer.answer, retrieved, primary=primary)
            if rebuilt:
                answer.citations = rebuilt
        answer.query_analysis = analysis
        answer.total_ms = round((time.perf_counter() - started) * 1000, 2)
        answer.retrieved_count = len(retrieved)
        answer.pages = sorted({item.chunk.page for item in retrieved})

        validation = self.citation.validate(answer)
        logger.info(
            "app.core.qa_engine",
            "问答完成",
            question=question,
            intent=analysis.intent,
            mode=answer.mode,
            is_unknown=answer.is_unknown,
            first_token_ms=answer.first_token_ms,
            total_ms=answer.total_ms,
            citations=validation["total"],
            citation_valid=validation["valid"],
            retrieved=len(retrieved),
            pages=answer.pages,
        )

        # ---- 7. 落库 ----
        self._persist_answer(cid, answer, save)
        if save:
            self.conversations.auto_title(cid, question)
        return answer

    # ------------------------------------------------------------------
    def stream(
        self,
        question: str,
        conversation_id: str | None = None,
        allow_llm: bool = True,
    ) -> Iterator[tuple[str, object]]:
        """流式问答，产出 ``(事件, 负载)``。

        事件：
        - ``first_token``：``{"first_token_ms": float}``，用于首字延迟展示
        - ``delta``：``{"text": str}``，增量文本
        - ``done``：``{"answer": Answer}``，最终完整答案（含引用）
        - ``error``：``{"message": str}``
        """
        started = time.perf_counter()
        try:
            question = (question or "").strip()
            if not question:
                raise ValueError("问题不能为空")
            if not self._chunks:
                answer = self._empty_index_answer(started, question)
                yield ("done", {"answer": answer})
                return

            cid = self.conversations.get_or_create(conversation_id, doc_id=self.doc_id)
            history = self.conversations.history_pairs(cid)
            self.conversations.add_user_message(cid, question)

            analysis = self.understander.analyze(question, history)
            search_query = self.understander.search_query(analysis)
            query_variants = self.understander.search_queries(analysis)
            retrieved = self.retriever.retrieve_multi(query_variants, analysis=analysis)
            if analysis.sub_questions:
                for sub_question in analysis.sub_questions:
                    sub_analysis = self.understander.analyze(sub_question, history)
                    extra = self.retriever.retrieve_multi(
                        self.understander.search_queries(sub_analysis), analysis=sub_analysis
                    )
                    retrieved = self._merge_retrieved(retrieved, extra)
            self._last_retrieved = retrieved

            overlap_probe = _chinese_part(_variant_text(query_variants)) or search_query
            if not self.retriever.is_confident(retrieved) or not self.retriever.is_answerable(
                overlap_probe, retrieved
            ):
                answer = self.generator._unknown(
                    "检索结果相关度不足", started, retrieved, language=analysis.language
                )
                answer.query_analysis = analysis
                answer.retrieved_count = len(retrieved)
                answer.pages = sorted({item.chunk.page for item in retrieved})
                answer.total_ms = round((time.perf_counter() - started) * 1000, 2)
                self._persist_answer(cid, answer, True)
                yield ("done", {"answer": answer})
                return

            # LLM 流式；不可用时直接走抽取式并一次性吐出
            if allow_llm and not self.generator.force_extractive and self.generator.check_llm_available():
                collected: list[str] = []
                final: Answer | None = None
                # 裁剪送入 LLM 的片段数（提示词越长 prefill 越慢）。
                # 引用仍基于完整 retrieved，所以裁剪不会减少可追溯的来源。
                limit = self.settings.llm.max_context_chunks
                llm_contexts = retrieved[:limit] if limit > 0 else retrieved
                for event, payload in self.generator.stream(analysis.original, llm_contexts, history):
                    if event == "first_token":
                        yield ("first_token", payload)
                    elif event == "delta":
                        collected.append(str(payload["text"]))  # type: ignore[index]
                        yield ("delta", payload)
                    elif event == "done":
                        final = payload  # type: ignore[assignment]
                    elif event == "error":
                        logger.warning("app.core.qa_engine", "流式生成中断，改用抽取式", error=str(payload))
                        break
                if final is not None:
                    answer = final
                else:
                    answer = self.generator.generate_extractive(
                        analysis.original, retrieved, analysis, started
                    )
                    yield ("first_token", {"first_token_ms": answer.first_token_ms})
                    yield ("delta", {"text": answer.answer})
            else:
                answer = self.generator.generate_extractive(analysis.original, retrieved, analysis, started)
                yield ("first_token", {"first_token_ms": answer.first_token_ms})
                yield ("delta", {"text": answer.answer})

            if not answer.is_unknown and not answer.citations:
                primary = next(
                    (item for item in retrieved if item.chunk.chunk_id == answer.primary_chunk_id), None
                )
                rebuilt = self.citation.build(answer.answer, retrieved, primary=primary)
                if rebuilt:
                    answer.citations = rebuilt
            answer.query_analysis = analysis
            answer.total_ms = round((time.perf_counter() - started) * 1000, 2)
            answer.retrieved_count = len(retrieved)
            answer.pages = sorted({item.chunk.page for item in retrieved})

            self._persist_answer(cid, answer, True)
            self.conversations.auto_title(cid, question)
            yield ("done", {"answer": answer})
        except Exception as exc:
            logger.exception("app.core.qa_engine", "流式问答异常", error=f"{type(exc).__name__}: {exc}")
            yield ("error", {"message": f"{type(exc).__name__}: {exc}"})

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_retrieved(
        base: list[RetrievedChunk], extra: list[RetrievedChunk], limit: int = 8
    ) -> list[RetrievedChunk]:
        """合并两批检索结果：同一 chunk 取较高分，最后按分数截断。"""
        merged: dict[str, RetrievedChunk] = {}
        for item in list(base) + list(extra):
            current = merged.get(item.chunk.chunk_id)
            if current is None or item.score > current.score:
                merged[item.chunk.chunk_id] = item
        ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ordered[:limit]

    def _empty_index_answer(self, started: float, question: str = "") -> Answer:
        """索引为空时的统一回复（语言跟随提问）。"""
        language = resolve_answer_language(
            question, self.settings.language.default_answer_language
        ) if question else "zh"
        answer = self.generator._unknown(
            "索引为空，请先运行 研发/scripts/build_index.py", started, language=language
        )
        answer.total_ms = round((time.perf_counter() - started) * 1000, 2)
        return answer

    def _persist_answer(self, conversation_id: str, answer: Answer, save: bool) -> None:
        """把助手回复写入 SQLite（失败只记日志，不影响回答返回）。"""
        if not save:
            return
        try:
            self.conversations.add_assistant_message(
                conversation_id, answer.answer, citations=answer.citations, first_token_ms=answer.first_token_ms
            )
        except Exception as exc:
            logger.exception(
                "app.core.qa_engine", "助手消息落库失败", error=f"{type(exc).__name__}: {exc}"
            )

    # ==================================================================
    # 供界面调用
    # ==================================================================
    def new_conversation(self, title: str = "新对话") -> str:
        return self.conversations.new_conversation(title=title, doc_id=self.doc_id)

    def list_conversations(self) -> list:
        return self.conversations.list_conversations()

    def switch_conversation(self, conversation_id: str) -> None:
        """切换会话（引擎无状态，此处只做存在性校验与日志）。"""
        conversation = self.conversations.get(conversation_id)
        if conversation is None:
            logger.warning("app.core.qa_engine", "切换会话失败：不存在", conversation_id=conversation_id)
            raise KeyError(f"会话不存在: {conversation_id}")
        logger.info("app.core.qa_engine", "切换会话", conversation_id=conversation_id)

    def clear_conversation(self, conversation_id: str) -> int:
        return self.conversations.clear(conversation_id)

    def get_messages(self, conversation_id: str) -> list:
        return self.conversations.messages(conversation_id)

    def last_retrieved(self) -> list[RetrievedChunk]:
        """最近一次检索结果（仅供界面"检索片段/页码"展示与调试）。"""
        return self._last_retrieved

    def submit_feedback(
        self,
        conversation_id: str,
        message_id: int | None,
        rating: str,
        comment: str = "",
        question: str = "",
    ) -> int:
        """提交点赞/点踩反馈。"""
        if rating not in {"up", "down"}:
            raise ValueError(f"rating 必须是 up 或 down，收到: {rating!r}")
        feedback_id = self.store.add_feedback(
            Feedback(
                conversation_id=conversation_id,
                message_id=message_id,
                rating=rating,  # type: ignore[arg-type]
                comment=comment,
                question=question,
            )
        )
        logger.info(
            "app.core.qa_engine",
            "收到用户反馈",
            conversation_id=conversation_id,
            rating=rating,
            comment=comment,
            feedback_id=feedback_id,
        )
        return feedback_id

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, object]:
        """知识库与运行时统计，供界面侧边栏展示。"""
        db_stats = self.store.stats()
        meta = self.store.get_document(self.doc_id) if self.doc_id else None
        return {
            "doc_id": self.doc_id or "",
            "doc_title": meta.title if meta else "",
            "pages": meta.page_count if meta else 0,
            "tables": meta.table_count if meta else 0,
            "chunks": len(self._chunks),
            "vector_count": self.vector_store.count(),
            "bm25_documents": self.retriever.bm25.size,
            "embedder": self.embedder.name,
            "embedder_semantic": self.embedder.is_semantic,
            "vector_backend": self.vector_store.name,
            "conversations": db_stats.get("conversations", 0),
            "messages": db_stats.get("messages", 0),
            "feedback": db_stats.get("feedback", 0),
            "index_ready": bool(self._chunks),
            "llm_base_url": self.settings.llm.base_url,
            "llm_model": self.settings.llm.model,
        }

    def health(self) -> dict[str, object]:
        """健康检查：各组件的可用性与降级原因。"""
        return {
            "engine": "ok" if self._chunks else "no_index",
            "retriever": self.retriever.health(),
            "stats": self.stats(),
            "unknown_answer": self.settings.app.unknown_answer,
            "first_token_budget_seconds": self.settings.app.first_token_budget_seconds,
        }


_engine: QAEngine | None = None


def get_qa_engine(doc_id: str | None = None, force_extractive: bool = False) -> QAEngine:
    """工厂函数：获取进程级单例引擎。"""
    global _engine
    if _engine is None or doc_id is not None:
        _engine = QAEngine(doc_id=doc_id, force_extractive=force_extractive)
    return _engine


def reset_qa_engine() -> None:
    """重置单例（测试用）。"""
    global _engine
    _engine = None
