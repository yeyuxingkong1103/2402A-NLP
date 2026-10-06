# -*- coding: utf-8 -*-
"""工单3 端到端问答引擎（设计/接口设计.md §3.22 冻结，**UI 唯一入口**）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

流程（严格按 v1.6 冻结顺序，分类一律取自本轮原文）：
    ① ``classify_question(本轮原文)`` → ② ``rewrite_query(本轮原文, history)``（**仅供检索**）
    → ③ ``classify_subject_expectation(strip_issuer_names(本轮原文))``
    → 检索（``HybridRetriever``，含数值锚定支持块）→ 可答性闸门 → 生成 + 引用 + 主体闸门
    → 落库（会话/消息/检索轨迹/运行）→ 返回 ``Answer``。

``warmup()`` 必须依次调用：``text_utils.warmup_tokenizer()`` → ``embedder.warmup()``
→ ``llm_client.probe_backends()``（0.5 s 无重试）→ 一次生成预热（可选）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from ..storage.sqlite_manager import SQLiteManager, get_sqlite_manager
from . import citation as citation_mod, embedder, language as language_mod, llm_client, text_utils
from .config import AppConfig, discover_pdfs, get_config
from .conversation import ConversationStore, Turn, get_conversation_store
from .errors import RagError, wrap
from .generator import Answer, AnswerDelta, AnswerGenerator, build_generator
from .query_understanding import understand
from .retriever import HybridRetriever, RetrievalResult, build_retriever
from .text_utils import text_digest

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"


def _lazy_logger(logger: Any, module: str = "qa_engine") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class EngineHealth:
    """引擎健康摘要（``/api/health`` 与 UI 侧栏共用）。"""

    ok: bool
    index: dict[str, Any] = field(default_factory=dict)      # §7 契约：{count,dim,model}
    llm: dict[str, Any] = field(default_factory=dict)
    files: int = 0                                           # §7 契约：文件**数量**
    retriever: dict[str, Any] = field(default_factory=dict)
    storage: dict[str, Any] = field(default_factory=dict)
    file_list: list[dict[str, Any]] = field(default_factory=list)
    warmup: dict[str, Any] = field(default_factory=dict)
    checked_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """契约字段（ok/index/llm/files）+ 附加字段（retriever/storage/file_list/warmup）。"""
        return {"ok": self.ok, "index": self.index, "llm": self.llm, "files": self.files,
                "retriever": self.retriever, "storage": self.storage, "file_list": self.file_list,
                "warmup": self.warmup, "checked_at": self.checked_at}


class QAEngine:
    """端到端问答引擎（检索 + 生成 + 引用 + 多轮 + 持久化）。"""

    def __init__(self, *, cfg: AppConfig | None = None, logger: Any = None,
                 retriever: HybridRetriever | None = None, generator: AnswerGenerator | None = None,
                 store: ConversationStore | None = None, sqlite: SQLiteManager | None = None) -> None:
        self.cfg = cfg or get_config()
        self.log = _lazy_logger(logger)
        self.retriever = retriever or build_retriever(cfg=self.cfg)
        self.generator = generator or build_generator(cfg=self.cfg, logger=self.log)
        self.store = store or get_conversation_store(cfg=self.cfg, logger=self.log)
        self.sqlite = sqlite or get_sqlite_manager(cfg=self.cfg, logger=self.log)
        self.warmup_info: dict[str, Any] = {}
        self._warmed = False

    # -- 预热 ------------------------------------------------------------
    def warmup(self) -> dict[str, Any]:
        """启动预热（硬要求）：分词器 → 嵌入 → 后端探测 →（可选）一次生成。"""
        started = time.perf_counter()
        with self.log.enter("QAEngine.warmup", {"cfg_llm": self.cfg.llm.ollama_gen_model}) as span:
            info: dict[str, Any] = {}
            t0 = time.perf_counter()
            tokenizer_info = text_utils.warmup_tokenizer(logger=self.log)
            info["tokenizer"] = tokenizer_info
            info["tokenizer_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            t0 = time.perf_counter()
            try:
                info["embedding"] = embedder.warmup(self.cfg, logger=self.log)
            except Exception as exc:  # noqa: BLE001 —— 嵌入不可用时显式降级（检索将走 BM25-only）
                self.log.log_event("qa.warmup_embed_failed", level="ERROR", error_type=type(exc).__name__,
                                   message=str(exc), degrade="检索退化为 BM25-only")
                info["embedding"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            info["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            t0 = time.perf_counter()
            try:
                backends = llm_client.probe_backends(self.cfg, logger=self.log)
                info["backends"] = [b.to_dict() for b in backends]
                active = next((b for b in backends if b.available), None)
                info["llm_probe_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                info["backend"] = None if active is None else active.name
            except Exception as exc:  # noqa: BLE001 —— 探测失败不阻断启动（生成侧还有抽取式兜底）
                self.log.log_event("qa.warmup_probe_failed", level="ERROR", error_type=type(exc).__name__,
                                   message=str(exc), degrade="生成侧按 extractive 兜底")
                info["llm_probe_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                info["backends"] = []
                info["backend"] = "extractive"
            # 检索器懒加载（内含 jieba + 嵌入预热）
            self.retriever._ensure_loaded() if hasattr(self.retriever, "_ensure_loaded") else self.retriever.load()
            # 生成预热（t17，§21.4）：把「生成模型加载」的开销从**用户第一个问题**挪到启动阶段。
            # 实测（本机 Ollama + qwen2.5:3b，模型被驱逐后，工单3 14 题评估）：
            #   不预热 → 首题首字 4304.4 ms（超 3000 预算），整轮均值 1185.0 ms（其余题 763~1272 ms）；
            #   预热一次极小生成后 → 首题回到 ~120 ms 量级（Arm C2 实测，见 部署/配置/环境事实.md §21）。
            t0 = time.perf_counter()
            info["generation"] = self._warmup_generation()
            info["generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            self._warmed = True
            info["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
            self.warmup_info = info
            self.log.log_event("qa.warmup", **{k: v for k, v in info.items() if k != "backends"})
            span.set_output({k: v for k, v in info.items() if k != "backends"})
            return info

    def _warmup_generation(self) -> dict[str, Any]:
        """一次极小生成（``max_tokens=1``）：让生成模型先被加载，避免**用户首题**承担模型加载开销（t17，§21.4）。

        实测根因（本机 Ollama + ``qwen2.5:3b``）：``warmup()`` 覆盖了分词器 / 嵌入 / 后端探测，
        但**没有真正发起过一次生成** —— 模型被驱逐（keep-alive 到期、Ollama 重启、换模型）后，
        第一个用户问题要等模型从磁盘加载，首字实测 **4304.4 ms**（超 3000 ms 预算）。

        纪律（硬要求「禁止静默失败」）：
            * 成功 → ``qa.warmup_generation``（含 ``first_token_ms`` / ``total_ms``）；
            * 失败 → ``qa.warmup_generation_failed``（ERROR + 异常类型/消息）+ 明确写出降级后果
              （首题将承担加载开销），**绝不阻断启动**（生成侧还有抽取式兜底）；
            * 后端不支持生成（如 extractive）→ ``qa.warmup_generation_skipped``（WARNING）。
        超时由 ``RAG_LLM__REQUEST_TIMEOUT_S``（默认 20 s）兜住，不额外引线程。
        """
        with self.log.enter("QAEngine._warmup_generation", {"cfg_llm": self.cfg.llm.ollama_gen_model}) as span:
            client = getattr(self.generator, "llm", None)
            backend = getattr(getattr(client, "backend", None), "name", None)
            if client is None or backend in (None, "extractive"):
                payload: dict[str, Any] = {"ok": False, "skipped": True, "backend": backend,
                                           "reason": "当前后端无生成模型（extractive 或未装配客户端）"}
                self.log.log_event("qa.warmup_generation_skipped", level="WARNING", **payload)
                span.set_output(payload)
                return payload
            try:
                result = client.generate_full("预热：只回答「好」。", max_tokens=1, temperature=0.0,
                                              logger=self.log)
            except Exception as exc:  # noqa: BLE001 —— 预热失败必须留痕并显式降级，绝不阻断启动
                message = f"{type(exc).__name__}: {exc}"
                payload = {"ok": False, "backend": backend, "error": message,
                           "degrade": "预热失败不影响可用性：首题将承担模型加载开销"}
                self.log.log_event("qa.warmup_generation_failed", level="ERROR", error_type=type(exc).__name__,
                                   message=str(exc), degrade=payload["degrade"])
                span.set_output(payload)
                return payload
            payload = {"ok": True, "backend": backend, "first_token_ms": float(result.first_token_ms or 0.0),
                       "total_ms": float(result.total_ms or 0.0), "chars": len(str(result.text or ""))}
            self.log.log_event("qa.warmup_generation", **payload)
            span.set_output(payload)
            return payload

    # -- 文件 / 健康 ------------------------------------------------------
    def files(self) -> list[dict[str, Any]]:
        """自动发现的 PDF 列表（UI 下拉用；来自 ``discover_pdfs()``，禁止硬编码）。"""
        with self.log.enter("QAEngine.files", {}) as span:
            docs = {d["file_name"]: d for d in self.sqlite.list_documents()}
            out: list[dict[str, Any]] = []
            for pdf in discover_pdfs(logger=self.log):
                name = str(getattr(pdf, "file_name", "") or getattr(pdf, "path", ""))
                name = name.split("\\")[-1].split("/")[-1] or name
                meta = docs.get(name, {})
                out.append({"file_name": name,
                            "page_count": int(meta.get("page_count") or getattr(pdf, "page_count", 0) or 0),
                            "size_bytes": int(meta.get("size_bytes") or getattr(pdf, "size_bytes", 0) or 0),
                            "chunk_count": self.sqlite.count_chunks(file_name=name),
                            "parsed": name in docs})
            self.log.log_event("qa.files", count=len(out), names=[f["file_name"] for f in out])
            span.set_output({"files": len(out)})
            return out

    def health(self) -> dict[str, Any]:
        """健康检查（§7 冻结 HTTP 契约形状 + UI 附加信息）。

        契约（``设计/接口设计.md`` §7）：
            ``{"ok":true,"index":{"count":N,"dim":1024,"model":"bge-m3:latest"},
                "llm":{"backend":…,"model":…,"available":true},"files":2}``
        注意 ``files`` 是**数量（int）**，不是列表；文件明细放在附加字段 ``file_list``（UI 用）。
        t14 修复：此前缺 ``index``、``files`` 误为列表、``llm`` 缺 ``available`` → 与契约不一致。
        """
        with self.log.enter("QAEngine.health", {}) as span:
            try:
                retriever_health = self.retriever.health()
            except Exception as exc:  # noqa: BLE001 —— 健康检查本身不得抛
                self.log.log_event("qa.health_retriever_failed", level="ERROR",
                                   error_type=type(exc).__name__, message=str(exc))
                retriever_health = {"error": f"{type(exc).__name__}: {exc}"}
            files = self.files()
            backend = getattr(getattr(self.generator, "llm", None), "backend", None)
            payload = EngineHealth(
                ok=not retriever_health.get("error"),
                index={"count": int(retriever_health.get("chunks") or 0),
                       "dim": int(retriever_health.get("dim") or 0),
                       "model": str(retriever_health.get("model") or self.cfg.llm.ollama_embed_model),
                       "bm25_count": int(retriever_health.get("bm25_chunks") or 0),
                       "bm25_vocab": int(retriever_health.get("bm25_vocab") or 0)},
                retriever=retriever_health,
                llm={"backend": getattr(backend, "name", "extractive"),
                     "model": getattr(backend, "model", "rule-based"),
                     "available": backend is not None,
                     "base_url": getattr(backend, "base_url", "")},
                storage=self.sqlite.to_dict(), files=len(files), file_list=files,
                warmup=self.warmup_info, checked_at=_now_iso())
            span.set_output({"ok": payload.ok, "index_count": payload.index.get("count")})
            return payload.to_dict()

    # -- HTTP 契约辅助（t14：两个界面共用同一份载荷构造，禁止各写一套）----
    def validate_files(self, file_names: Sequence[str] | None) -> list[str] | None:
        """校验请求里的 ``file_names``：不在语料内 → 抛 ``RagError``（HTTP 层转 400，§7 约定）。

        契约（§7 状态码表）：``400 = 入参非法（空问题、file_names 含不存在的文件）``。
        实测缺陷（t14）：不存在的文件名此前返回 200（静默空检索），与契约不符。
        """
        if not file_names:
            return None
        known = {row["file_name"] for row in self.files()}
        unknown = [str(name) for name in file_names if str(name) not in known]
        if unknown:
            self.log.log_event("qa.invalid_file_names", level="WARNING", unknown=unknown,
                               known=sorted(known), fallback="返回 400（契约 §7：入参非法）")
            raise RagError(f"file_names 含不存在的文件：{unknown}；语料内文件：{sorted(known)}",
                           code="RAG-6001", stage="ui", detail={"unknown": unknown})
        return [str(name) for name in file_names]

    @staticmethod
    def chunk_summary(chunks: Sequence[Any], *, content_limit: int = 120) -> list[dict[str, Any]]:
        """把块列表转成契约要求的摘要（``chunks[].content_digest = {chars, head}``，§7）。"""
        rows: list[dict[str, Any]] = []
        for rank, chunk in enumerate(chunks, start=1):
            content = str(getattr(chunk, "content", "") or "")
            rows.append({
                "rank": rank,
                "chunk_id": str(getattr(chunk, "chunk_id", "") or ""),
                "file_name": str(getattr(chunk, "file_name", "") or ""),
                "page": int(getattr(chunk, "page", 0) or 0),
                "type": str(getattr(chunk, "type", "") or ""),
                "score": float(getattr(chunk, "score", 0.0) or 0.0),
                "content_digest": {"chars": len(content), "head": content[:content_limit]},
                "preview": content[:600],                 # 附加：UI 展示用（契约字段不变）
            })
        return rows

    def answer_payload(self, answer: Answer, *, session_id: str | None, wall_ms: float) -> dict[str, Any]:
        """按 §7 契约构造 ``/api/ask`` 响应体（**唯一实现**，FastAPI 与标准库界面共用）。"""
        detection = getattr(answer, "retrieval", None)
        top_chunks = list(getattr(detection, "chunks", []) or []) if detection is not None else []
        support_chunks = list(getattr(detection, "support_chunks", []) or []) if detection is not None else []
        return {
            "answer_id": answer.answer_id,
            "text": answer.text,                       # 契约字段名（t14 修复：此前误叫 answer）
            "citations": [c.to_dict() for c in answer.citations],
            "is_unknown": bool(answer.is_unknown),
            "unknown_reason": answer.unknown_reason,
            "first_token_ms": answer.first_token_ms,
            "total_ms": answer.total_ms,
            "backend": answer.backend,
            "model": answer.model,
            "language": answer.language,
            "session_id": session_id,
            "trace_id": answer.trace_id,
            # 附加：闸门审计（在线/用户套件断言）。Answer 上存的是 SubjectGateResult 对象，
            # 载荷里序列化为 dict 便于 HTTP/JSON 消费。
            "subject_gate": (lambda g: (g.to_dict() if hasattr(g, "to_dict") else g))(
                getattr(answer, "subject_gate", None)),
            "chunks": self.chunk_summary(top_chunks),                # 契约字段
            "support_chunks": self.chunk_summary(support_chunks),    # 附加：支持块（UI 展示）
            "retrieval_stages": (dict(getattr(detection, "stages", {}) or {}) if detection is not None else {}),
            "wall_ms": wall_ms,
        }

    def stream_event(self, delta: AnswerDelta, *, index: int) -> dict[str, Any]:
        """把流式 delta 转成契约的 SSE 事件体（§7：``{"text":…,"index":N}``；末事件为完整答案）。"""
        if delta.done and delta.answer is not None:
            payload = self.answer_payload(delta.answer, session_id=None, wall_ms=0.0)
            payload.pop("wall_ms", None)
            payload.update({"done": True, "index": index})
            return payload
        event: dict[str, Any] = {"text": delta.text, "index": index, "done": False}
        if delta.is_first:
            event["first_token_ms"] = delta.first_token_ms
        return event

    # -- 检索 ------------------------------------------------------------
    def retrieve_only(self, question: str, *, file_names: Sequence[str] | None = None,
                      top_k: int | None = None, history: Sequence[Turn] | None = None,
                      trace_id: str | None = None) -> RetrievalResult:
        """只检索（UI 展示片段与页码、测试用）。"""
        with self.log.enter("QAEngine.retrieve_only",
                            {"question": question[:60], "files": list(file_names or []), "top_k": top_k}) as span:
            info = understand(question, history, cfg=self.cfg, llm=None, logger=self.log)
            # t21（§24）：键名统一 —— understand() 返回的键是 ``rewritten``，此前调用方读 ``rewritten_query``
            # 恒为 None → **多轮改写算出来了却从未用于检索**（A4 根因）。
            result = self.retriever.retrieve(question, top_k=top_k, file_names=file_names,
                                            rewritten_query=_rewritten_query(info),
                                            trace_id=trace_id, logger=self.log)
            span.set_output({"chunks": len(result.chunks), "support": len(result.support_chunks)})
            return result

    # -- 问答 ------------------------------------------------------------
    def ask(self, question: str, *, session_id: str | None = None,
            file_names: Sequence[str] | None = None, top_k: int | None = None,
            stream: bool = False, logger: Any = None) -> Answer | Iterator[AnswerDelta]:
        """问答主入口；``stream=True`` 时返回增量迭代器（末个 delta 携带完整 ``Answer``）。"""
        log = logger or self.log
        if not self._warmed:
            self.warmup()
        trace_id = f"q{int(time.perf_counter() * 1000) % 100000000:08d}"
        started = time.perf_counter()
        with log.enter("QAEngine.ask", {"question": question[:60], "session_id": session_id,
                                        "files": list(file_names or []), "stream": stream,
                                        "trace_id": trace_id}) as span:
            try:
                sid = session_id or self.store.create_session()
                history = self.store.history(sid, last_n=5)
                self.store.append_turn(sid, Turn(session_id=sid, role="user", content=question,
                                                 created_at=_now_iso()))
                log.log_event("qa.ask.start", trace_id=trace_id, question_digest=text_digest(question, limit=60),
                              session_id=sid, file_names=list(file_names or []), history_turns=len(history))
                info = understand(question, history, cfg=self.cfg, llm=None, logger=log)
                rewritten = _rewritten_query(info)
                log.log_event("qa.understand", trace_id=trace_id, classified_from="original",
                              field_type=info.get("field_type"), expects_numeric=info.get("expects_numeric"),
                              rewritten_query=rewritten)
                retrieval = self.retriever.retrieve(question, top_k=top_k, file_names=file_names,
                                                    rewritten_query=rewritten,
                                                    trace_id=trace_id, logger=log)
                self.sqlite.record_retrieval({
                    "trace_id": trace_id, "session_id": sid, "question": question,
                    "rewritten_query": rewritten, "file_names": list(file_names or []),
                    "top_k": int(top_k or self.cfg.retrieval.top_k),
                    "chunk_ids": [c.chunk_id for c in retrieval.chunks],
                    "scores": [c.score for c in retrieval.chunks], "stages": retrieval.stages,
                })
                if stream:
                    span.set_output({"stream": True, "chunks": len(retrieval.chunks)})
                    return self._stream_answer(question, retrieval, sid, history, trace_id, started, log)
                answer = self.generator.answer(question, retrieval, history=history, trace_id=trace_id, logger=log)
                self._persist_answer(sid, answer)
                self.sqlite.record_run(run_id=trace_id, kind="qa.ask", ok=not answer.is_unknown,
                                       stats={"first_token_ms": answer.first_token_ms,
                                              "total_ms": answer.total_ms,
                                              "citations": len(answer.citations),
                                              "backend": answer.backend})
                log.log_event("qa.ask.done", trace_id=trace_id, is_unknown=answer.is_unknown,
                              first_token_ms=answer.first_token_ms, total_ms=answer.total_ms,
                              citations=[c.render() for c in answer.citations], backend=answer.backend,
                              wall_ms=round((time.perf_counter() - started) * 1000, 2))
                span.set_output({"unknown": answer.is_unknown, "citations": len(answer.citations),
                                 "first_token_ms": answer.first_token_ms})
                return answer
            except Exception as exc:  # noqa: BLE001 —— 统一转 RagError（日志含堆栈），UI 显示「不清楚」
                log.log_event("qa.ask.error", level="ERROR", trace_id=trace_id,
                              error_type=type(exc).__name__, message=str(exc))
                if isinstance(exc, RagError):
                    raise
                raise wrap(exc, code="RAG-5000", stage="qa", trace_id=trace_id,
                           question=question[:80]) from exc

    def _stream_answer(self, question: str, retrieval: RetrievalResult, sid: str, history: Sequence[Turn],
                       trace_id: str, started: float, log: Any) -> Iterator[AnswerDelta]:
        """流式作答包装：末个 delta 落库（会话/消息/运行）并返回完整 ``Answer``。"""
        with log.enter("QAEngine._stream_answer", {"trace_id": trace_id}) as span:
            final: Answer | None = None
            for delta in self.generator.stream(question, retrieval, history=history, trace_id=trace_id,
                                               logger=log):
                if delta.done and delta.answer is not None:
                    final = delta.answer
                yield delta
            if final is not None:
                self._persist_answer(sid, final)
                self.sqlite.record_run(run_id=trace_id, kind="qa.ask.stream", ok=not final.is_unknown,
                                       stats={"first_token_ms": final.first_token_ms,
                                              "total_ms": final.total_ms, "citations": len(final.citations)})
                log.log_event("qa.ask.done", trace_id=trace_id, stream=True, is_unknown=final.is_unknown,
                              first_token_ms=final.first_token_ms, total_ms=final.total_ms,
                              citations=[c.render() for c in final.citations],
                              wall_ms=round((time.perf_counter() - started) * 1000, 2))
                span.set_output({"unknown": final.is_unknown, "citations": len(final.citations)})

    def _persist_answer(self, session_id: str, answer: Answer) -> None:
        """把助手回答写入会话（引用按 ``Citation`` 对象传给 store，由 store 负责序列化）。"""
        self.store.append_turn(session_id, Turn(
            session_id=session_id, role="assistant", content=answer.text,
            citations=list(answer.citations), first_token_ms=answer.first_token_ms,
            total_ms=answer.total_ms, created_at=_now_iso()))

    # -- 界面辅助 --------------------------------------------------------
    def citation_source(self, file_name: str, page: int, *, limit: int = 800) -> str:
        """取引用页原文（UI「可展开的引用原文」用）。"""
        with self.log.enter("QAEngine.citation_source", {"file_name": file_name, "page": page}) as span:
            text = self.generator.page_text_lookup(file_name, int(page)) or ""
            span.set_output({"chars": len(text)})
            return text[:limit]

    def sessions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """会话列表。"""
        return self.store.list_sessions(limit=limit)

    def messages(self, session_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """会话消息（UI 多轮历史）。"""
        turns = self.store.history(session_id, last_n=max(1, limit // 2))
        return [t.to_dict() for t in turns]

    def feedback(self, *, rating: str, session_id: str | None = None, message_id: int | None = None,
                 answer_id: str | None = None, trace_id: str | None = None,
                 comment: str | None = None) -> int:
        """点赞/点踩（落 ``feedback`` 表）。"""
        return self.sqlite.record_feedback(rating=rating, session_id=session_id, message_id=message_id,
                                           answer_id=answer_id, trace_id=trace_id, comment=comment)

    def clear_conversation(self, session_id: str) -> int:
        """清空会话（UI「清空对话」）。"""
        removed = self.store.clear(session_id)
        self.log.log_event("qa.clear_conversation", session_id=session_id, removed=removed)
        return removed


def _rewritten_query(info: dict[str, Any]) -> str | None:
    """从 ``understand()`` 的结果里取**改写问句**（t21，§24）。

    实测缺陷：``understand()`` 返回的键是 ``rewritten``，而调用方此前读 ``rewritten_query`` → 恒为
    ``None`` → **多轮改写被算出却从未用于检索**（A4：轮 2「那法定代表人呢？」用原问题检索，命中简历/
    中介机构块，答成「法定代表人为程勇波」并引 p255）。本函数同时兼容两种键名，避免再次出现键名漂移。
    """
    if not isinstance(info, dict):
        return None
    value = info.get("rewritten") or info.get("rewritten_query")
    return str(value) if value else None


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="milliseconds")


_engine: QAEngine | None = None
_engine_lock = threading.Lock()


def build_engine(*, cfg: AppConfig | None = None, warmup: bool = True, logger: Any = None) -> QAEngine:
    """构造引擎（``warmup=True`` 时立即预热）。"""
    log = _lazy_logger(logger)
    engine = QAEngine(cfg=cfg, logger=log)
    if warmup:
        engine.warmup()
    return engine


def get_engine() -> QAEngine:
    """进程级单例（UI/HTTP 共用，避免重复加载索引）。"""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = build_engine(warmup=True)
        return _engine


__all__ = ["QAEngine", "EngineHealth", "build_engine", "get_engine", "Answer", "AnswerDelta",
           "RetrievalResult"]
