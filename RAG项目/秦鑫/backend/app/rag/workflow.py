import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

from ..config import Settings
from ..models import ModelGateway
from ..storage.redis import RedisStore
from ..storage.vector import MilvusStore
from ..system import set_user_context
from ..workspace import WorkspaceService
from ..memory.api import MemoryOrchestrator
from .context import ContextBuilder
from .answer import AnswerGenerator
from .understand import QueryUnderstanding
from .plan import SearchRouter
from .search import EvidenceBuilder, ProcessingPipeline, Reranker
from .search import Retriever


logger = logging.getLogger("law_rag.rag")

FINAL_ANSWER_OPEN_TAG = "<final_answer>"
FINAL_ANSWER_CLOSE_TAG = "</final_answer>"
UNTAGGED_STREAM_MIN_CHARS = 24
NOISY_ANSWER_PREFIXES = ("对，", "对。", "不对，", "不对。", "等下，", "等下。", "然后", "首先得", "用户现在")
RESET_CONTEXT_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "用户明确表示本轮问题与之前或上次内容无关。"
        "本轮必须按新的独立问题处理，不得引用之前对话、旧材料、旧结论或旧上传文件；"
        "如果当前事实不足，应说明缺少本轮事实；只有用户明确要求查看附件或上传材料时，才说明当前会话没有可用材料。"
    ),
}


class RagWorkflow:
    """法律问答主流程：理解问题、规划检索、召回证据、重排融合并生成答案。"""

    def __init__(self, settings: Settings, model: ModelGateway, milvus: MilvusStore, redis: RedisStore, workspace: WorkspaceService, memory: MemoryOrchestrator | None = None, user_state=None):
        self.settings = settings
        self.model = model
        self.redis = redis
        self.memory = memory
        self.user_state = user_state
        self.understanding = QueryUnderstanding(model)
        self.router = SearchRouter(settings)
        self.retriever = Retriever(settings, model, milvus, workspace)
        self.ranker = Reranker(model, input_limit=settings.retrieval_rerank_input_max, doc_max_chars=settings.reranker_doc_max_chars)
        self.evidence = EvidenceBuilder(
            limit=settings.retrieval_evidence_limit,
            max_content_chars=settings.evidence_max_chars,
            exact_limit=settings.retrieval_evidence_limit,
        )
        self.processing = ProcessingPipeline(
            self.ranker,
            self.evidence,
            rrf_limit=settings.retrieval_candidate_pool_max,
            rrf_k=settings.retrieval_rrf_k,
        )
        self.context_builder = ContextBuilder(
            max_tokens=settings.context_max_tokens,
            budget={"evidence": settings.context_evidence_token_budget},
            evidence_max_chars=settings.context_evidence_max_chars,
            evidence_limit=settings.retrieval_evidence_limit,
        )
        self.generator = AnswerGenerator(model)

    @staticmethod
    def user_facing_answer(text: str) -> str:
        """从模型可能返回的草稿/标签内容中截取真正展示给用户的回答。"""
        value = str(text or "").strip()
        if FINAL_ANSWER_OPEN_TAG in value:
            value = value.split(FINAL_ANSWER_OPEN_TAG, 1)[1]
        if FINAL_ANSWER_CLOSE_TAG in value:
            value = value.split(FINAL_ANSWER_CLOSE_TAG, 1)[0]
        return RagWorkflow.sanitize_answer(value)

    @staticmethod
    def sanitize_answer(text: str) -> str:
        value = str(text or "").strip()
        conclusion_prefixes = ("先说结论：", "先说结论:", "结论：", "结论:")
        for prefix in conclusion_prefixes:
            if value.startswith(prefix):
                return value[len(prefix):].lstrip()
        markers = ["一、", "一.", "根据"]
        if any(value.startswith(prefix) for prefix in markers):
            return value
        if value.startswith("关于"):
            return value
        starts = [index for marker in [*conclusion_prefixes, *markers] if (index := value.find(marker)) > 0]
        if starts:
            return RagWorkflow.sanitize_answer(value[min(starts):].strip())
        if value.startswith(NOISY_ANSWER_PREFIXES):
            sentences = value.replace("\r\n", "\n").split("\n")
            clean_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
            return "\n".join(clean_sentences[-3:]).strip() if len(clean_sentences) > 3 else value
        return value

    @staticmethod
    def should_stream_untagged_answer(buffer: str) -> bool:
        value = str(buffer or "").strip()
        if len(value) < UNTAGGED_STREAM_MIN_CHARS:
            return False
        if value.startswith(NOISY_ANSWER_PREFIXES):
            return False
        draft_markers = ("思考", "草稿", "内部", "先分析", "自我纠错")
        return not any(marker in value[:80] for marker in draft_markers)

    @staticmethod
    def user_facing_answer_stream(deltas):
        buffer = ""
        inside_final_answer = False
        final_answer_closed = False
        streaming_untagged = False
        for delta in deltas:
            if not delta:
                continue
            if final_answer_closed:
                continue
            if streaming_untagged:
                yield delta
                continue
            buffer += str(delta)
            if not inside_final_answer:
                open_index = buffer.find(FINAL_ANSWER_OPEN_TAG)
                if open_index >= 0:
                    inside_final_answer = True
                    buffer = buffer[open_index + len(FINAL_ANSWER_OPEN_TAG):]
                elif RagWorkflow.should_stream_untagged_answer(buffer):
                    streaming_untagged = True
                    piece = RagWorkflow.sanitize_answer(buffer)
                    buffer = ""
                    if piece:
                        yield piece
                    continue
                else:
                    continue
            close_index = buffer.find(FINAL_ANSWER_CLOSE_TAG)
            if close_index >= 0:
                piece = buffer[:close_index]
                buffer = ""
                final_answer_closed = True
            else:
                keep = len(FINAL_ANSWER_CLOSE_TAG) - 1
                if len(buffer) <= keep:
                    continue
                piece = buffer[:-keep]
                buffer = buffer[-keep:]
            if piece:
                yield piece
        if streaming_untagged:
            return
        if inside_final_answer and not final_answer_closed and buffer:
            yield buffer
        elif not inside_final_answer and buffer:
            piece = RagWorkflow.user_facing_answer(buffer)
            if piece:
                yield piece

    @staticmethod
    def source_snapshot(rows: list[dict]) -> list[dict]:
        return [
            {
                "chunk_id": row.get("source_id", ""),
                "source_type": row.get("source_type", "unknown"),
                "collection": row.get("collection", ""),
                "retrieval_channel": row.get("retrieval_channel", ""),
                "title": row.get("title", ""),
                "score": row.get("score"),
                "rerank_score": row.get("rerank_score"),
            }
            for row in rows
        ]

    @staticmethod
    def thinking_event(summary: str, step: str, status: str = "running", mode: str = "append", sources: list[dict] | None = None) -> tuple[str, dict]:
        # 新增 smooth 字段，便于前端做平滑动画
        payload = {
            "status": status,
            "summary": summary,
            "mode": mode,
            "step": step,
            "steps": [step] if step else [],
            "smooth": True,
        }
        if sources is not None:
            payload["sources"] = sources
        return "thinking", payload

    @staticmethod
    def generation_thinking_step(answer_length: int) -> str:
        if answer_length < 80:
            return "正在把检索到的事实和规则组织成正式结论。"
        if answer_length < 220:
            return "正在核对回答中的法律依据是否能对应到具体事实。"
        if answer_length < 420:
            return "正在补充证据缺口、对方可能抗辩和可操作路径。"
        return "正在收束最终答案，避免遗漏风险提示和下一步建议。"

    def warn_low_confidence(self, rows: list[dict]) -> int:
        threshold = getattr(self.settings, "log_low_confidence_score", 0.7)
        low_rows = [
            row for row in rows
            if float(row.get("rerank_score", row.get("score", 1)) or 0) < threshold
        ]
        if low_rows:
            logger.warning(
                "检索结果置信度低，建议人工复核",
                extra={
                    "event": "low_confidence_retrieval",
                    "fields": {
                        "threshold": threshold,
                        "low_confidence_sources": self.source_snapshot(low_rows),
                    },
                },
            )
        return len(low_rows)

    @staticmethod
    def _search_task(label: str, func, *args):
        """包装一个检索通道，统一记录耗时和错误。"""
        start = perf_counter()
        try:
            rows = func(*args)
            error = ""
        except Exception as exc:
            rows = []
            error = str(exc)
        return label, rows, round((perf_counter() - start) * 1000, 2), error

    @staticmethod
    def channel_counts(rows: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            channel = str(row.get("retrieval_channel") or row.get("source_type") or "unknown")
            counts[channel] = counts.get(channel, 0) + 1
        return counts

    def _thinking_enabled(self, thinking_enabled: bool | None = None) -> bool:
        if thinking_enabled is not None:
            return bool(thinking_enabled)
        return bool(getattr(self.settings, "deepseek_thinking", True))

    @staticmethod
    def _stream_answer_text(generator, question: str, evidence: list[dict], history: list[dict], thinking_enabled: bool):
        try:
            yield from generator.stream_answer_text(question, evidence, history, thinking_enabled=thinking_enabled)
        except TypeError:
            yield from generator.stream_answer_text(question, evidence, history)

    @staticmethod
    def _generate_answer(generator, question: str, evidence: list[dict], history: list[dict], thinking_enabled: bool) -> dict:
        try:
            return generator.generate(question, evidence, history, thinking_enabled=thinking_enabled)
        except TypeError:
            return generator.generate(question, evidence, history)

    @staticmethod
    def _understand_query(understanding, question: str, memory_context):
        try:
            return understanding.understand(question, memory_context)
        except TypeError:
            return understanding.understand(question)

    def _run_stream(self, question: str, user: dict | None = None, session_id: str | None = None, include_web: bool = True, thinking_enabled: bool | None = None):
        start = perf_counter()
        include_web = bool(include_web)
        user_id = user.get("user_id") if user else "anonymous"
        if not user:
            session_id = None
        set_user_context(user_id, session_id)
        reset_previous_context = bool(
            user
            and session_id
            and getattr(getattr(self.retriever, "workspace", None), "query_resets_previous_context", lambda _query: False)(question)
        )
        thinking_enabled = self._thinking_enabled(thinking_enabled)
        if thinking_enabled:
            yield self.thinking_event("正在启动深度思考", "正在接收问题并准备拆解事实、诉求和可用证据。")
        memory = getattr(self, "memory", None)
        if reset_previous_context and session_id:
            try:
                if memory:
                    memory.delete_session(user_id, session_id)
                else:
                    self.redis.delete_history(user_id, session_id)
            except Exception:
                logger.warning(
                    "清理隔离会话记忆失败，继续按无历史上下文处理本轮问题",
                    extra={"event": "rag_reset_context_cleanup_failed", "fields": {"session_id": session_id or ""}},
                    exc_info=True,
                )
        user_profile = dict((user or {}).get("profile") or {})
        user_state = getattr(self, "user_state", None)
        if user_state and user and not user_profile:
            try:
                user_profile = user_state.get_profile(user_id)
            except Exception:
                user_profile = {}
        memory_context = None if reset_previous_context else (memory.load_context(user_id, session_id, user_profile=user_profile, query=question) if memory and session_id else None)
        effective_question = getattr(memory_context, "resolved_query", "") or question
        history = [RESET_CONTEXT_SYSTEM_MESSAGE] if reset_previous_context else (memory_context.short_term if memory_context else (self.redis.get_history(user_id, session_id) if session_id else []))
        stage_timings: dict[str, float] = {}

        yield "status", {"message": "正在理解问题", "stage": "understanding"}
        stage_start = perf_counter()
        info = self._understand_query(self.understanding, effective_question, memory_context)
        stage_timings["understanding"] = round((perf_counter() - stage_start) * 1000, 2)
        yield "step", {
            "stage": "understanding",
            "status": "completed",
            "message": "已识别问题结构",
            "detail": f"关键词 {len(info.get('keywords', []))} 个 / 意图 {info.get('intent', 'general')}",
            "elapsed_ms": stage_timings["understanding"],
            "route": info.get("route", "full_retrieval"),
            "keywords": info.get("keywords", []),
            "article_numbers": info.get("article_numbers", []),
            "legal_domains": info.get("legal_domains", []),
            "intent": info.get("intent", "general"),
            "confidence": info.get("confidence", 0.0),
            "intent_reason": info.get("intent_reason", ""),
            "understanding_source": info.get("understanding_source", ""),
            "understanding_error": info.get("understanding_error", ""),
            "route_memory_used": info.get("route_memory_used", False),
            "query_variants": info.get("query_variants", []),
            "original_query": info.get("original_query", question),
            "rewritten_query": info.get("rewritten_query", question),
            "resolved_references": {**getattr(memory_context, "resolved_references", {}), **info.get("resolved_references", {})} if memory_context else info.get("resolved_references", {}),
            "reset_previous_context": reset_previous_context,
        }
        if thinking_enabled:
            domains = "、".join(info.get("legal_domains", [])[:3]) or "相关法律关系"
            yield self.thinking_event(
                "已完成问题拆解，正在规划检索",
                f"已识别 {len(info.get('keywords', []))} 个关键词，初步聚焦 {domains} 和用户的具体诉求。",
            )

        retrieval_start = perf_counter()
        yield "status", {"message": "正在规划检索", "stage": "retrieval"}
        stage_start = perf_counter()
        plan = self.router.route(info, include_web)
        stage_timings["planning"] = round((perf_counter() - stage_start) * 1000, 2)
        logger.info(
            "RAG 检索计划生成",
            extra={
                "event": "rag_plan_created",
                "fields": {
                    "include_web": include_web,
                    "question_length": len(question),
                    "rewritten_query": plan.rewritten_query,
                    "session_id": session_id or "",
                    "searched_collections": plan.collections,
                    "priority_collections": plan.priority_collections,
                    "route": plan.route,
                    "route_reason": plan.route_reason,
                    "query_variants": plan.query_variants,
                    "article_numbers": plan.article_numbers,
                    "legal_domains": plan.legal_domains,
                    "intent": plan.intent,
                    "route_memory_used": info.get("route_memory_used", False),
                    "retrieval_notes": plan.retrieval_notes,
                },
            },
        )

        search_targets = ["公共知识库"]
        if user and plan.search_private:
            search_targets.insert(0, "私有材料")
        if plan.include_web:
            search_targets.append("网页")
        if thinking_enabled:
            yield self.thinking_event(
                f"正在并行检索{'、'.join(search_targets)}",
                f"检索计划已确定：优先从 {'、'.join(search_targets)} 中寻找能对应具体事实的材料。",
            )
        yield "status", {"message": f"正在并行检索{'、'.join(search_targets)}", "stage": "retrieval"}
        search_results = {"private": [], "public": [], "web": []}
        search_errors: dict[str, str] = {}
        stage_timings["private_search"] = 0.0
        stage_timings["public_search"] = 0.0
        stage_timings["web_search"] = 0.0
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            if user and plan.search_private:
                futures[executor.submit(self._search_task, "private", self.retriever.search_user_materials, plan, user, session_id)] = "private"
            if plan.search_public:
                futures[executor.submit(self._search_task, "public", self.retriever.search_public_knowledge, plan)] = "public"
            if plan.include_web:
                futures[executor.submit(self._search_task, "web", self.retriever.search_web_pages, plan)] = "web"
            for future in as_completed(futures):
                label = futures[future]
                task_label, rows, elapsed_ms, error = future.result()
                search_results[label] = rows
                stage_timings[f"{task_label}_search"] = elapsed_ms
                if error:
                    search_errors[label] = error
                    logger.warning(
                        "RAG %s 检索失败",
                        label,
                        extra={
                            "event": "retrieval_failed",
                            "fields": {
                                "source": label,
                                "error": error,
                                "include_web": include_web,
                                "session_id": session_id or "",
                            },
                        },
                    )
                if thinking_enabled:
                    channel_names = {"private": "私有材料", "public": "公共法律库", "web": "网页资料"}
                    channel_name = channel_names.get(label, label)
                    if error:
                        yield self.thinking_event("检索仍在继续，部分来源暂不可用", f"{channel_name} 检索暂未成功，正在保留其他可用来源继续分析。", status="completed")
                        yield self.thinking_event("检索仍在继续，部分来源暂可恢复", f"{channel_name} 正在切回检索准备状态，继续补齐其他来源。", status="running")
                    else:
                        yield self.thinking_event(
                            "正在汇总检索结果",
                            f"{channel_name} 已返回 {len(rows)} 条候选材料，用时 {(elapsed_ms / 1000):.1f} 秒。",
                        )

        raw = search_results["private"] + search_results["public"] + search_results["web"]
        retrieval_channel_counts = self.channel_counts(raw)
        channels = [
            rows for rows in (
                search_results["private"],
                search_results["public"],
                search_results["web"],
            )
            if rows
        ]
        stage_timings["retrieval"] = round((perf_counter() - retrieval_start) * 1000, 2)
        logger.info(
            "RAG 原始检索完成",
            extra={
                "event": "retrieval_completed",
                "fields": {
                    "retrieval_sources": self.source_snapshot(raw),
                    "raw_result_count": len(raw),
                    "retrieval_counts": {
                        "private": len(search_results["private"]),
                        "public": len(search_results["public"]),
                        "web": len(search_results["web"]),
                    },
                    "retrieval_channel_counts": retrieval_channel_counts,
                    "search_errors": search_errors,
                },
            },
        )
        yield "step", {
            "stage": "retrieval",
            "status": "completed",
            "message": f"检索到 {len(raw)} 条候选",
            "detail": f"私有 {len(search_results['private'])} 条 / 公共 {len(search_results['public'])} 条 / 网页 {len(search_results['web'])} 条",
            "elapsed_ms": stage_timings["retrieval"],
            "count": len(raw),
            "searched_collections": plan.collections,
            "priority_collections": plan.priority_collections,
            "route": plan.route,
            "route_reason": plan.route_reason,
            "query_variants": plan.query_variants,
            "retrieval_notes": plan.retrieval_notes,
        }
        if thinking_enabled:
            yield self.thinking_event(
                "检索完成，正在筛选核心证据",
                f"共找到 {len(raw)} 条候选材料，接下来会融合排序并保留最能支撑判断的证据。",
            )

        yield "status", {"message": "正在融合与重排", "stage": "ranking"}
        stage_start = perf_counter()
        processing = getattr(self, "processing", ProcessingPipeline(self.ranker, self.evidence))
        fusion_start = perf_counter()
        fused = processing.merge_results(channels or [raw])
        stage_timings["fusion"] = round((perf_counter() - fusion_start) * 1000, 2)
        rerank_start = perf_counter()
        ranked = processing.rank(plan.query, fused)
        stage_timings["rerank"] = round((perf_counter() - rerank_start) * 1000, 2)
        evidence_start = perf_counter()
        low_confidence = self.warn_low_confidence(ranked)
        evidence = processing.build_evidence(ranked)
        stage_timings["evidence"] = round((perf_counter() - evidence_start) * 1000, 2)
        stage_timings["ranking"] = round((perf_counter() - stage_start) * 1000, 2)
        yield "step", {
            "stage": "ranking",
            "status": "completed",
            "message": f"保留 {len(evidence)} 条证据",
            "detail": f"融合 {len(fused)} 条 / 证据 {len(evidence)} 条",
            "elapsed_ms": stage_timings["ranking"],
            "count": len(evidence),
        }
        if thinking_enabled:
            yield self.thinking_event(
                "已筛出核心证据，准备生成答案",
                f"已从 {len(fused)} 条融合候选中保留 {len(evidence)} 条核心证据，正在对应事实、规则和风险点。",
            )

        yield "status", {"message": "正在生成回答", "stage": "generation"}
        context_builder = getattr(self, "context_builder", ContextBuilder())
        context_payload = context_builder.build_context(plan.query, memory_context, evidence, user_profile=user_profile)
        answer_history = context_payload["history"]
        metadata_builder = getattr(self.generator, "answer_metadata", None)
        pre_answer_metadata = metadata_builder(context_payload["evidence"]) if callable(metadata_builder) else {}
        if thinking_enabled:
            for step in [
                "正在核对用户问题中的法律关系、具体诉求和会影响结果的关键事实。",
                f"正在结合 {len(evidence)} 条核心证据区分事实材料、法律规则和补充参考。",
                "正在把检索到的法条、司法解释或类案对应到具体事实，避免只罗列来源名称。",
                "正在同步检查证据缺口、对方可能抗辩和下一步可操作路径。",
            ]:
                yield self.thinking_event(f"正在结合问题和 {len(evidence)} 条核心证据做案情分析", step)
            thinking_sources = pre_answer_metadata.get("citations", [])
            yield self.thinking_event(
                "思考完成，开始输出正式答案",
                "已完成事实、证据和法律依据核对，下面开始输出正式答案。",
                status="completed",
                mode="replace",
                sources=thinking_sources,
            )
        stage_start = perf_counter()
        answer_text_parts: list[str] = []
        if evidence:
            deltas = self._stream_answer_text(self.generator, plan.query, context_payload["evidence"], answer_history, thinking_enabled)
            for piece in self.user_facing_answer_stream(deltas):
                for character in piece:
                    if not answer_text_parts and not character.strip():
                        continue
                    answer_text_parts.append(character)
                    yield "chunk", {"delta": character}
            answer_text = self.user_facing_answer("".join(answer_text_parts))
            if not answer_text:
                answer = self._generate_answer(self.generator, plan.query, context_payload["evidence"], answer_history, thinking_enabled)
                answer_text = self.user_facing_answer(str(answer.get("answer", "")))
                if low_confidence:
                    answer["risk_notice"] = (answer.get("risk_notice") or "") + " 本次检索依据置信度较低，结论仅供参考，建议补充更直接的证据后再次确认。"
                for character in answer_text:
                    answer_text_parts.append(character)
                    yield "chunk", {"delta": character}
            else:
                highlight_enricher = getattr(self.generator, "ensure_source_highlights", None)
                enriched_answer = (
                    highlight_enricher(answer_text, plan.query, context_payload["evidence"])
                    if callable(highlight_enricher)
                    else answer_text
                )
                if enriched_answer != answer_text:
                    extra = enriched_answer[len(answer_text):] if enriched_answer.startswith(answer_text) else f"\n\n{enriched_answer}"
                    for character in extra:
                        yield "chunk", {"delta": character}
                    answer_text = enriched_answer
                answer = {"answer": answer_text, **self.generator.answer_metadata(context_payload["evidence"])}
                if low_confidence:
                    answer["risk_notice"] = (answer.get("risk_notice") or "") + " 本次检索依据置信度较低，结论仅供参考，建议补充更直接的证据后再次确认。"
        else:
            answer = self._generate_answer(self.generator, plan.query, context_payload["evidence"], answer_history, thinking_enabled)
            answer_text = self.user_facing_answer(str(answer.get("answer", "")))
            answer["answer"] = answer_text
            for character in answer_text:
                answer_text_parts.append(character)
                yield "chunk", {"delta": character}
        case_analysis = answer.get("case_analysis") or []
        if not case_analysis and hasattr(self.generator, "build_case_analysis"):
            case_analysis = self.generator.build_case_analysis(plan.query, context_payload["evidence"], answer_text, thinking_enabled=False)
            answer["case_analysis"] = case_analysis
        stage_timings["generation"] = round((perf_counter() - stage_start) * 1000, 2)
        yield "step", {
            "stage": "generation",
            "status": "completed",
            "message": "回答已生成",
            "detail": "输出结论、依据和建议",
            "elapsed_ms": stage_timings["generation"],
            "answer_length": len(str(answer.get("answer", ""))),
        }

        if session_id:
            if memory:
                memory.save_turn(
                    user_id,
                    session_id,
                    question,
                    answer["answer"],
                    answer_payload=answer,
                    source_count=len(evidence),
                )
            else:
                self.redis.append_history(user_id, session_id, {"role": "user", "content": question}, self.settings.history_ttl)
                self.redis.append_history(user_id, session_id, {"role": "assistant", "content": answer["answer"]}, self.settings.history_ttl)

        retrieved_collections = sorted({
            str(row.get("collection", ""))
            for row in raw
            if row.get("source_type") == "public" and row.get("collection")
        })
        citation_source_ids = [citation.get("source_id", "") for citation in answer.get("citations", [])]
        elapsed_ms = round((perf_counter() - start) * 1000, 2)
        logger.info(
            "RAG 回答生成完成",
            extra={
                "event": "rag_answer_generated",
                "fields": {
                    "include_web": include_web,
                    "question_length": len(question),
                    "original_query": plan.original_query,
                    "rewritten_query": plan.rewritten_query,
                    "resolved_references": {**getattr(memory_context, "resolved_references", {}), **info.get("resolved_references", {})} if memory_context else info.get("resolved_references", {}),
                    "history_before": len(history),
                    "searched_collections": plan.collections,
                    "priority_collections": plan.priority_collections,
                    "route": plan.route,
                    "route_reason": plan.route_reason,
                    "query_variants": plan.query_variants,
                    "article_numbers": plan.article_numbers,
                    "legal_domains": plan.legal_domains,
                    "intent": plan.intent,
                    "route_memory_used": info.get("route_memory_used", False),
                    "retrieval_notes": plan.retrieval_notes,
                    "retrieved_collections": retrieved_collections,
                    "raw_result_count": len(raw),
                    "evidence_count": len(evidence),
                    "evidence_sources": self.source_snapshot(evidence),
                    "citation_source_ids": citation_source_ids,
                    "citation_to_retrieval_matched": set(citation_source_ids).issubset({row.get("source_id", "") for row in raw}),
                    "elapsed_ms": elapsed_ms,
                    "stage_timings": stage_timings,
                    "context": context_payload["meta"],
                    "retrieval_counts": {
                        "private": len(search_results["private"]),
                        "public": len(search_results["public"]),
                        "web": len(search_results["web"]),
                        "total": len(raw),
                    },
                    "retrieval_channel_counts": retrieval_channel_counts,
                    "retrieval_errors": search_errors,
                },
            },
        )
        result = {
            "answer": answer,
            "sources": evidence,
            "meta": {
                "elapsed_ms": elapsed_ms,
                "history_before": len(history),
                "original_query": plan.original_query,
                "rewritten_query": plan.rewritten_query,
                "resolved_references": {**getattr(memory_context, "resolved_references", {}), **info.get("resolved_references", {})} if memory_context else info.get("resolved_references", {}),
                "searched_collections": plan.collections,
                "priority_collections": plan.priority_collections,
                "route": plan.route,
                "route_reason": plan.route_reason,
                "query_variants": plan.query_variants,
                "article_numbers": plan.article_numbers,
                "legal_domains": plan.legal_domains,
                "intent": plan.intent,
                "confidence": info.get("confidence", 0.0),
                "intent_reason": info.get("intent_reason", ""),
                "understanding_source": info.get("understanding_source", ""),
                "understanding_error": info.get("understanding_error", ""),
                "route_memory_used": info.get("route_memory_used", False),
                "retrieval_notes": plan.retrieval_notes,
                "retrieved_collections": retrieved_collections,
                "raw_result_count": len(raw),
                "evidence_count": len(evidence),
                "stage_timings": stage_timings,
                "context": context_payload["meta"],
                "retrieval_counts": {
                    "private": len(search_results["private"]),
                    "public": len(search_results["public"]),
                    "web": len(search_results["web"]),
                    "total": len(raw),
                },
                "retrieval_channel_counts": retrieval_channel_counts,
                "retrieval_errors": search_errors,
            },
        }
        result["sources"] = [{k: v for k, v in row.items() if k != "source_id"} for row in result.get("sources", [])]
        yield "complete", result

    def run(self, question: str, user: dict | None = None, session_id: str | None = None, include_web: bool = True, thinking_enabled: bool | None = None) -> dict:
        result = None
        for event, data in self._run_stream(question, user, session_id, include_web, thinking_enabled):
            if event == "complete":
                result = data
        if result is None:
            raise RuntimeError("RAG workflow did not complete")
        return result

    def stream(self, *args, **kwargs):
        yield from self._run_stream(*args, **kwargs)
