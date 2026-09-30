"""问答主链路编排。

一次提问的完整流程：

    会话校验 → 用户/角色权限（API 层）→ 记忆加载（Redis + Milvus）
    → 查询改写（可选）→ 三路混合检索（Milvus + BM25，加权 RRF 融合）
    → 角色化提示词 → 本地模型流式生成 → 引用校验 → 角色护栏
    → 回写聊天记录（Redis List/Hash/zSet/Set）→ 抽取事实写长期记忆
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from ..config import Config, get_config
from ..errors import NotFoundError, ValidationError
from ..logging_conf import get_logger
from ..memory.memory import MemoryBundle, get_memory
from ..models.llm import GenerationResult, get_llm
from ..retrieval.retriever import RetrievalResult, RetrievedChunk, get_retriever
from ..roles import Role, RoleRegistry
from ..store.redis_store import get_redis
from .prompts import build_messages, rewrite_prompt

logger = get_logger(__name__)

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass(slots=True)
class Citation:
    """引用条目：编号 → 具体知识块。"""

    index: int
    chunk_id: str
    doc_title: str
    section: str
    source: str
    scope: str
    snippet: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "doc_title": self.doc_title,
            "section": self.section,
            "source": self.source,
            "scope": self.scope,
            "snippet": self.snippet,
            "score": round(self.score, 6),
        }


@dataclass(slots=True)
class PreparedContext:
    """生成前的全部上下文物料。"""

    role: Role
    user_id: str
    session_id: str
    question: str
    rewritten: str
    retrieval: RetrievalResult
    memory: MemoryBundle
    contexts: list[RetrievedChunk]
    messages: list[dict[str, str]]
    timings: dict[str, float] = field(default_factory=dict)
    safety_notice: str = ""


@dataclass(slots=True)
class ChatResult:
    """一次问答的完整结果。"""

    session_id: str
    user_id: str
    role_id: str
    question: str
    answer: str
    citations: list[Citation]
    retrieval: dict[str, Any]
    memory: dict[str, Any]
    usage: dict[str, Any]
    timings: dict[str, float]
    guardrails: dict[str, Any]
    rewritten_query: str = ""
    cited: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "role_id": self.role_id,
            "question": self.question,
            "answer": self.answer,
            "citations": [item.to_dict() for item in self.citations],
            "cited": self.cited,
            "rewritten_query": self.rewritten_query,
            "retrieval": self.retrieval,
            "memory": self.memory,
            "usage": self.usage,
            "timings": {key: round(value, 4) for key, value in self.timings.items()},
            "guardrails": self.guardrails,
            "created_at": self.created_at,
        }


class RagPipeline:
    """角色化 RAG_try 问答编排器。"""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.roles = RoleRegistry.from_config(self.config)
        self.retriever = get_retriever(self.config)
        self.memory = get_memory(self.config)
        self.redis = get_redis(self.config)
        self.llm = get_llm(self.config)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 会话
    def ensure_session(self, user_id: str, role_id: str, session_id: str | None) -> str:
        """校验会话归属；不存在或不属于该用户时新建会话。"""

        if session_id:
            meta = self.redis.session_meta(session_id)
            if meta and str(meta.get("user_id")) == user_id:
                if str(meta.get("role_id")) != role_id:
                    self.redis.update_session(session_id, role_id=role_id)
                self.redis.touch_session(session_id)
                return session_id
            if meta:
                raise NotFoundError("会话不属于当前用户", session_id=session_id)
        return self.redis.create_session(user_id, role_id)

    # ------------------------------------------------------------------ 准备
    def prepare(
        self,
        user_id: str,
        role_id: str,
        question: str,
        session_id: str | None = None,
        top_k: int | None = None,
        mode: str = "hybrid",
        fusion: str = "app",
        use_cache: bool = True,
    ) -> PreparedContext:
        question = (question or "").strip()
        if not question:
            raise ValidationError("问题不能为空")
        if len(question) > 2000:
            raise ValidationError("问题过长（上限 2000 字）")

        role = self.roles.require_enabled(role_id)
        session_id = self.ensure_session(user_id, role_id, session_id)
        timings: dict[str, float] = {}

        mark = time.perf_counter()
        memory = self.memory.load(user_id, role_id, session_id, query=question)
        timings["memory"] = time.perf_counter() - mark

        rewritten = self._maybe_rewrite(question, memory, timings)
        safety_notice = self.roles.safety_notice(role, question)
        if safety_notice:
            logger.warning("问题命中角色红线：role=%s question=%s", role.id, question[:60])
        retrieval = self.retriever.search(
            rewritten or question,
            role_id,
            top_k=top_k,
            mode=mode,
            fusion=fusion,
            use_cache=use_cache,
            debug=True,
        )
        contexts = self._trim_contexts(retrieval.results)
        messages = build_messages(
            role, question, contexts, memory,
            max_recent_turns=int(self.config.get("memory.short_term_turns", 5)),
            safety_notice=safety_notice,
        )
        return PreparedContext(
            role=role,
            user_id=user_id,
            session_id=session_id,
            question=question,
            rewritten=rewritten,
            retrieval=retrieval,
            memory=memory,
            contexts=contexts,
            messages=messages,
            timings=timings,
            safety_notice=safety_notice,
        )

    def _maybe_rewrite(self, question: str, memory: MemoryBundle, timings: dict[str, float]) -> str:
        """查询改写：仅在开启且存在历史时执行，输出异常则回退原问题。"""

        if not bool(self.config.get("rag.rewrite_enabled", False)):
            return ""
        history = memory.recent
        if not history:
            return ""
        lines = [
            f"{'用户' if item.get('role') == 'user' else '助手'}：{str(item.get('content', ''))[:120]}"
            for item in history[-4:]
        ]
        mark = time.perf_counter()
        try:
            result = self.llm.complete(
                rewrite_prompt(lines, question),
                max_new_tokens=int(self.config.get("rag.rewrite_max_tokens", 96)),
                temperature=0.0,
            )
            candidate = result.text.strip().strip('"“”')
            if 2 <= len(candidate) <= 120 and "\n" not in candidate and candidate != question:
                timings["rewrite"] = time.perf_counter() - mark
                logger.info("查询改写：%s → %s", question, candidate)
                return candidate
        except Exception as exc:  # pragma: no cover
            logger.warning("查询改写失败，使用原问题：%s", exc)
        timings["rewrite"] = time.perf_counter() - mark
        return ""

    def _trim_contexts(self, contexts: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        budget = int(self.config.get("models.llm.max_context_chars", 7000))
        kept: list[RetrievedChunk] = []
        used = 0
        for chunk in contexts:
            length = len(chunk.text) + 80
            if kept and used + length > budget:
                break
            kept.append(chunk)
            used += length
        return kept

    # ------------------------------------------------------------------ 收尾
    def finalize(
        self,
        prepared: PreparedContext,
        answer: str,
        usage: dict[str, Any] | None = None,
        generation_seconds: float = 0.0,
    ) -> ChatResult:
        raw_answer = (answer or "").strip()
        if not raw_answer:
            raw_answer = "（模型没有返回内容，请重试或换一个问法。）"

        final_answer, hits = self.roles.apply_guardrails(prepared.role, raw_answer)
        citations, cited = self._extract_citations(raw_answer, prepared.contexts)

        timings = dict(prepared.timings)
        timings["generation"] = generation_seconds
        timings["total"] = sum(
            value for key, value in timings.items() if key not in {"total"}
        ) + float(prepared.retrieval.timings.get("total", 0.0))

        usage = dict(usage or {})
        result = ChatResult(
            session_id=prepared.session_id,
            user_id=prepared.user_id,
            role_id=prepared.role.id,
            question=prepared.question,
            answer=final_answer,
            citations=citations,
            retrieval=prepared.retrieval.to_dict(with_text=False),
            memory=prepared.memory.to_dict(),
            usage=usage,
            timings=timings,
            guardrails={"triggered": bool(hits), "hits": hits,
                        "disclaimer_added": hits and final_answer != raw_answer},
            rewritten_query=prepared.rewritten,
            cited=cited,
        )
        self._persist(result, prepared)
        return result

    def _extract_citations(
        self, answer: str, contexts: Sequence[RetrievedChunk]
    ) -> tuple[list[Citation], bool]:
        """校验引用编号：剔除模型自造的编号，只保留指向真实片段的引用。"""

        indices: list[int] = []
        for found in _CITATION_RE.findall(answer):
            value = int(found)
            if 1 <= value <= len(contexts) and value not in indices:
                indices.append(value)
        citations = [
            Citation(
                index=value,
                chunk_id=contexts[value - 1].chunk_id,
                doc_title=contexts[value - 1].doc_title,
                section=contexts[value - 1].section,
                source=contexts[value - 1].source,
                scope=contexts[value - 1].scope,
                snippet=contexts[value - 1].text[:200],
                score=contexts[value - 1].score,
            )
            for value in indices
        ]
        return citations, bool(citations)

    def _persist(self, result: ChatResult, prepared: PreparedContext) -> None:
        """回写 Redis：聊天记录、会话元信息、引用集合、统计与长期记忆。"""

        try:
            if not prepared.retrieval.cached:
                self.redis.incr_stat("retrieval_miss")
            else:
                self.redis.incr_stat("retrieval_hit")
            self.redis.append_message(
                result.session_id,
                {
                    "role": "user",
                    "content": result.question,
                    "role_id": result.role_id,
                    "rewritten": result.rewritten_query,
                    "route": result.retrieval.get("mode", "hybrid"),
                },
            )
            self.redis.append_message(
                result.session_id,
                {
                    "role": "assistant",
                    "content": result.answer,
                    "role_id": result.role_id,
                    "citations": [item.to_dict() for item in result.citations],
                    "usage": result.usage,
                },
            )
            meta = self.redis.session_meta(result.session_id) or {}
            turns = int(meta.get("turns", 0) or 0) + 1
            updates: dict[str, Any] = {"turns": turns}
            if str(meta.get("title", "")) in {"", "新会话"}:
                updates["title"] = result.question[:24]
            self.redis.update_session(result.session_id, **updates)
            self.redis.add_citations(
                result.session_id, [item.chunk_id for item in result.citations]
            )
            self.redis.mark_online(result.user_id)

            self.redis.incr_stat("chat_requests")
            self.redis.incr_stat("chat_tokens", int(result.usage.get("completion_tokens", 0) or 0))
            self.redis.incr_stat(
                f"role_{result.role_id}", daily=True
            )
            if result.guardrails.get("triggered"):
                self.redis.incr_stat("guardrail_triggered")
            if not result.cited:
                self.redis.incr_stat("answers_without_citation")

            memory_report = self.memory.remember(
                result.user_id, result.role_id, result.session_id, result.question, result.answer
            )
            result.memory["write_back"] = memory_report
        except Exception as exc:  # pragma: no cover - 回写失败不影响回答
            logger.warning("会话回写失败：%s", exc)

    # ------------------------------------------------------------------ 生成
    def answer(
        self,
        user_id: str,
        role_id: str,
        question: str,
        session_id: str | None = None,
        top_k: int | None = None,
        mode: str = "hybrid",
        fusion: str = "app",
        use_cache: bool = True,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> ChatResult:
        """非流式问答（命令行、测试、脚本调用）。"""

        prepared = self.prepare(
            user_id, role_id, question, session_id=session_id, top_k=top_k,
            mode=mode, fusion=fusion, use_cache=use_cache,
        )
        started = time.perf_counter()
        generation: GenerationResult = self.llm.complete(
            prepared.messages,
            temperature=prepared.role.temperature if temperature is None else temperature,
            max_new_tokens=prepared.role.max_new_tokens if max_new_tokens is None else max_new_tokens,
        )
        return self.finalize(prepared, generation.text, generation.to_dict(),
                             time.perf_counter() - started)

    def stream(
        self,
        user_id: str,
        role_id: str,
        question: str,
        session_id: str | None = None,
        top_k: int | None = None,
        mode: str = "hybrid",
        fusion: str = "app",
        use_cache: bool = True,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """流式问答：产出事件流（meta / delta / done / error）。"""

        prepared = self.prepare(
            user_id, role_id, question, session_id=session_id, top_k=top_k,
            mode=mode, fusion=fusion, use_cache=use_cache,
        )
        yield {
            "type": "meta",
            "session_id": prepared.session_id,
            "role_id": prepared.role.id,
            "role_name": prepared.role.name,
            "question": prepared.question,
            "rewritten_query": prepared.rewritten,
            "retrieval": prepared.retrieval.to_dict(with_text=True),
            "memory": prepared.memory.to_dict(),
            "safety_notice": bool(prepared.safety_notice),
            "timings": {key: round(value, 4) for key, value in prepared.timings.items()},
        }

        started = time.perf_counter()
        collected: list[str] = []
        usage: dict[str, Any] = {}

        def _on_done(generation: GenerationResult) -> None:
            usage.update(generation.to_dict())

        try:
            for piece in self.llm.stream(
                prepared.messages,
                on_done=_on_done,
                temperature=prepared.role.temperature if temperature is None else temperature,
                max_new_tokens=prepared.role.max_new_tokens if max_new_tokens is None else max_new_tokens,
            ):
                collected.append(piece)
                yield {"type": "delta", "text": piece}
        except Exception as exc:  # pragma: no cover - 生成异常
            logger.error("生成失败：%s", exc)
            yield {"type": "error", "code": "generation_failed", "message": str(exc)}
            return

        result = self.finalize(prepared, "".join(collected), usage, time.perf_counter() - started)
        yield {
            "type": "done",
            "answer": result.answer,
            "citations": [item.to_dict() for item in result.citations],
            "cited": result.cited,
            "guardrails": result.guardrails,
            "usage": result.usage,
            "timings": {key: round(value, 4) for key, value in result.timings.items()},
            "session_id": result.session_id,
        }


_pipeline: RagPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline(config: Config | None = None) -> RagPipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = RagPipeline(config)
        return _pipeline
