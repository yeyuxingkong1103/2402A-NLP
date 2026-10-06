# -*- coding: utf-8 -*-
"""在线 RAG 问答编排器：多轮对话 / 多角色 / 多用户 + Redis 短期记忆。

一次提问的完整流程：

    会话校验（归属）→ 取最近 N 轮上下文（Redis 短期记忆）
    → HybridRetriever 混合检索（Milvus + BM25 + 可选 bge 重排）
    → 角色化提示词（角色 system + 历史 + 知识片段 + 问题）
    → LLM 生成（流式 / 非流式）→ 回写 Redis（会话消息 + 元信息 + TTL）

    from rag2 import RAGChat

    chat = RAGChat()
    for event in chat.stream_answer("小麦常见病虫害有哪些？", user_id="alice", role_id="agriculture_expert"):
        print(event["type"], event.get("text") or event.get("answer"))
"""

from __future__ import annotations

import time
from typing import Any, Iterator, Sequence

from .config import RAG2Config, load_config
from .hybrid_retriever import Hit, HybridRetriever
from .llm_client import LLMClient
from .logging_config import get_logger
from .redis_memory import RedisMemory
from .roles import Role, get_role, list_roles, load_roles

logger = get_logger("online_chat")

CITATION_RULES = (
    "# 回答规则\n"
    "1. 只依据「知识片段」作答；片段里没有的内容，直接回答「知识库中没有相关内容」，不要编造。\n"
    "2. 每个关键结论后面标注来源编号，格式为 [1]、[2]；编号必须来自知识片段。\n"
    "3. 不要编造编号、文献、条文、数据或链接。\n"
    "4. 用简体中文作答；先给结论，再给依据与适用条件；条目化、可执行。\n"
    "5. 与角色身份无关或超出知识范围的问题，礼貌说明并给出可执行的下一步。"
)


class RAGChat:
    """角色化 RAG 问答编排器（懒构建各依赖组件）。"""

    def __init__(
        self,
        config: RAG2Config | None = None,
        roles: Sequence[Role] | None = None,
        retriever: HybridRetriever | None = None,
        llm: LLMClient | None = None,
        memory: RedisMemory | None = None,
        embed_fn: Any = None,
    ) -> None:
        self.config = config or load_config()
        self._roles = list(roles) if roles is not None else load_roles(self.config.roles_file())
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._embed_fn = embed_fn

    # ------------------------------------------------------------------ 依赖
    def get_memory(self) -> RedisMemory:
        if self._memory is None:
            rc = self.config.redis
            self._memory = RedisMemory(
                host=rc.host, port=rc.port, db=rc.db, password=rc.password,
                prefix=rc.prefix, history_ttl=rc.history_ttl, token_ttl=rc.token_ttl,
            )
        return self._memory

    def get_llm(self) -> LLMClient:
        if self._llm is None:
            lc = self.config.llm
            self._llm = LLMClient(
                base_url=lc.base_url, api_key=lc.api_key, model=lc.model,
                temperature=lc.temperature, max_tokens=lc.max_tokens,
                reasoning_effort=lc.reasoning_effort, timeout=lc.timeout,
            )
        return self._llm

    def get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            mc = self.config.milvus
            self._retriever = HybridRetriever(
                uri=mc.uri,
                collection=mc.collection,
                dim=mc.dim,
                embed_fn=self._embed_fn,
                embed_model=mc.embed_model,
                device=mc.device,
                rerank_model=mc.rerank_model,
            )
            logger.info("初始化混合检索器：%s / %s", mc.uri, mc.collection)
            try:
                # 从已存在的集合重建 BM25 索引（hybrid 模式的关键词路）
                self._retriever.rebuild_from_milvus()
            except Exception as exc:  # noqa: BLE001 - BM25 构建失败不影响稠密路
                logger.warning("BM25 索引构建失败（仅剩稠密路）：%s", exc)
        return self._retriever

    # ------------------------------------------------------------------ 角色
    def get_role(self, role_id: str) -> Role:
        return get_role(role_id, self._roles)

    def roles(self) -> list[Role]:
        return list_roles(self._roles)

    def set_roles(self, roles: Sequence[Role]) -> None:
        """热更新角色表（管理员改角色后调用）。"""
        self._roles = list(roles)

    def apply_config(self, config: RAG2Config | None = None) -> None:
        """热应用配置：刷新 self.config，并重置受影响的组件缓存。

        检索参数（top_k/mode/rerank）与 short_term_turns 在每次调用时读取，
        故无需处理；这里只重建 LLM 客户端、刷新 Redis 短期记忆 TTL。
        """
        if config is not None:
            self.config = config
        self._llm = None  # LLM 地址/模型/温度等变化 → 下次 get_llm() 重建
        if self._memory is not None:
            self._memory.history_ttl = self.config.redis.history_ttl
            self._memory.token_ttl = self.config.redis.token_ttl

    # ------------------------------------------------------------------ 会话
    def ensure_session(self, user_id: str, role_id: str, session_id: str | None) -> str:
        """校验会话归属；不存在或不属于该用户时新建会话。"""
        memory = self.get_memory()
        if session_id:
            meta = memory.session_meta(session_id)
            if meta and str(meta.get("user_id")) == user_id:
                if str(meta.get("role_id")) != role_id:
                    memory.update_session(session_id, role_id=role_id)
                memory.touch_session(session_id)
                return session_id
            if meta:
                raise PermissionError("会话不属于当前用户")
        return memory.create_session(user_id, role_id)

    # ------------------------------------------------------------------ 检索
    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        mode: str | None = None,
        rerank: bool | None = None,
    ) -> list[Hit]:
        """混合检索（可被 /api/evaluate 等复用）。"""
        rc = self.config.retrieval
        kwargs: dict[str, Any] = {
            "top_k": top_k or rc.top_k,
            "mode": mode or rc.mode,
        }
        use_rerank = rc.rerank if rerank is None else rerank
        if use_rerank:
            kwargs["rerank"] = True
            kwargs["rerank_top_k"] = rc.rerank_top_k
        return self.get_retriever().search(question, **kwargs)

    # ------------------------------------------------------------------ 提示词
    @staticmethod
    def _format_contexts(hits: Sequence[Hit]) -> str:
        if not hits:
            return "（本轮没有检索到相关知识片段）"
        blocks: list[str] = []
        for index, hit in enumerate(hits, start=1):
            source = hit.source or hit.doc_id or "未命名文档"
            text = (hit.text or "").strip()
            header = f"[{index}] 《{source}》"
            if hit.page:
                header += f"（第 {hit.page} 页）"
            blocks.append(f"{header}\n{text}")
        return "\n\n".join(blocks)

    def _build_messages(
        self,
        role: Role,
        question: str,
        hits: Sequence[Hit],
        history: Sequence[dict[str, Any]],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": f"{role.system_prompt}\n\n{CITATION_RULES}"}
        ]
        recent = list(history)[-(self.config.memory.short_term_turns * 2):]
        for item in recent:
            who = "user" if item.get("role") == "user" else "assistant"
            content = str(item.get("content", "")).strip()
            if content:
                messages.append({"role": who, "content": content[:1500]})
        user_content = (
            "# 知识片段\n"
            f"{self._format_contexts(hits)}\n\n"
            "# 用户问题\n"
            f"{question.strip()}\n\n"
            "请依据上面的知识片段回答，并在句末标注来源编号。"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    # ------------------------------------------------------------------ 引用
    @staticmethod
    def _citations(hits: Sequence[Hit]) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "chunk_id": hit.chunk_id,
                "doc_id": hit.doc_id,
                "source": hit.source,
                "page": hit.page,
                "score": round(hit.score, 6),
                "snippet": (hit.text or "")[:200],
            }
            for index, hit in enumerate(hits, start=1)
        ]

    @staticmethod
    def _context_texts(hits: Sequence[Hit]) -> list[str]:
        return [hit.text for hit in hits]

    # ------------------------------------------------------------------ 回写
    def _persist(
        self,
        user_id: str,
        role_id: str,
        session_id: str,
        question: str,
        answer: str,
        hits: Sequence[Hit],
    ) -> None:
        try:
            memory = self.get_memory()
            memory.append_message(session_id, {"role": "user", "content": question, "role_id": role_id})
            memory.append_message(
                session_id,
                {
                    "role": "assistant",
                    "content": answer,
                    "role_id": role_id,
                    "citations": self._citations(hits),
                },
            )
            meta = memory.session_meta(session_id) or {}
            turns = int(meta.get("turns", 0) or 0) + 1
            updates: dict[str, Any] = {"turns": turns}
            if str(meta.get("title", "")) in {"", "新会话"}:
                updates["title"] = question[:24]
            memory.update_session(session_id, **updates)
        except Exception as exc:  # noqa: BLE001 - 回写失败不影响回答
            logger.warning("会话回写失败：%s", exc)

    # ------------------------------------------------------------------ 问答
    def answer(
        self,
        question: str,
        user_id: str,
        role_id: str,
        session_id: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
        rerank: bool | None = None,
    ) -> dict[str, Any]:
        """非流式问答，返回完整结果字典。"""
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        role = self.get_role(role_id)
        session_id = self.ensure_session(user_id, role_id, session_id)
        memory = self.get_memory()
        history = memory.recent_context(session_id, self.config.memory.short_term_turns)
        hits = self.retrieve(question, top_k=top_k, mode=mode, rerank=rerank)
        messages = self._build_messages(role, question, hits, history)
        answer_text = self.get_llm().chat(messages)
        self._persist(user_id, role_id, session_id, question, answer_text, hits)
        return {
            "session_id": session_id,
            "user_id": user_id,
            "role_id": role_id,
            "question": question,
            "answer": answer_text,
            "citations": self._citations(hits),
            "contexts": self._context_texts(hits),
        }

    def stream_answer(
        self,
        question: str,
        user_id: str,
        role_id: str,
        session_id: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
        rerank: bool | None = None,
    ) -> Iterator[dict[str, Any]]:
        """流式问答：产出事件流（meta / delta / done / error）。"""
        question = (question or "").strip()
        if not question:
            yield {"type": "error", "message": "问题不能为空"}
            return
        try:
            role = self.get_role(role_id)
            session_id = self.ensure_session(user_id, role_id, session_id)
            memory = self.get_memory()
            history = memory.recent_context(session_id, self.config.memory.short_term_turns)
            hits = self.retrieve(question, top_k=top_k, mode=mode, rerank=rerank)
            citations = self._citations(hits)
            messages = self._build_messages(role, question, hits, history)

            yield {
                "type": "meta",
                "session_id": session_id,
                "role_id": role_id,
                "role_name": role.name,
                "question": question,
                "citations": citations,
                "contexts": self._context_texts(hits),
            }

            collected: list[str] = []
            started = time.perf_counter()
            for piece in self.get_llm().stream(messages):
                collected.append(piece)
                yield {"type": "delta", "text": piece}

            answer_text = "".join(collected).strip()
            if not answer_text:
                answer_text = "（模型没有返回内容，请重试或换一个问法。）"
            self._persist(user_id, role_id, session_id, question, answer_text, hits)
            yield {
                "type": "done",
                "answer": answer_text,
                "citations": citations,
                "session_id": session_id,
                "timings": {"generation": round(time.perf_counter() - started, 3)},
            }
        except Exception as exc:  # noqa: BLE001 - 生成异常
            logger.exception("流式生成失败")
            yield {"type": "error", "message": str(exc)}
