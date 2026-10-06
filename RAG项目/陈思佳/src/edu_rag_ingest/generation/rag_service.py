from __future__ import annotations

"""教师问答服务，串联会话记忆、知识检索和大模型回答。"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ..config.config import AppConfig
from .llm import DashScopeChatClient
from ..storage.memory_store import MemoryStore
from ..retrieval.retriever import MilvusRetriever, build_citations, build_teacher_prompt


@dataclass(frozen=True)
class AnswerResult:
    """一次问答的正文和引用结果。"""
    answer: str
    citations: list[dict[str, Any]]


class TeacherRAGService:
    """编排 Redis 记忆、知识检索和大模型生成。"""
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.retriever = MilvusRetriever(config)
        self.llm = DashScopeChatClient(config.qa)
        self.memory = MemoryStore(config.memory)

    def ask(
        self,
        question: str,
        session_id: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> AnswerResult:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")
        session = self.memory.create_session(session_id)
        recent_messages = self.memory.get_session_messages(session_id)
        long_term_memories = self.memory.get_long_term_memories(session_id)
        memory_context = self._format_memory(recent_messages, long_term_memories)
        chunks = self.retriever.search(question, top_k=top_k, filters=filters)
        prompt = build_teacher_prompt(question, chunks, self.config.qa.max_context_chars, memory_context)
        try:
            answer = self.llm.generate(prompt)
        except Exception as error:
            answer = self._fallback_answer(question, chunks, error)
        self.memory.append_message(session_id, {"role": "user", "content": question})
        self.memory.append_message(session_id, {"role": "assistant", "content": answer})
        self.memory.save_long_term_memory(session_id, f"教师问题：{question}\n助手回答：{answer}")
        self.memory.touch_session(session_id, title=question[:24])
        return AnswerResult(answer=answer, citations=build_citations(chunks))

    def stream_ask(
        self,
        question: str,
        session_id: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")
        self.memory.create_session(session_id)
        recent_messages = self.memory.get_session_messages(session_id)
        long_term_memories = self.memory.get_long_term_memories(session_id)
        memory_context = self._format_memory(recent_messages, long_term_memories)
        chunks = self.retriever.search(question, top_k=top_k, filters=filters)
        prompt = build_teacher_prompt(question, chunks, self.config.qa.max_context_chars, memory_context)
        citations = build_citations(chunks)
        yield {"event": "meta", "session_id": session_id, "citations": citations}
        parts: list[str] = []
        try:
            for part in self.llm.stream_generate(prompt):
                parts.append(part)
                yield {"event": "token", "content": part}
        except Exception as error:
            fallback = self._fallback_answer(question, chunks, error)
            parts = [fallback]
            yield {"event": "error", "content": fallback}
        answer = "".join(parts).strip()
        self.memory.append_message(session_id, {"role": "user", "content": question})
        self.memory.append_message(session_id, {"role": "assistant", "content": answer})
        self.memory.save_long_term_memory(session_id, f"教师问题：{question}\n助手回答：{answer}")
        self.memory.touch_session(session_id, title=question[:24])
        yield {"event": "done", "session_id": session_id, "citations": citations}


        if not chunks:
            return (
                "已完成知识库检索，但当前没有找到足够相关的资料。\n\n"
                f"外部大模型暂时不可用：{error}\n\n"
                "建议稍后恢复网络或检查通义千问 API 连接后重试。"
            )
        excerpts = []
        for index, chunk in enumerate(chunks[:3], start=1):
            metadata = chunk.metadata
            source = metadata.get("document_type") or metadata.get("title") or metadata.get("source_file") or "知识库资料"
            content = chunk.content.strip().replace("\n", " ")[:260]
            excerpts.append(f"{index}. 来源：{source}\n   摘要：{content}...")
        return (
            "外部大模型暂时无法连接，我先基于本地知识库检索结果给出参考摘要。\n\n"
            f"教师问题：{question}\n\n"
            "检索到的相关资料：\n"
            + "\n".join(excerpts)
            + f"\n\n连接通义千问失败：{error}\n"
            "你可以先根据以上资料备课；网络恢复后再次提问可获得更完整的生成式回答。"
        )

    @staticmethod
    def _format_memory(
        recent_messages: list[dict[str, Any]],
        long_term_memories: list[dict[str, Any]],
    ) -> str:
        sections: list[str] = []
        if recent_messages:
            recent = "\n".join(
                f"{item.get('role', 'unknown')}: {item.get('content', '')}" for item in reversed(recent_messages)
            )
            sections.append(f"最近对话：\n{recent}")
        if long_term_memories:
            durable = "\n".join(item.get("content", "") for item in long_term_memories[:5])
            sections.append(f"长期记忆：\n{durable}")
        return "\n\n".join(sections)
