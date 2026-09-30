"""短期记忆与长期记忆的协调器。

- Redis：保存当前会话最近消息，供下一轮直接拼接上下文；
- Milvus：保存跨会话摘要、重要事实和用户偏好；
- MySQL：保存压缩日志和最终会话归档。

记忆是问答增强能力，不应因为单次写入失败而让主问答接口整体失败。
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.memory.extractors import InformationExtractor
from app.memory.memory_models import MemoryContext
from app.memory.milvus_memory import MilvusMemory
from app.memory.redis_memory import RedisMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """Coordinate short-term messages, long-term recall, compression, and archive."""

    def __init__(
        self,
        redis_memory: RedisMemory,
        milvus_memory: MilvusMemory,
        mysql_client: Any,
        llm_client: Any | None = None,
    ) -> None:
        self.redis_memory = redis_memory
        self.milvus_memory = milvus_memory
        self.mysql = mysql_client
        self.llm_client = llm_client
        self.extractor = InformationExtractor()
        self.compression_threshold = 10
        self.max_short_term_turns = 5

    def get_memory_context(
        self,
        session_id: str,
        user_id: int,
        tenant_id: int,
        current_query: str,
        enable_long_term: bool = True,
        max_short_turns: int | None = None,
    ) -> MemoryContext:
        """组合当前会话消息与租户隔离的长期记忆。"""
        # 短期消息按轮数截断，避免会话无限增长挤占 Prompt 上下文。
        short_term = self.redis_memory.get_messages(
            session_id,
            max_turns=max_short_turns or self.max_short_term_turns,
        )
        long_term = {"summaries": [], "facts": [], "preferences": []}
        if enable_long_term:
            try:
                # user_id + tenant_id 是长期记忆检索的隔离边界，不能只按相似度搜索。
                long_term = self.milvus_memory.retrieve_long_term_memory(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    query=current_query,
                )
            except Exception as exc:
                # Milvus 记忆暂时不可用时退化为短期记忆，不影响用户继续问答。
                logger.warning("长期记忆检索失败: %s", exc)
        return MemoryContext(
            short_term,
            long_term.get("summaries", []),
            long_term.get("preferences", []),
            long_term.get("facts", []),
        )

    def add_message(self, session_id: str, user_id: int, tenant_id: int, message: dict[str, Any]) -> None:
        """Write a message to Redis, then observe it for long-term memory."""
        try:
            self.redis_memory.add_message(session_id, message)
            self._process_message(session_id, user_id, tenant_id, message)
        except Exception as exc:
            logger.error("添加消息失败: %s", exc, exc_info=True)

    def observe_turn(
        self,
        session_id: str,
        user_id: int,
        tenant_id: int,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """观察一轮已写入 Redis 的消息，只提取长期信息，不重复写短期记忆。"""
        try:
            # 用户消息负责实体、意图和重要事实提取；助手消息写完后再触发压缩检查。
            self._process_message(
                session_id,
                user_id,
                tenant_id,
                {"role": "user", "content": user_text},
                check_compression=False,
            )
            self._process_message(
                session_id,
                user_id,
                tenant_id,
                {"role": "assistant", "content": assistant_text},
                check_compression=True,
            )
        except Exception as exc:
            logger.warning("长期记忆观察失败: %s", exc)

    def _process_message(self, session_id: str, user_id: int, tenant_id: int, message: dict[str, Any], *, check_compression: bool = True) -> None:
        if message.get("role") == "user":
            content = message.get("content", "")
            entities = self.extractor.extract_entities(content)
            intent = self.extractor.classify_intent(content)
            message["metadata"] = {"entities": entities, "intent": intent, "sentiment": self.extractor.detect_sentiment(content)}
            if self.extractor.is_important_message(message, entities, intent):
                self._save_important_fact(user_id, tenant_id, message, entities, intent, session_id)
        count = self.redis_memory.get_message_count(session_id) if check_compression else 0
        if check_compression and count >= self.compression_threshold and count % self.compression_threshold == 0:
            self.compress_memory(session_id, user_id, tenant_id)

    def _save_important_fact(self, user_id: int, tenant_id: int, message: dict[str, Any], entities: dict[str, Any], intent: str, session_id: str) -> None:
        content = self._redact_pii(message.get("content", ""))
        fact_type = "order" if entities.get("order_id") else "issue" if intent == "complaint" else "feedback" if message.get("metadata", {}).get("sentiment") == "negative" else "general"
        try:
            self.milvus_memory.save_important_fact(user_id=user_id, tenant_id=tenant_id, fact_type=fact_type, fact_text=content, source_session_id=session_id, confidence=0.9)
        except Exception as exc:
            logger.warning("保存重要事实失败: %s", exc)

    @staticmethod
    def _redact_pii(content: str) -> str:
        content = re.sub(InformationExtractor.ENTITY_PATTERNS["phone"], "[手机号已脱敏]", content)
        return re.sub(InformationExtractor.ENTITY_PATTERNS["email"], "[邮箱已脱敏]", content, flags=re.IGNORECASE)

    def compress_memory(self, session_id: str, user_id: int, tenant_id: int) -> bool:
        """把较早消息压缩成摘要，同时保留最近两轮原始对话。"""
        try:
            messages = self.redis_memory.get_all_messages(session_id)
            if len(messages) <= 4:
                return False

            # 最近四条消息（两轮）继续以原文参与 Prompt，较旧消息被总结以控制上下文长度。
            summary = self._generate_summary(messages[:-4])
            if not summary:
                return False
            summary_message = {"role": "system", "content": f"[历史对话摘要] {summary}", "timestamp": messages[-1].get("timestamp"), "is_summary": True}
            before = len(messages)
            after = self.redis_memory.compress_messages(session_id, summary_message, keep_last_turns=2)
            self.milvus_memory.save_conversation_summary(session_id=session_id, user_id=user_id, tenant_id=tenant_id, summary_text=summary, turn_count=max(1, len(messages[:-4]) // 2), key_intents=self.extractor.extract_key_intents(messages[:-4]), session_start=int(time.time()) - len(messages) * 60, session_end=int(time.time()))
            self._save_preferences(user_id, tenant_id, messages)
            self._log_compression(session_id, "partial", before, after, summary)
            return True
        except Exception as exc:
            logger.error("压缩记忆失败: %s", exc, exc_info=True)
            return False

    def _generate_summary(self, messages: list[dict[str, Any]]) -> str | None:
        if not messages:
            return ""
        if not self.llm_client:
            return self._simple_summary(messages)
        conversation = "\n".join(f"{item.get('role')}: {item.get('content')}" for item in messages)
        try:
            return self.llm_client.generate(prompt=f"请简洁总结以下对话的要点（50字以内）：\n\n{conversation}", max_tokens=150, temperature=0.3).strip()
        except Exception as exc:
            logger.warning("LLM摘要生成失败: %s", exc)
            return self._simple_summary(messages)

    @staticmethod
    def _simple_summary(messages: list[dict[str, Any]]) -> str:
        questions = [item.get("content", "")[:50] for item in messages if item.get("role") == "user"]
        return f"用户咨询了 {len(questions)} 个问题，包括：{questions[0]}..." if questions else f"对话包含 {len(messages)} 条消息"

    def _save_preferences(self, user_id: int, tenant_id: int, messages: list[dict[str, Any]]) -> None:
        for preference in self.extractor.extract_user_preference(messages):
            self.milvus_memory.save_user_preference(user_id=user_id, tenant_id=tenant_id, preference_type=preference["type"], preference_text=preference["text"], confidence=float(preference.get("confidence", 1.0)))

    def _log_compression(self, session_id: str, compression_type: str, before_count: int, after_count: int, summary_text: str) -> None:
        try:
            self.mysql.execute("INSERT INTO memory_compression_logs (session_id, compression_type, before_message_count, after_message_count, summary_text, compression_ratio) VALUES (%s, %s, %s, %s, %s, %s)", (session_id, compression_type, before_count, after_count, summary_text, after_count / before_count if before_count else 1.0))
        except Exception as exc:
            logger.warning("记录压缩日志失败: %s", exc)

    def archive_session(self, session_id: str, user_id: int, tenant_id: int) -> bool:
        """会话结束时同时保存可检索摘要和可审计 JSON 原文归档。"""
        try:
            messages = self.redis_memory.get_all_messages(session_id)
            if not messages:
                return False
            summary = self._generate_summary(messages) or ""
            intents = self.extractor.extract_key_intents(messages)
            now = int(time.time())
            self.milvus_memory.save_conversation_summary(session_id=session_id, user_id=user_id, tenant_id=tenant_id, summary_text=summary, turn_count=len(messages) // 2, key_intents=intents, session_start=now - len(messages) * 60, session_end=now)
            self.mysql.execute("INSERT INTO conversation_archives (session_id, user_id, tenant_id, messages, summary, turn_count, main_intents) VALUES (%s, %s, %s, %s, %s, %s, %s)", (session_id, user_id, tenant_id, json.dumps(messages, ensure_ascii=False), summary, len(messages) // 2, json.dumps(intents, ensure_ascii=False)))
            self._log_compression(session_id, "final", len(messages), 1, summary)
            return True
        except Exception as exc:
            logger.error("归档会话失败: %s", exc, exc_info=True)
            return False

    @staticmethod
    def format_memory_for_prompt(context: MemoryContext) -> str:
        """Render bounded memory sections for the prompt builder."""
        sections = []
        if context.user_preferences:
            sections.append("用户偏好：\n" + "\n".join(f"- {item['text']}" for item in context.user_preferences[:3]))
        if context.important_facts:
            sections.append("重要信息：\n" + "\n".join(f"- {item['text']}" for item in context.important_facts[:3]))
        if context.long_term_summaries:
            sections.append("历史对话：\n" + "\n".join(f"- {item['summary']}" for item in context.long_term_summaries[:2]))
        if context.short_term_messages:
            sections.append("近期对话：\n" + "\n".join(f"{item['role']}: {item['content']}" for item in context.short_term_messages[-6:]))
        return "\n\n".join(sections)


__all__ = ["MemoryContext", "MemoryManager"]
