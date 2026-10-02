from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
import json

from fastapi import APIRouter, Request
from pydantic import BaseModel, field_validator

from ..auth import current_user
from ..models import ModelGateway
from ..storage.redis import RedisStore
from ..storage.mysql import (
    ConversationHistoryStore,
    LongTermMemoryStore,
    MemoryConflictStore,
    MySQLStore,
    UserStateStore,
)
from .long_memory import LongMemoryManager, MemoryExtractor
from .resolver import CoreferenceResolver
from .short_memory import ContextCompressor, ShortMemoryRedisStore


logger = logging.getLogger("law_rag.memory")
router = APIRouter(prefix="/api/v1/memory", tags=["memory"])
RECENT_MESSAGE_LIMIT = 40


class MemorySettingsRequest(BaseModel):
    enable_long_memory: bool = False


class ConflictResolveRequest(BaseModel):
    action: str

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: str) -> str:
        value = str(value or "").strip()
        if value not in {"accept_new", "keep_old", "discard"}:
            raise ValueError("冲突处理动作必须是 accept_new、keep_old 或 discard")
        return value


@dataclass
class MemoryContext:
    """一次问答需要的记忆集合，包含短期对话、长期事实、旧版案件记忆、案件摘要和指代消解结果。"""

    short_term: list[dict] = field(default_factory=list)
    short_memory: dict = field(default_factory=dict)
    original_hits: list[dict] = field(default_factory=list)
    long_memory: list[dict] = field(default_factory=list)
    legacy_case_memory: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    user_profile: dict = field(default_factory=dict)
    long_term_count: int = 0
    resolved_query: str = ""
    resolved_references: dict = field(default_factory=dict)
    enable_long_memory: bool = False
    user_id: str = ""
    conversation_id: str = ""
    history_ids: list[int] = field(default_factory=list)


class MemoryOrchestrator:
    """统一编排短期记忆、长期记忆、原始消息回查和指代消解。"""

    def __init__(
        self,
        settings,
        redis: RedisStore,
        mysql: MySQLStore,
        model: ModelGateway,
        user_state: UserStateStore | None = None,
    ):
        self.settings = settings
        self.model = model
        self.mysql = mysql
        self.user_state = user_state
        self.history = ConversationHistoryStore(mysql)
        self.redis = redis
        self.short_memory = ShortMemoryRedisStore(redis, ttl=getattr(settings, "history_ttl", 60 * 60 * 24 * 30))
        self.compressor = ContextCompressor(model, token_limit=getattr(settings, "short_memory_compress_token_limit", 8000))
        self.long_memory_store = LongTermMemoryStore(mysql)
        self.conflict_store = MemoryConflictStore(mysql)
        self.long_memory = LongMemoryManager(self.long_memory_store, self.conflict_store)
        self.resolver = CoreferenceResolver()
        self._long_memory_cache = {}  # 简单缓存，避免频繁查库

    def _estimate_tokens(self, value: object) -> int:
        try:
            return max(0, len(json.dumps(value, ensure_ascii=False)) // 2)
        except Exception:
            return 0

    def _persist_long_memory_snapshot(self, user_id: str, conversation_id: str) -> dict:
        threshold = max(1000, int(getattr(self.settings, "long_memory_persist_token_limit", 80000) or 80000))
        messages = self.history.list_all(user_id, conversation_id, limit=2000)
        if not messages:
            return {"persisted": 0, "skipped": 0, "message_count": 0, "token_estimate": 0, "threshold": threshold}

        token_estimate = self._estimate_tokens(messages)
        if token_estimate < threshold:
            return {"persisted": 0, "skipped": len(messages), "message_count": len(messages), "token_estimate": token_estimate, "threshold": threshold}

        last_persisted_message_id = self.long_memory_store.latest_source_message_id(user_id, conversation_id)
        new_messages = [row for row in messages if int(row.get("id") or 0) > last_persisted_message_id]
        if last_persisted_message_id and self._estimate_tokens(new_messages) < threshold:
            return {"persisted": 0, "skipped": len(messages), "message_count": len(messages), "token_estimate": token_estimate, "threshold": threshold}
        last_message_id = max(int(row.get("id") or 0) for row in messages)
        if last_message_id <= last_persisted_message_id:
            return {"persisted": 0, "skipped": len(messages), "message_count": len(messages), "token_estimate": token_estimate, "threshold": threshold}

        extractor = MemoryExtractor()
        candidates = extractor.extract(new_messages)
        persisted = 0
        for candidate in candidates:
            memory_id = self.long_memory.add_memory(
                user_id=user_id,
                memory_type=candidate.memory_type,
                content=candidate.content,
                importance=candidate.importance,
                confidence=candidate.confidence,
                scope=candidate.scope,
                source_conversation_id=candidate.source_conversation_id,
                source_message_id=candidate.source_message_id,
            )
            if memory_id:
                persisted += 1

        return {"persisted": persisted, "skipped": 0, "message_count": len(messages), "token_estimate": token_estimate, "threshold": threshold}

    def enable_long_memory(self, user_id: str) -> bool:
        """检查用户是否开启了长期记忆功能。"""
        if not user_id:
            return False
        
        # 检查缓存
        cache_key = f"enable_long_memory:{user_id}"
        if cache_key in self._long_memory_cache:
            return self._long_memory_cache[cache_key]
        
        try:
            if self.user_state:
                # 从用户状态表读取设置
                setting = self.user_state.get_setting(user_id, "enable_long_memory")
                enabled = setting is True
            else:
                # 回退到从 MySQL 直接查询
                row = self.mysql.fetch_one(
                    "SELECT enable_long_memory FROM user_states WHERE user_id=%s",
                    (user_id,)
                )
                enabled = bool(row and row.get("enable_long_memory"))
            
            # 缓存结果（5分钟）
            self._long_memory_cache[cache_key] = enabled
            return enabled
        except Exception as exc:
            logger.warning(
                "读取长期记忆设置失败，默认关闭",
                extra={"event": "enable_long_memory_failed", "fields": {"user_id": user_id, "error": str(exc)}},
                exc_info=True,
            )
            return False

    def save_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        *,
        question_text: str = "",
        source_count: int = 0,
        message_meta: dict | None = None,
        solution_json: dict | None = None
    ) -> int:
        """保存单条消息到历史记录。"""
        message_id = self.history.add(
            user_id,
            conversation_id,
            role,
            content,
            question_text=question_text,
            source_count=source_count,
            message_meta=message_meta,
            solution_json=solution_json,
        )
        return message_id

    def save_turn(
        self,
        user_id: str,
        conversation_id: str,
        question: str,
        answer: str,
        *,
        answer_payload: dict | None = None,
        source_count: int = 0,
        solution_json: dict | None = None
    ) -> None:
        """保存一轮用户问题和助手回答，并尝试同步短期压缩记忆。"""
        user_history_id = self.save_message(user_id, conversation_id, "user", question, question_text=question)
        assistant_history_id = self.save_message(
            user_id,
            conversation_id,
            "assistant",
            answer,
            question_text=question,
            source_count=source_count,
            message_meta=answer_payload or {},
            solution_json=solution_json,
        )
        
        # 更新短期记忆
        try:
            with self.short_memory.with_lock(user_id, conversation_id) as acquired:
                memory = self.short_memory.append_turn(user_id, conversation_id, question, answer, (user_history_id, assistant_history_id))
                if acquired:
                    memory = self.compressor.maybe_compress(memory)
                    self.short_memory.save(user_id, conversation_id, memory)
        except Exception as exc:
            logger.warning(
                "短期记忆更新失败，conversation history 已保留",
                extra={"event": "short_memory_update_failed", "fields": {"conversation_id": conversation_id, "error": str(exc)}},
                exc_info=True,
            )

        try:
            with self.redis.lock(f"long_memory_snapshot_lock:{user_id}:{conversation_id}", ttl=60) as acquired:
                if acquired:
                    result = self._persist_long_memory_snapshot(user_id, conversation_id)
                    if result.get("persisted"):
                        logger.info(
                            "长期记忆已落库",
                            extra={"event": "long_memory_persisted", "fields": {"user_id": user_id, "conversation_id": conversation_id, **result}},
                        )
        except Exception as exc:
            logger.warning(
                "长期记忆落库失败",
                extra={"event": "long_memory_persist_failed", "fields": {"conversation_id": conversation_id, "error": str(exc)}},
                exc_info=True,
            )

    def long_term_count(self, user_id: str, conversation_id: str) -> int:
        """获取会话的消息总数。"""
        count = self.history.count(user_id, conversation_id)
        return count

    def load_context(
        self,
        user_id: str,
        conversation_id: str,
        user_profile: dict | None = None,
        query: str = ""
    ) -> MemoryContext:
        """读取当前问答所需记忆，并把指代消解后的 query 一并返回。"""
        # 1. 加载短期记忆
        short = self.short_memory.load(user_id, conversation_id)
        messages = short.get("messages") or []

        # 2. 获取最近消息（从短期记忆或历史记录）
        if messages:
            recent = messages[-RECENT_MESSAGE_LIMIT:]
        else:
            rows = self.history.list_recent(user_id, conversation_id, limit=RECENT_MESSAGE_LIMIT)
            if rows:
                recent = [{"role": row.get("role"), "content": row.get("content"), "history_id": row.get("id")} for row in rows]
            else:
                recent = []

        # 3. 长期记忆仅落库，不参与本次回答检索。
        long_memory_enabled = False
        long_memories: list[dict] = []

        # 5. 指代消解
        resolved = self.resolver.resolve(
            query,
            short,
            long_memories,
            str(short.get("summary") or "")
        ) if query else {"resolved_query": query, "resolved_references": {}}

        # 6. 原始消息回查（当用户使用指代词时）
        original_hits: list[dict] = []
        if query:
            compact_query = re.sub(r"[^\w一-鿿]+", "", str(query or "")).casefold()
            summary_text = str(short.get("summary") or "").strip()
            if len(compact_query) >= 2 and summary_text:
                resolved_references = dict(resolved.get("resolved_references") or {})
                should_retrieve = len(recent or []) < 4 or (
                    compact_query and compact_query not in re.sub(r"[^\w一-鿿]+", "", summary_text).casefold()
                ) or (
                    not resolved_references and any(marker in str(query or "") for marker in ("他", "她", "它", "这个", "那个", "上述", "刚才", "前面", "多少", "怎么赔"))
                )
                if should_retrieve:
                    try:
                        rows = self.history.search_messages(user_id, conversation_id, query, limit=6)
                        if not rows and any(marker in str(query or "") for marker in ("他", "她", "它", "这个", "那个", "上述", "刚才", "前面", "多少", "怎么赔")):
                            rows = [{**row, "score": 0.61, "retrieval_reason": "original_recent_fallback"} for row in self.history.list_recent(user_id, conversation_id, limit=6)]
                    except Exception as exc:
                        logger.warning(
                            "原始会话内容回查失败，继续使用摘要上下文",
                            extra={"event": "original_context_retrieval_failed", "fields": {"conversation_id": conversation_id, "error": str(exc)}},
                            exc_info=True,
                        )
                        rows = []
                    recent_ids = {str(row.get("id") or "") for row in recent or [] if row.get("id")}
                    for row in rows:
                        message_id = str(row.get("id") or "")
                        if message_id and message_id in recent_ids:
                            continue
                        content = str(row.get("content") or "").strip()
                        if not content:
                            continue
                        original_hits.append(
                            {
                                "source_id": f"conversation:{conversation_id}:{message_id or len(original_hits)}",
                                "source_type": "conversation_original",
                                "retrieval_channel": "original_context",
                                "retrieval_reason": row.get("retrieval_reason", "original_keyword"),
                                "title": "原始会话记录",
                                "role": row.get("role", ""),
                                "content": content,
                                "score": float(row.get("score", 0) or 0),
                                "message_id": row.get("id"),
                                "conversation_id": conversation_id,
                            }
                        )

        # 7. 加载旧版案件记忆（兼容）
        legacy_case_memory = {}
        row = self.mysql.fetch_one(
            "SELECT memory_json FROM case_memories WHERE user_id=%s AND session_id=%s",
            (user_id, conversation_id),
        )
        if row and row.get("memory_json"):
            value = row.get("memory_json")
            if isinstance(value, dict):
                legacy_case_memory = value
            else:
                try:
                    parsed = json.loads(str(value))
                    if isinstance(parsed, dict):
                        legacy_case_memory = parsed
                except Exception:
                    legacy_case_memory = {}

        history_ids = [int(row.get("history_id") or row.get("id") or 0) for row in (recent or []) if str(row.get("history_id") or row.get("id") or "0").isdigit()]
        return MemoryContext(
            short_term=recent,
            short_memory=short,
            original_hits=original_hits,
            long_memory=long_memories,
            legacy_case_memory=legacy_case_memory,
            summary={"summary": short.get("summary", ""), "message_count": self.long_term_count(user_id, conversation_id)} if short.get("summary") else {},
            user_profile=user_profile or {},
            long_term_count=self.long_term_count(user_id, conversation_id),
            resolved_query=str(resolved.get("resolved_query") or query or ""),
            resolved_references=dict(resolved.get("resolved_references") or {}),
            enable_long_memory=long_memory_enabled,
            user_id=str(user_id),
            conversation_id=str(conversation_id),
            history_ids=list(dict.fromkeys(history_ids)),
        )

    def delete_session(self, user_id: str, conversation_id: str) -> None:
        """按用户和会话删除聊天记录、短期记忆及由该会话提取的长期记忆。"""
        session = str(conversation_id or "").strip()
        if not session:
            return
        self.short_memory.delete(user_id, session)
        self.history.delete_session(user_id, session)
        self.long_memory_store.delete_session(user_id, session)
        cache_key = f"enable_long_memory:{user_id}"
        if cache_key in self._long_memory_cache:
            del self._long_memory_cache[cache_key]
        logger.info(
            "会话已删除",
            extra={"event": "memory_session_deleted", "fields": {"user_id": user_id, "conversation_id": session}},
        )

    def run_daily_for_user(self, user_id: str, batch_size: int = 200) -> dict:
        """为用户运行记忆落库任务；长期记忆只保存，不参与问答检索。"""
        try:
            conversations = self.history.list_conversations(user_id, limit=100)
            total_processed = 0
            total_inserted = 0
            total_skipped = 0
            
            for conv in conversations:
                conv_id = conv.get("conversation_id")
                if not conv_id:
                    continue
                result = self._persist_long_memory_snapshot(user_id, str(conv_id))
                total_processed += int(result.get("message_count") or 0)
                total_inserted += int(result.get("persisted") or 0)
                total_skipped += int(result.get("skipped") or 0)
            
            return {
                "user_id": user_id,
                "processed_messages": total_processed,
                "processed": total_processed,
                "inserted": total_inserted,
                "merged": 0,
                "conflicts": 0,
                "skipped": total_skipped,
                "disabled": False,
            }
        except Exception as exc:
            logger.error(
                "每日记忆提取失败",
                extra={"event": "daily_memory_extract_failed", "fields": {"user_id": user_id, "error": str(exc)}},
                exc_info=True,
            )
            return {
                "user_id": user_id,
                "processed_messages": 0,
                "processed": 0,
                "inserted": 0,
                "merged": 0,
                "conflicts": 0,
                "skipped": 0,
                "disabled": False,
                "error": str(exc),
            }


def memory_service(request: Request) -> MemoryOrchestrator:
    """从请求中获取记忆服务实例。"""
    return request.app.state.services["memory"]


@router.get("/settings")
def get_memory_settings(request: Request):
    """获取用户记忆设置。"""
    user = current_user(request)
    service = memory_service(request)
    enabled = service.enable_long_memory(user["user_id"])
    return {"success": True, "data": {"enable_long_memory": enabled}}


@router.put("/settings")
def update_memory_settings(payload: MemorySettingsRequest, request: Request):
    """更新用户记忆设置。"""
    user = current_user(request)
    service = memory_service(request)
    
    try:
        if service.user_state:
            service.user_state.set_setting(user["user_id"], "enable_long_memory", payload.enable_long_memory)
        else:
            # 直接更新数据库
            service.mysql.execute(
                """
                INSERT INTO user_states (user_id, enable_long_memory, updated_at)
                VALUES (%s, %s, NOW())
                ON DUPLICATE KEY UPDATE enable_long_memory=%s, updated_at=NOW()
                """,
                (user["user_id"], payload.enable_long_memory, payload.enable_long_memory)
            )
        
        # 清除缓存
        cache_key = f"enable_long_memory:{user['user_id']}"
        if cache_key in service._long_memory_cache:
            del service._long_memory_cache[cache_key]
        
        enabled = service.enable_long_memory(user["user_id"])
        logger.info(
            "记忆设置已更新",
            extra={"event": "memory_settings_updated", "fields": {"user_id": user["user_id"], "enable_long_memory": enabled}}
        )
        return {"success": True, "data": {"enable_long_memory": enabled}}
    except Exception as exc:
        logger.error(
            "更新记忆设置失败",
            extra={"event": "memory_settings_update_failed", "fields": {"user_id": user["user_id"], "error": str(exc)}},
            exc_info=True,
        )
        return {"success": False, "message": str(exc)}


@router.get("/conflicts/pending")
def list_conflicts(request: Request, limit: int = 50):
    """列出待解决的记忆冲突。"""
    user = current_user(request)
    return {"success": True, "data": memory_service(request).long_memory.list_conflicts(user["user_id"], limit=limit)}


@router.post("/conflicts/{conflict_id}/resolve")
def resolve_conflict(conflict_id: int, payload: ConflictResolveRequest, request: Request):
    """解决记忆冲突。"""
    user = current_user(request)
    return {"success": True, "data": memory_service(request).long_memory.resolve_conflict(user["user_id"], conflict_id, payload.action)}


@router.get("")
def list_memories(request: Request, limit: int = 100, offset: int = 0):
    """列出用户的长期记忆。"""
    user = current_user(request)
    return {"success": True, "data": memory_service(request).long_memory.list_memories(user["user_id"], limit=limit, offset=offset)}


@router.delete("")
def delete_all_memories(request: Request):
    """删除所有长期记忆（仅当长期记忆启用时）。"""
    user = current_user(request)
    service = memory_service(request)
    
    if not service.enable_long_memory(user["user_id"]):
        return {"success": True, "data": {"deleted": False, "count": 0, "locked": True, "message": "长期记忆功能未启用"}}
    
    # 批量删除（软删除）
    count = service.long_memory_store.delete_all(user["user_id"])
    return {"success": True, "data": {"deleted": True, "count": count, "locked": False, "message": f"已删除 {count} 条长期记忆"}}


@router.get("/{memory_id}/explain")
def explain_memory(memory_id: int, request: Request):
    """解释某条长期记忆的来源。"""
    user = current_user(request)
    return {"success": True, "data": memory_service(request).long_memory.explain(user["user_id"], memory_id)}


@router.delete("/{memory_id}")
def delete_memory(memory_id: int, request: Request):
    """删除单条长期记忆。"""
    user = current_user(request)
    service = memory_service(request)
    if not service.enable_long_memory(user["user_id"]):
        return {"success": True, "data": {"deleted": False, "memory_id": memory_id, "locked": True, "message": "长期记忆功能未启用"}}
    deleted = service.long_memory_store.delete(user["user_id"], memory_id)
    return {"success": True, "data": {"deleted": deleted, "memory_id": memory_id, "locked": False, "message": "已删除" if deleted else "未找到该记忆"}}


def run_daily_memory_job(settings=None, user_limit: int = 500) -> dict:
    """运行每日记忆提取任务。"""
    # 这是一个独立的函数，需要在主应用中注入 service
    # 实际使用时需要从 app.state 获取 services
    return {
        "user_count": 0,
        "results": [],
        "disabled": False,
        "message": "需要在主应用中注入 MemoryOrchestrator 才能运行"
    }


__all__ = ["MemoryContext", "MemoryOrchestrator", "router", "run_daily_memory_job"]
