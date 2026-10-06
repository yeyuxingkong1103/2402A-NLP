"""记忆服务：Redis 短期记忆（最近 N 轮）+ Milvus 长期记忆（重要摘要）。

为什么要分两层：
- 短期记忆（Redis）：对话上下文只在最近几轮内有效、要求低延迟、且允许丢失（可从 MySQL 回填），
  用 Redis List + TTL 最合适——读写快、自动过期、不会无限膨胀。
- 长期记忆（Milvus）：要"跨会话记住用户"，就得把历史压缩成摘要向量；这类信息随会话量
  线性增长，只有向量库能在海量记忆中按相似度召回最相关的几条。
  两层各司其职：Redis 保"当下对话连贯"，Milvus 保"长期陪伴连续性"。

被谁调用：rag_service（问答时取短期+长期记忆、问答后写回）、
conversation_service（删除会话时清短期）。
依赖：db.redis/db.milvus（存储）、rag.embedder（向量化）、rag.prompt（摘要提示词）、llm_service（摘要生成）。

降级原则：记忆是"增强能力"而非"必需能力"——任何一层失败都不能阻断问答主流程。
"""
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.logging import get_logger
from src.db import milvus as milvus_db
from src.db import redis as redis_db
from src.rag import prompt as prompt_utils
from src.rag.embedder import get_embedder
from src.services import conversation_service, llm_service

logger = get_logger("service.memory")


# ---------------- 短期记忆（Redis） ----------------
# 记忆分两层：短期用 Redis（读写快、自动 TTL 过期，只保留最近几轮上下文），
# 长期用 Milvus（向量检索，跨会话沉淀用户重要信息）。这样既保证低延迟，又实现"记住用户"。
def get_short_term(db: Session, user_id: int, persona_id: int, conversation_id: int,
                   limit: Optional[int] = None) -> List[Dict[str, str]]:
    """读取短期记忆（最近 N 轮对话）。优先 Redis（快、有 TTL 自动过期）。"""
    messages = redis_db.get_recent_messages(user_id, persona_id, conversation_id, limit)
    if messages:
        return messages
    # Redis 未命中（过期/重启）时从 MySQL 回填：短期记忆是易失缓存，MySQL 才是消息的持久化真源。
    messages = conversation_service.recent_messages_from_db(
        db, conversation_id, limit or settings.short_term_max_turns * 2
    )
    for msg in messages:
        redis_db.append_message(user_id, persona_id, conversation_id, msg["role"], msg["content"])
    if messages:
        logger.info("短期记忆从 MySQL 回填 %d 条，conversation=%s", len(messages), conversation_id)
    return messages


def append_short_term(user_id: int, persona_id: int, conversation_id: int,
                      role: str, content: str) -> None:
    """把一条消息追加到 Redis 短期记忆列表（由 redis_db 负责裁剪长度与设置 TTL）。"""
    redis_db.append_message(user_id, persona_id, conversation_id, role, content)


def clear_short_term(user_id: int, persona_id: int, conversation_id: int) -> None:
    """清空某会话的 Redis 短期记忆（删除会话时调用，避免残留上下文串味到新会话）。"""
    redis_db.clear_short_term(user_id, persona_id, conversation_id)


def short_term_turns(user_id: int, persona_id: int, conversation_id: int) -> int:
    """返回短期记忆当前保留的对话轮数（供展示/调试用）。"""
    # 消息是"用户+咨询师"成对存储的，所以条数除以 2 得到轮数。
    return len(redis_db.get_recent_messages(user_id, persona_id, conversation_id)) // 2


# ---------------- 长期记忆（Milvus） ----------------
def summarize_conversation(messages: List[Dict[str, str]], max_turns: int = 40) -> str:
    """调用大模型把对话压缩为摘要；失败时退化为规则摘要，保证长期记忆链路不因 LLM 故障中断。"""
    if not messages:
        return ""
    text = "\n".join(
        f"{'用户' if m.get('role') == 'user' else '咨询师'}：{m.get('content', '')}"
        for m in messages[-max_turns * 2:]
    )
    summary = llm_service.simple_complete(
        prompt_utils.SUMMARY_PROMPT.format(conversation=text),
        temperature=0.3, max_tokens=400,
    ).strip()
    if summary:
        return summary
    # 规则兜底：直接拼接用户最近发言，虽不如 LLM 精炼，但保证不丢长期记忆。
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    tail = "；".join(user_msgs[-3:])
    return f"用户主要倾诉内容：{tail[:200]}" if tail else ""


def save_long_term(db: Session, user_id: int, persona_id: int, conversation_id: int,
                   summary: str) -> Optional[int]:
    """把摘要写入 Milvus 长期记忆库。摘要需先向量化，Milvus 按向量相似度检索历史记忆。"""
    if not summary:
        return None
    vector = get_embedder().encode_query(summary)
    memory_id = milvus_db.insert_memory(persona_id, user_id, conversation_id, summary, vector)
    logger.info("长期记忆写入完成 user=%s persona=%s conversation=%s id=%s",
                user_id, persona_id, conversation_id, memory_id)
    return memory_id


def maybe_save_long_term(db: Session, conversation, user_id: int, persona_id: int,
                         force: bool = False) -> Optional[str]:
    """达到轮次阈值或强制保存时，生成并写入长期记忆。

    按固定轮次（long_term_summary_trigger）触发而非每条都存：长期记忆存 Milvus 成本较高，
    且相邻几轮的语义高度重复，每 N 轮压缩一次即可覆盖足够上下文。
    """
    trigger = settings.long_term_summary_trigger
    message_count = conversation.message_count or 0
    if not force and (message_count == 0 or message_count % trigger != 0):
        return None

    messages = conversation_service.recent_messages_from_db(db, conversation.id, limit=trigger * 2)
    summary = summarize_conversation(messages)
    if not summary:
        return None
    save_long_term(db, user_id, persona_id, conversation.id, summary)
    # 把摘要也写一份到 Redis 带 TTL，供最近会话快速取用，避免频繁查 Milvus。
    redis_db.set_json(
        redis_db.memory_summary_key(user_id, persona_id, conversation.id),
        {"summary": summary, "message_count": message_count},
        ttl=settings.short_term_ttl,
    )
    return summary


def get_long_term(user_id: int, persona_id: int, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """按当前问题检索该用户在该角色下的历史摘要。向量化失败时返回空列表（长期记忆非必需，不能阻断问答）。"""
    try:
        vector = get_embedder().encode_query(query)
    except Exception as exc:
        logger.error("长期记忆查询向量化失败：%s", exc)
        return []
    memories = milvus_db.search_memory(persona_id, user_id, vector, top_k=top_k)
    # 只返回有摘要内容的结果，过滤掉空记录。
    return [m for m in memories if m.get("summary")]


def health_check() -> bool:
    """健康检查：短期记忆依赖 Redis，故直接探活 Redis。"""
    return redis_db.health_check()