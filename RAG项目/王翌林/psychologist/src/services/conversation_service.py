"""会话与消息服务（MySQL 持久化 + Redis 短期记忆联动）。

职责：管理"会话（Conversation）"与"消息（Message）"两类实体，是 RAG 问答链路的
持久化底座——MySQL 保存可追溯的完整聊天记录，Redis 只做最近几轮的短期记忆缓存。

被谁调用：
- rag_service：问答前建会话/取历史，问答后写 user、assistant 两条消息并触发长期记忆；
- memory_service：Redis 短期记忆丢失时，从本模块 recent_messages_from_db 回填；
- api 层（conversation 路由）：会话列表 / 详情 / 消息分页 / 删除。

依赖：core（异常、日志）、db.redis（缓存联动）、models（ORM）。本模块不调用 LLM/检索，
保持"纯持久化"职责，避免与 rag_service 形成循环依赖。

事务边界约定：每个写操作（建会话、加消息、改标题）自成一次 commit；读取不加锁。
"""
import datetime as dt
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.core.exceptions import NotFoundError, PermissionError_
from src.core.logging import get_logger
from src.db import redis as redis_db
from src.models import Conversation, CounselorPersona, Message

logger = get_logger("service.conversation")


def _conv_to_dict(conv: Conversation, persona: Optional[CounselorPersona] = None) -> Dict:
    """把会话 ORM 对象转成可 JSON 序列化的字典；顺带带上角色名称/编码，前端无需再查一次。"""
    return {
        "id": conv.id,
        "user_id": conv.user_id,
        "persona_id": conv.persona_id,
        "persona_code": persona.persona_code if persona else None,
        "persona_name": persona.name if persona else None,
        "title": conv.title,
        "status": conv.status,
        "message_count": conv.message_count,
        "created_at": conv.created_at.strftime("%Y-%m-%d %H:%M:%S") if conv.created_at else None,
        "updated_at": conv.updated_at.strftime("%Y-%m-%d %H:%M:%S") if conv.updated_at else None,
    }


def create_conversation(db: Session, user_id: int, persona_id: int,
                        title: Optional[str] = None) -> Conversation:
    """新建一条会话记录（status=1 正常，0 逻辑删除）。title 缺省时用默认标题。"""
    conv = Conversation(
        user_id=user_id,
        persona_id=persona_id,
        title=title or "新的心理咨询会话",
        status=1,
        message_count=0,
    )
    # 事务边界：新建会话一次 commit；refresh 拿回自增 id 供后续消息挂载使用。
    db.add(conv)
    db.commit()
    db.refresh(conv)
    logger.info("创建会话：id=%s user=%s persona=%s", conv.id, user_id, persona_id)
    return conv


def get_conversation(db: Session, conversation_id: int, user_id: Optional[int] = None) -> Conversation:
    """按 id 取会话，做"存在性 + 软删状态 + 归属越权"三重校验（内部流程可传 user_id=None 跳过归属校验）。"""
    conv = db.get(Conversation, conversation_id)
    if not conv or conv.status == 0:
        raise NotFoundError(f"会话不存在或已删除：{conversation_id}")
    # 越权校验：会话只能由所属用户访问（user_id 为空时跳过，用于内部流程）。
    if user_id is not None and conv.user_id != user_id:
        raise PermissionError_("无权访问该会话")
    return conv


def list_conversations(db: Session, user_id: int, persona_id: Optional[int] = None,
                       include_deleted: bool = False, limit: int = 100) -> List[Dict]:
    """列出用户的会话：默认只返回未删除的，可按角色过滤，按最近更新倒序（最近的排最前）。"""
    stmt = select(Conversation).where(Conversation.user_id == user_id)
    if not include_deleted:
        stmt = stmt.where(Conversation.status == 1)
    if persona_id:
        stmt = stmt.where(Conversation.persona_id == persona_id)
    stmt = stmt.order_by(Conversation.updated_at.desc(), Conversation.id.desc()).limit(limit)
    convs = list(db.execute(stmt).scalars().all())
    # 一次查询批量取回所有会话涉及的角色，避免在循环里逐个查（N+1 查询问题）。
    personas = {
        p.id: p for p in db.execute(
            select(CounselorPersona).where(
                CounselorPersona.id.in_([c.persona_id for c in convs] or [0])
            )
        ).scalars().all()
    }
    return [_conv_to_dict(c, personas.get(c.persona_id)) for c in convs]


def delete_conversation(db: Session, conversation_id: int, user_id: int) -> None:
    """软删除会话，并同步清理对应的 Redis 短期记忆（数据仍留在 MySQL 可追溯）。"""
    conv = get_conversation(db, conversation_id, user_id)
    conv.status = 0  # 逻辑删除（软删除），保留历史数据可追溯，不物理删行
    db.commit()
    # 同步清空 Redis 短期记忆：会话已删，残留的最近消息缓存应一并失效。
    redis_db.clear_short_term(conv.user_id, conv.persona_id, conv.id)
    logger.info("逻辑删除会话：%s", conversation_id)


def get_or_create_conversation(db: Session, user_id: int, persona_id: int,
                               conversation_id: Optional[int]) -> Conversation:
    """有 conversation_id 就复用（含越权校验），否则为该用户与角色新建会话——支持"首轮自动建会话"。"""
    if conversation_id:
        return get_conversation(db, conversation_id, user_id)
    return create_conversation(db, user_id, persona_id)


def add_message(db: Session, conversation_id: int, role: str, content: str,
                tokens: Optional[int] = None, refs: Optional[Any] = None) -> Message:
    """追加一条消息（role 为 user/assistant）；refs 存该条回答引用的知识片段，用于前端溯源展示。"""
    msg = Message(conversation_id=conversation_id, role=role, content=content,
                  tokens=tokens, refs=refs)
    db.add(msg)
    conv = db.get(Conversation, conversation_id)
    if conv:
        # 同一事务里维护会话的消息计数与更新时间，保证列表排序与统计一致。
        conv.message_count = (conv.message_count or 0) + 1
        conv.updated_at = dt.datetime.now()
    db.commit()
    db.refresh(msg)
    return msg


def list_messages(db: Session, conversation_id: int, user_id: int,
                  limit: int = 200, offset: int = 0) -> Dict:
    """分页读取会话内的消息（按 id 正序，即真实对话顺序），返回 {会话id, 总数, 当前页列表}。"""
    conv = get_conversation(db, conversation_id, user_id)
    total = int(db.execute(
        select(func.count(Message.id)).where(Message.conversation_id == conv.id)
    ).scalar() or 0)
    rows = list(db.execute(
        select(Message).where(Message.conversation_id == conv.id)
        .order_by(Message.id).offset(offset).limit(limit)
    ).scalars().all())
    items = [
        {
            "id": m.id,
            "conversation_id": m.conversation_id,
            "role": m.role,
            "content": m.content,
            "tokens": m.tokens,
            "refs": m.refs,
            "created_at": m.created_at.strftime("%Y-%m-%d %H:%M:%S") if m.created_at else None,
        }
        for m in rows
    ]
    return {"conversation_id": conv.id, "total": total, "items": items}


def recent_messages_from_db(db: Session, conversation_id: int, limit: int = 20) -> List[Dict[str, str]]:
    """Redis 短期记忆丢失（过期/重启）时，从 MySQL 按 id 倒序取最近消息再反转回正序回填。"""
    rows = list(db.execute(
        select(Message).where(Message.conversation_id == conversation_id)
        .order_by(Message.id.desc()).limit(limit)
    ).scalars().all())
    return [{"role": m.role, "content": m.content} for m in reversed(rows)]


def auto_title(db: Session, conversation: Conversation, user_message: str) -> None:
    """首次对话时用用户消息生成标题（截断 20 字）；已有非默认标题则不再覆盖。"""
    if conversation.title and conversation.title != "新的心理咨询会话":
        return
    title = user_message.strip().replace("\n", " ")[:20]
    conversation.title = title or conversation.title
    db.commit()