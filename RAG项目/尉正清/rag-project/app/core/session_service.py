# app/core/session_service.py
"""会话与消息的持久化（MySQL）。

Redis 只保存最近若干轮热数据，完整历史归档在 MySQL，
这样既保证多轮对话的低延迟，也不会因为 Redis 过期而丢失记录。
"""
import json
import uuid
from typing import Dict, List, Optional

from sqlalchemy import delete, desc, select
from sqlalchemy.orm import Session

from app.models.tables import ChatSession, Message
import logging

logger = logging.getLogger(__name__)


class SessionService:

    @staticmethod
    def create(db: Session, user_id: int, role_key: str,
               title: str = "", session_id: Optional[str] = None) -> ChatSession:
        s = ChatSession(
            session_id=session_id or uuid.uuid4().hex,
            user_id=user_id,
            role_key=role_key,
            title=(title or "")[:120],
            turn_count=0,
        )
        db.add(s)
        db.flush()
        logger.info("新建会话 %s (user=%s, role=%s)", s.session_id, user_id, role_key)
        return s

    @staticmethod
    def get(db: Session, session_id: str) -> Optional[ChatSession]:
        if not session_id:
            return None
        return db.execute(select(ChatSession).where(
            ChatSession.session_id == session_id)).scalars().first()

    @staticmethod
    def get_or_create(db: Session, user_id: int, role_key: str,
                      session_id: Optional[str], title: str = "") -> ChatSession:
        """传了 session_id 就复用（并校验归属），否则新建。"""
        if session_id:
            s = SessionService.get(db, session_id)
            if s is None:
                return SessionService.create(db, user_id, role_key,
                                             title=title, session_id=session_id)
            if s.user_id != user_id:
                raise PermissionError("会话不属于当前用户")
            return s
        return SessionService.create(db, user_id, role_key, title=title)

    @staticmethod
    def touch(db: Session, session: ChatSession, question: str) -> None:
        """累加轮次；首轮时用提问做标题。"""
        session.turn_count = (session.turn_count or 0) + 1
        if not session.title:
            session.title = question[:120]
        db.flush()

    @staticmethod
    def add_message(db: Session, session_id: str, role: str, content: str,
                    sources: Optional[List[Dict]] = None) -> Message:
        m = Message(
            session_id=session_id, role=role, content=content,
            sources=json.dumps(sources, ensure_ascii=False) if sources else None,
        )
        db.add(m)
        db.flush()
        return m

    @staticmethod
    def list_sessions(db: Session, user_id: int,
                      role_key: Optional[str] = None,
                      limit: int = 50) -> List[ChatSession]:
        stmt = select(ChatSession).where(ChatSession.user_id == user_id)
        if role_key:
            stmt = stmt.where(ChatSession.role_key == role_key)
        stmt = stmt.order_by(desc(ChatSession.updated_at)).limit(limit)
        return list(db.execute(stmt).scalars().all())

    @staticmethod
    def get_messages(db: Session, session_id: str,
                     limit: int = 50) -> List[Message]:
        stmt = (select(Message).where(Message.session_id == session_id)
                .order_by(Message.id).limit(limit))
        return list(db.execute(stmt).scalars().all())

    @staticmethod
    def delete(db: Session, session_id: str) -> bool:
        s = SessionService.get(db, session_id)
        if s is None:
            return False
        db.execute(delete(Message).where(Message.session_id == session_id))
        db.delete(s)
        return True
