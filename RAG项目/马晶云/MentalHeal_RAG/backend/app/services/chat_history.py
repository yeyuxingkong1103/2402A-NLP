import json
from datetime import datetime
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.chat import AIRole, ChatMessage, ChatSession


class ChatHistoryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_or_create_session(
        self,
        session_id: str | None,
        title: str,
        user_id: str,
        role_id: str,
    ) -> ChatSession:
        if session_id:
            existing = self.session.scalar(
                select(ChatSession).where(ChatSession.session_id == session_id)
            )
            if existing:
                if existing.user_id != user_id or existing.status != "active":
                    raise PermissionError("无权访问该会话")
                if existing.role_id != role_id:
                    existing.role_id = role_id
                return existing
        now = datetime.utcnow()
        chat_session = ChatSession(
            session_id=session_id or str(uuid4()),
            user_id=user_id,
            title=title[:120],
            last_message=title[:300],
            role_id=role_id,
            status="active",
            created_at=now,
            updated_at=now,
        )
        self.session.add(chat_session)
        self.session.flush()
        return chat_session

    def get_owned_session(self, session_id: str, user_id: str) -> ChatSession | None:
        return self.session.scalar(
            select(ChatSession).where(
                ChatSession.session_id == session_id,
                ChatSession.user_id == user_id,
                ChatSession.status == "active",
            )
        )

    def add_message(
        self,
        chat_session: ChatSession,
        role: str,
        content: str,
        sources: list[dict] | None = None,
    ) -> ChatMessage:
        now = datetime.utcnow()
        message = ChatMessage(
            message_id=str(uuid4()),
            session_id=chat_session.session_id,
            user_id=chat_session.user_id,
            role=role,
            content=content,
            citations_json=json.dumps(sources or [], ensure_ascii=False),
            created_at=now,
        )
        chat_session.last_message = content[:300]
        chat_session.updated_at = now
        self.session.add(message)
        self.session.flush()
        return message

    def list_conversations(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[tuple[ChatSession, list[ChatMessage]]]:
        sessions = self.session.scalars(
            select(ChatSession)
            .where(ChatSession.user_id == user_id, ChatSession.status == "active")
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
        ).all()
        if not sessions:
            return []
        session_ids = [item.session_id for item in sessions]
        messages = self.session.scalars(
            select(ChatMessage)
            .where(
                ChatMessage.user_id == user_id,
                ChatMessage.session_id.in_(session_ids),
            )
            .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        ).all()
        grouped = {session_id: [] for session_id in session_ids}
        for message in messages:
            grouped.setdefault(message.session_id, []).append(message)
        return [(chat_session, grouped[chat_session.session_id]) for chat_session in sessions]

    def rename_session(self, chat_session: ChatSession, title: str) -> None:
        chat_session.title = title.strip()[:120]
        chat_session.updated_at = datetime.utcnow()

    def delete_session(self, chat_session: ChatSession) -> None:
        chat_session.status = "deleted"
        chat_session.updated_at = datetime.utcnow()
        self.session.execute(
            delete(ChatMessage).where(ChatMessage.session_id == chat_session.session_id)
        )

    def commit(self) -> None:
        self.session.commit()
