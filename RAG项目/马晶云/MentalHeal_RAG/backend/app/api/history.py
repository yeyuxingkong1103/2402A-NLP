import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.models.chat import AIRole, ChatMessage, User
from app.schemas.history import (
    HistoryConversation,
    HistoryMessage,
    HistoryMessageUpdate,
    HistoryResponse,
)
from app.security.auth import get_current_user
from app.services.chat_history import ChatHistoryRepository

router = APIRouter(prefix="/api/v1", tags=["history"])


def parse_sources(value: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


@router.get("/history", response_model=HistoryResponse)
def list_history(
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> HistoryResponse:
    repository = ChatHistoryRepository(db)
    role_names = {
        role.role_id: role.name
        for role in db.scalars(select(AIRole)).all()
    }
    conversations = []
    for chat_session, messages in repository.list_conversations(current_user.user_id):
        conversations.append(
            HistoryConversation(
                session_id=chat_session.session_id,
                title=chat_session.title,
                role_id=chat_session.role_id,
                role_name=role_names.get(chat_session.role_id, "心理健康助手"),
                created_at=chat_session.created_at,
                updated_at=chat_session.updated_at,
                messages=[
                    HistoryMessage(
                        role=message.role,
                        content=message.content,
                        sources=parse_sources(message.citations_json),
                    )
                    for message in messages
                ],
            )
        )
    return HistoryResponse(conversations=conversations)


@router.patch("/history/{session_id}", response_model=HistoryConversation)
def rename_history(
    session_id: str,
    request: HistoryMessageUpdate,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> HistoryConversation:
    repository = ChatHistoryRepository(db)
    chat_session = repository.get_owned_session(session_id, current_user.user_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail="会话不存在")
    repository.rename_session(chat_session, request.title)
    repository.commit()
    return _serialize_conversation(db, repository, chat_session)


@router.delete("/history/{session_id}")
def delete_history(
    session_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    repository = ChatHistoryRepository(db)
    chat_session = repository.get_owned_session(session_id, current_user.user_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail="会话不存在")
    repository.delete_session(chat_session)
    repository.commit()
    return {"session_id": session_id, "message": "会话已删除"}


def _serialize_conversation(
    db: Session,
    repository: ChatHistoryRepository,
    chat_session: Any,
) -> HistoryConversation:
    role = db.scalar(select(AIRole).where(AIRole.role_id == chat_session.role_id))
    messages = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == chat_session.session_id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
    ).all()
    return HistoryConversation(
        session_id=chat_session.session_id,
        title=chat_session.title,
        role_id=chat_session.role_id,
        role_name=role.name if role else "心理健康助手",
        created_at=chat_session.created_at,
        updated_at=chat_session.updated_at,
        messages=[
            HistoryMessage(
                role=message.role,
                content=message.content,
                sources=parse_sources(message.citations_json),
            )
            for message in messages
        ],
    )
