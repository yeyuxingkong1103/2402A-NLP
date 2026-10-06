import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.api.knowledge import require_kb
from backend.app.core.database import get_db
from backend.app.core.security import get_current_user
from backend.app.models.entities import ChatMessage, ChatSession, User
from backend.app.models.schemas import ChatRequest, ChatResponse, ReferenceOut
from generation.deepseek_generator import LLMService
from memory.memory_service import MemoryService
from retrieval.hybrid_retriever import RetrievalService


router = APIRouter(prefix="/chat", tags=["问答"])


@router.get("/sessions")
def list_sessions(knowledge_base_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict]:
    require_kb(db, current_user.id, knowledge_base_id)
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id, ChatSession.knowledge_base_id == knowledge_base_id)
        .order_by(ChatSession.updated_at.desc())
        .limit(30)
        .all()
    )
    return [{"id": item.id, "title": item.title, "updated_at": item.updated_at.isoformat()} for item in sessions]


@router.get("/sessions/{session_id}/messages")
def list_messages(session_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict]:
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        return []
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.id.asc()).all()
    return [
        {"id": item.id, "role": item.role, "content": item.content, "references": json.loads(item.references_json or "[]")}
        for item in messages
    ]


@router.post("/ask", response_model=ChatResponse)
async def ask(payload: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> ChatResponse:
    require_kb(db, current_user.id, payload.knowledge_base_id)
    session = _get_or_create_session(db, current_user.id, payload.knowledge_base_id, payload.session_id, payload.question)
    retrieval_service = RetrievalService()
    references = retrieval_service.retrieve(db, current_user.id, payload.knowledge_base_id, payload.question)
    memory_service = MemoryService()
    short_memory = memory_service.get_short_messages(session.id)
    long_memory = memory_service.search_long_memory(current_user.id, payload.question)
    answer = await LLMService().answer(payload.question, references, short_memory, long_memory)
    reference_out = [_to_reference(index, item) for index, item in enumerate(references, start=1)]
    user_message = ChatMessage(session_id=session.id, user_id=current_user.id, role="user", content=payload.question)
    assistant_message = ChatMessage(
        session_id=session.id,
        user_id=current_user.id,
        role="assistant",
        content=answer,
        references_json=json.dumps([item.model_dump() for item in reference_out], ensure_ascii=False),
    )
    db.add(user_message)
    db.add(assistant_message)
    db.commit()
    memory_service.append_short_message(session.id, "user", payload.question)
    memory_service.append_short_message(session.id, "assistant", answer)
    memory_service.save_long_memory(current_user.id, f"用户曾询问：{payload.question}\n系统回答摘要：{answer[:300]}")
    return ChatResponse(session_id=session.id, answer=answer, references=reference_out)


def _get_or_create_session(db: Session, user_id: int, knowledge_base_id: int, session_id: int | None, question: str) -> ChatSession:
    if session_id:
        session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id).first()
        if session:
            return session
    title = question[:30] if question else "新会话"
    session = ChatSession(user_id=user_id, knowledge_base_id=knowledge_base_id, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _to_reference(index: int, item: dict) -> ReferenceOut:
    return ReferenceOut(
        index=index,
        document_id=item.get("document_id", 0),
        filename=item.get("filename", "未知文档"),
        chunk_uid=item.get("chunk_uid", ""),
        content=item.get("content", ""),
        score=float(item.get("score", 0.0)),
        page_number=int(item.get("page_number", 0) or 0),
        title_path=item.get("title_path", ""),
    )
