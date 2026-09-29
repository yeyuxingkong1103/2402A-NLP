from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import current_user
from app.database import get_db
from app.models import Role, User
from app.schemas import ChatRequest, ChatResponse
from app.chat_memory import append_turn, clear_memory, recent_messages
from app.rag_main import generate_answer, hybrid_search

router = APIRouter(prefix="/chat", tags=["对话"])


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    user: User = Depends(current_user),
    database: Session = Depends(get_db),
) -> ChatResponse:
    role = database.get(Role, payload.role_id)
    if not role or not role.is_public:
        raise HTTPException(status_code=404, detail="角色不存在")
    hits = hybrid_search(payload.message, role.id, user.id)
    history = recent_messages(user.id, role.id, payload.session_id)
    answer = generate_answer(payload.message, role.system_prompt, history, hits)
    append_turn(user.id, role.id, payload.session_id, payload.message, answer)
    return ChatResponse(answer=answer, sources=hits, session_id=payload.session_id)


@router.delete("/{role_id}/{session_id}/memory", status_code=204)
def delete_memory(
    role_id: int, session_id: str, user: User = Depends(current_user)
) -> None:
    clear_memory(user.id, role_id, session_id)
