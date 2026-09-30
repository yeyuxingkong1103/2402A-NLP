from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.models.chat import LongTermMemory, User
from app.schemas.memories import (
    LongTermMemoryCreateRequest,
    LongTermMemoryListResponse,
    LongTermMemoryResponse,
)
from app.security.auth import get_current_user

router = APIRouter(prefix="/api/v1/memories", tags=["memories"])


def serialize_memory(memory: LongTermMemory) -> LongTermMemoryResponse:
    return LongTermMemoryResponse(
        memory_id=memory.memory_id,
        memory_type=memory.memory_type,
        content=memory.content,
        is_confirmed=memory.is_confirmed,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
    )


@router.get("", response_model=LongTermMemoryListResponse)
def list_memories(
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> LongTermMemoryListResponse:
    memories = db.scalars(
        select(LongTermMemory)
        .where(
            LongTermMemory.user_id == current_user.user_id,
            LongTermMemory.is_active.is_(True),
            LongTermMemory.is_confirmed.is_(True),
        )
        .order_by(LongTermMemory.updated_at.desc())
    ).all()
    return LongTermMemoryListResponse(memories=[serialize_memory(item) for item in memories])


@router.post("", response_model=LongTermMemoryResponse, status_code=status.HTTP_201_CREATED)
def create_memory(
    request: LongTermMemoryCreateRequest,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> LongTermMemoryResponse:
    if not request.confirmed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="需要用户明确确认后才能保存长期记忆",
        )
    now = datetime.utcnow()
    memory = LongTermMemory(
        memory_id=uuid4().hex,
        user_id=current_user.user_id,
        memory_type=request.memory_type,
        content=request.content.strip(),
        is_confirmed=True,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    db.add(memory)
    db.commit()
    db.refresh(memory)
    return serialize_memory(memory)


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory(
    memory_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> None:
    memory = db.scalar(
        select(LongTermMemory).where(
            LongTermMemory.memory_id == memory_id,
            LongTermMemory.user_id == current_user.user_id,
            LongTermMemory.is_active.is_(True),
        )
    )
    if not memory:
        raise HTTPException(status_code=404, detail="长期记忆不存在")
    memory.is_active = False
    memory.updated_at = datetime.utcnow()
    db.commit()
