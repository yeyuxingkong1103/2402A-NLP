from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.schemas.conversation import ConversationCreate, ConversationRead
from app.services import character_service, conversation_service

router = APIRouter()


@router.post(
    "/characters/{character_id}/conversations",
    response_model=ConversationRead,
    status_code=201,
)
async def create_conversation(
    character_id: int,
    payload: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    char = await character_service.get_character(db, current_user.id, character_id)
    if char is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    return await conversation_service.create_conversation(
        db, current_user.id, character_id, payload.title
    )


@router.get(
    "/characters/{character_id}/conversations",
    response_model=list[ConversationRead],
)
async def list_conversations(
    character_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await conversation_service.list_conversations(db, current_user.id, character_id)


@router.delete(
    "/characters/{character_id}/conversations/{conversation_id}", status_code=204
)
async def delete_conversation(
    character_id: int,
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ok = await conversation_service.delete_conversation(
        db, current_user.id, character_id, conversation_id
    )
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
