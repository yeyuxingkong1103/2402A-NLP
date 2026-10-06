import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.config import get_settings
from app.core.character_templates import CHARACTER_TEMPLATES
from app.core.memory import ShortTermMemory
from app.core.redis_client import get_redis
from app.core.vector_store import VectorStore
from app.models.user import User
from app.schemas.character import CharacterCreate, CharacterRead, CharacterUpdate
from app.services import character_service, conversation_service

router = APIRouter()


@router.get("/characters/templates")
async def list_templates(current_user: User = Depends(get_current_user)):
    return [
        {
            "key": t["key"],
            "name": t["name"],
            "description": t["description"],
            "system_prompt": t["system_prompt"],
        }
        for t in CHARACTER_TEMPLATES
    ]


@router.post("/characters", response_model=CharacterRead, status_code=201)
async def create_character(
    payload: CharacterCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await character_service.create_character(db, current_user.id, payload)


@router.get("/characters", response_model=list[CharacterRead])
async def list_characters(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await character_service.list_characters(db, current_user.id)


@router.get("/characters/{character_id}", response_model=CharacterRead)
async def get_character(
    character_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    char = await character_service.get_character(db, current_user.id, character_id)
    if char is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    return char


@router.patch("/characters/{character_id}", response_model=CharacterRead)
async def update_character(
    character_id: int,
    payload: CharacterUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    char = await character_service.update_character(
        db, current_user.id, character_id, payload
    )
    if char is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    return char


@router.delete("/characters/{character_id}", status_code=204)
async def delete_character(
    character_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ok = await character_service.delete_character(db, current_user.id, character_id)
    if not ok:
        raise HTTPException(status_code=404, detail="角色不存在")

    # 级联清理：清空该角色的会话 Redis 记忆
    try:
        convs = await conversation_service.list_conversations(
            db, current_user.id, character_id
        )
        mem = ShortTermMemory(get_redis())
        for c in convs:
            await mem.clear(c.id)
    except Exception:
        pass

    # 级联清理：删除该角色的 Milvus 向量
    try:
        settings = get_settings()
        store = VectorStore(
            uri=settings.milvus_uri,
            text_dim=settings.text_embedding_dim,
            image_dim=settings.image_embedding_dim,
        )
        await asyncio.to_thread(store.delete_by_character, character_id)
    except Exception:
        pass
