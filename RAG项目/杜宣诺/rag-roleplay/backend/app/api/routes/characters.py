import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.base import get_db
from ...db.models import Character, MemoryTask, Message, Session, User
from ...deps import get_milvus
from ...services.character_service import chunk_and_index, to_public
from ..deps import get_current_user
from ..schemas import CharacterIn, ok

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/characters", tags=["characters"])


async def _index_safe(character: Character) -> None:
    """重建角色设定向量；Milvus 不可用时不影响角色 CRUD（降级矩阵）。"""
    try:
        await chunk_and_index(get_milvus(), character)
    except Exception:
        logger.exception("角色设定索引失败 character_id=%s", character.id)


@router.post("")
async def create_character(body: CharacterIn, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    c = Character(owner_user_id=user.id, **body.model_dump())
    db.add(c)
    await db.commit()
    await db.refresh(c)
    await _index_safe(c)
    return ok(to_public(c, user.id))


@router.get("")
async def list_characters(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    rows = await db.scalars(
        select(Character).where(
            (Character.owner_user_id == user.id) | (Character.owner_user_id.is_(None))
        ).order_by(Character.id)
    )
    return ok([to_public(c, user.id) for c in rows])


@router.get("/{character_id}")
async def get_character(character_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    c = await db.get(Character, character_id)
    if c is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    return ok(to_public(c, user.id))


@router.put("/{character_id}")
async def update_character(character_id: int, body: CharacterIn, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    c = await db.get(Character, character_id)
    if c is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    if c.owner_user_id != user.id or c.is_preset:
        raise HTTPException(status_code=403, detail="无权限编辑该角色")
    for k, v in body.model_dump().items():
        setattr(c, k, v)
    await db.commit()
    await db.refresh(c)
    await _index_safe(c)
    return ok(to_public(c, user.id))


@router.delete("/{character_id}")
async def delete_character(character_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    c = await db.get(Character, character_id)
    if c is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    if c.owner_user_id != user.id or c.is_preset:
        raise HTTPException(status_code=403, detail="无权限删除该角色")

    # 显式级联清理：会话 → 消息 / 抽取任务 → 会话本身（不依赖外键级联）
    session_ids = (await db.scalars(select(Session.id).where(Session.character_id == c.id))).all()
    if session_ids:
        await db.execute(delete(MemoryTask).where(MemoryTask.session_id.in_(session_ids)))
        await db.execute(delete(Message).where(Message.session_id.in_(session_ids)))
        await db.execute(delete(Session).where(Session.character_id == c.id))
    await db.execute(delete(Character).where(Character.id == c.id))
    await db.commit()

    try:
        await get_milvus().delete_by_filter("character_settings", f"character_id == {c.id}")
    except Exception:
        logger.exception("角色设定向量删除失败 character_id=%s", c.id)
    return ok()
