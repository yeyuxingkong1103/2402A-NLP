from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.base import get_db
from ...db.models import Character, Favorite, Message, Session, User
from ..deps import get_current_user
from ..schemas import ok

router = APIRouter(prefix="/api/v1/favorites", tags=["favorites"])


class FavoriteIn(BaseModel):
    message_id: int


@router.post("")
async def create_favorite(body: FavoriteIn, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    msg = await db.get(Message, body.message_id)
    if msg is None or msg.role != "assistant":
        raise HTTPException(status_code=404, detail="回答不存在")
    sess = await db.get(Session, msg.session_id)
    if sess is None or sess.user_id != user.id:
        raise HTTPException(status_code=404, detail="回答不存在")

    # 幂等：已收藏则直接返回
    existing = await db.scalar(
        select(Favorite).where(Favorite.user_id == user.id, Favorite.message_id == body.message_id)
    )
    if existing is not None:
        return ok({"id": existing.id})

    fav = Favorite(user_id=user.id, message_id=body.message_id)
    db.add(fav)
    await db.commit()
    await db.refresh(fav)
    return ok({"id": fav.id})


@router.get("")
async def list_favorites(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    rows = await db.execute(
        select(Favorite, Message, Session, Character)
        .join(Message, Message.id == Favorite.message_id)
        .join(Session, Session.id == Message.session_id)
        .join(Character, Character.id == Session.character_id)
        .where(Favorite.user_id == user.id)
        .order_by(Favorite.id.desc())
    )
    out = []
    for fav, msg, sess, char in rows.all():
        out.append({
            "id": fav.id,
            "message_id": msg.id,
            "session_id": sess.id,
            "character_id": char.id,
            "character_name": char.name,
            "content": msg.content,
            "created_at": fav.created_at.isoformat(),
        })
    return ok(out)


@router.delete("/{favorite_id}")
async def delete_favorite(favorite_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    fav = await db.get(Favorite, favorite_id)
    if fav is None or fav.user_id != user.id:
        raise HTTPException(status_code=404, detail="收藏不存在")
    await db.delete(fav)
    await db.commit()
    return ok()
