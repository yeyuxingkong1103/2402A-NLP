import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.base import get_db
from ...db.models import Character, Favorite, Message, Session, User
from ..deps import get_current_user
from ..schemas import ok

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


class SessionIn(BaseModel):
    character_id: int


@router.get("")
async def list_sessions(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    rows = await db.scalars(select(Session).where(Session.user_id == user.id).order_by(Session.updated_at.desc()))
    return ok([{"id": s.id, "character_id": s.character_id, "title": s.title} for s in rows])


@router.post("")
async def create_session(body: SessionIn, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    character = await db.get(Character, body.character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    s = Session(user_id=user.id, character_id=character.id, title=character.name)
    db.add(s)
    await db.flush()
    if character.greeting:
        db.add(Message(session_id=s.id, role="assistant", content=character.greeting))
    await db.commit()
    await db.refresh(s)
    return ok({"id": s.id, "character_id": s.character_id, "title": s.title})


@router.get("/{session_id}/messages")
async def list_messages(session_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    s = await db.get(Session, session_id)
    if s is None or s.user_id != user.id:
        raise HTTPException(status_code=404, detail="会话不存在")
    rows = await db.scalars(select(Message).where(Message.session_id == session_id).order_by(Message.id))
    fav_msg_ids = set(await db.scalars(select(Favorite.message_id).where(Favorite.user_id == user.id)))
    out = []
    for m in rows:
        try:
            sources = json.loads(m.sources) if m.sources else []
        except (ValueError, TypeError):
            sources = []
        out.append({
            "id": m.id, "role": m.role, "content": m.content, "sources": sources,
            "is_favorite": m.id in fav_msg_ids,
        })
    return ok(out)


@router.delete("/{session_id}")
async def delete_session(session_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    s = await db.get(Session, session_id)
    if s is None or s.user_id != user.id:
        raise HTTPException(status_code=404, detail="会话不存在")
    await db.delete(s)
    await db.commit()
    return ok()
