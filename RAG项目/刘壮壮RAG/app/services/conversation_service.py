from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation


async def create_conversation(
    db: AsyncSession, user_id: int, character_id: int, title: str = "新对话"
) -> Conversation:
    conv = Conversation(user_id=user_id, character_id=character_id, title=title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def list_conversations(
    db: AsyncSession, user_id: int, character_id: int
) -> list[Conversation]:
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.user_id == user_id,
            Conversation.character_id == character_id,
        )
        .order_by(Conversation.id.desc())
    )
    return list(result.scalars().all())


async def get_conversation(
    db: AsyncSession, user_id: int, character_id: int, conversation_id: int
) -> Conversation | None:
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
            Conversation.character_id == character_id,
        )
    )
    return result.scalar_one_or_none()


async def delete_conversation(
    db: AsyncSession, user_id: int, character_id: int, conversation_id: int
) -> bool:
    conv = await get_conversation(db, user_id, character_id, conversation_id)
    if conv is None:
        return False
    await db.delete(conv)
    await db.commit()
    return True
