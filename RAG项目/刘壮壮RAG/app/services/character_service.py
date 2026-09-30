from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.character_templates import CHARACTER_TEMPLATES
from app.models.character import Character
from app.schemas.character import CharacterCreate, CharacterUpdate


async def create_character(
    db: AsyncSession, user_id: int, data: CharacterCreate
) -> Character:
    system_prompt = data.system_prompt
    if data.template_key and not data.system_prompt:
        for t in CHARACTER_TEMPLATES:
            if t["key"] == data.template_key:
                system_prompt = t["system_prompt"]
                break
    char = Character(
        user_id=user_id,
        name=data.name,
        system_prompt=system_prompt,
        model_name=data.model_name,
        base_url=data.base_url,
        temperature=data.temperature,
        top_p=data.top_p,
        max_tokens=data.max_tokens,
    )
    db.add(char)
    await db.commit()
    await db.refresh(char)
    return char


async def list_characters(db: AsyncSession, user_id: int) -> list[Character]:
    result = await db.execute(
        select(Character).where(Character.user_id == user_id).order_by(Character.id)
    )
    return list(result.scalars().all())


async def get_character(
    db: AsyncSession, user_id: int, character_id: int
) -> Character | None:
    result = await db.execute(
        select(Character).where(
            Character.id == character_id, Character.user_id == user_id
        )
    )
    return result.scalar_one_or_none()


async def update_character(
    db: AsyncSession, user_id: int, character_id: int, data: CharacterUpdate
) -> Character | None:
    char = await get_character(db, user_id, character_id)
    if char is None:
        return None
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(char, field, value)
    await db.commit()
    await db.refresh(char)
    return char


async def delete_character(db: AsyncSession, user_id: int, character_id: int) -> bool:
    char = await get_character(db, user_id, character_id)
    if char is None:
        return False
    await db.delete(char)
    await db.commit()
    return True
