from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_file import KnowledgeFile


async def create_file(
    db: AsyncSession,
    user_id: int,
    character_id: int,
    filename: str,
    file_type: str,
    file_path: str,
) -> KnowledgeFile:
    f = KnowledgeFile(
        user_id=user_id,
        character_id=character_id,
        filename=filename,
        file_type=file_type,
        file_path=file_path,
        status="pending",
    )
    db.add(f)
    await db.commit()
    await db.refresh(f)
    return f


async def list_files(
    db: AsyncSession, user_id: int, character_id: int
) -> list[KnowledgeFile]:
    result = await db.execute(
        select(KnowledgeFile)
        .where(
            KnowledgeFile.user_id == user_id,
            KnowledgeFile.character_id == character_id,
        )
        .order_by(KnowledgeFile.id)
    )
    return list(result.scalars().all())


async def get_file(
    db: AsyncSession, user_id: int, character_id: int, file_id: int
) -> KnowledgeFile | None:
    result = await db.execute(
        select(KnowledgeFile).where(
            KnowledgeFile.id == file_id,
            KnowledgeFile.user_id == user_id,
            KnowledgeFile.character_id == character_id,
        )
    )
    return result.scalar_one_or_none()


async def delete_file(
    db: AsyncSession, user_id: int, character_id: int, file_id: int
) -> KnowledgeFile | None:
    f = await get_file(db, user_id, character_id, file_id)
    if f is None:
        return None
    Path(f.file_path).unlink(missing_ok=True)
    await db.delete(f)
    await db.commit()
    return f
