from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.file_processing import classify_file_type
from app.models.user import User
from app.schemas.knowledge import KnowledgeFileRead
from app.services import character_service, knowledge_service

router = APIRouter()

UPLOAD_ROOT = Path("data/uploads")
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB


@router.post(
    "/characters/{character_id}/knowledge",
    response_model=KnowledgeFileRead,
    status_code=201,
)
async def upload_file(
    character_id: int,
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    char = await character_service.get_character(db, current_user.id, character_id)
    if char is None:
        raise HTTPException(status_code=404, detail="角色不存在")

    try:
        file_type = classify_file_type(file.filename or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="文件超过 100MB 限制")

    record = await knowledge_service.create_file(
        db, current_user.id, character_id, file.filename or "unknown", file_type, ""
    )

    dir_path = UPLOAD_ROOT / str(current_user.id) / str(character_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{record.id}_{file.filename}"
    file_path.write_bytes(content)

    record.file_path = str(file_path)
    await db.commit()
    await db.refresh(record)
    return record


@router.get(
    "/characters/{character_id}/knowledge",
    response_model=list[KnowledgeFileRead],
)
async def list_files(
    character_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await knowledge_service.list_files(db, current_user.id, character_id)


@router.delete(
    "/characters/{character_id}/knowledge/{file_id}", status_code=204
)
async def delete_file(
    character_id: int,
    file_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    f = await knowledge_service.delete_file(db, current_user.id, character_id, file_id)
    if f is None:
        raise HTTPException(status_code=404, detail="文件不存在")
