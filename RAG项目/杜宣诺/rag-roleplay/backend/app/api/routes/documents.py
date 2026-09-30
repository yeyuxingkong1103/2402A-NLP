import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import get_settings
from ...db.base import get_db
from ...db.models import Document, User
from ...deps import get_milvus
from ..deps import get_current_user
from ..schemas import ok

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

ALLOWED_EXT = {"txt", "docx", "pdf"}

MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@router.post("/upload")
async def upload_document(file: UploadFile, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="仅支持 txt / docx / pdf 格式")

    settings = get_settings()
    # 流式读取并限制大小，避免一次性读入内存
    parts: list[bytes] = []
    size = 0
    while True:
        part = await file.read(1024 * 1024)
        if not part:
            break
        size += len(part)
        if size > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"文件超过大小上限 {settings.max_upload_bytes // 1024 // 1024}MB")
        parts.append(part)

    upload_dir = Path(settings.doc_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid.uuid4().hex}.{ext}"
    path.write_bytes(b"".join(parts))

    doc = Document(user_id=user.id, filename=filename, ext=ext, path=str(path), status="pending")
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # 入队 ARQ，异步解析 + 入库
    from arq import create_pool
    from ...worker.run import WorkerSettings
    redis = await create_pool(WorkerSettings.redis_settings)
    job = await redis.enqueue_job("ingest_document_task", doc.id)
    await redis.aclose()
    return ok({"id": doc.id, "job_id": job.job_id})


@router.get("/{document_id}/download")
async def download_document(document_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    path = Path(doc.path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文件已丢失")
    return FileResponse(path, filename=doc.filename, media_type=MEDIA_TYPES.get(doc.ext, "application/octet-stream"))


@router.get("")
async def list_documents(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    rows = await db.scalars(select(Document).order_by(Document.id.desc()))
    return ok([
        {
            "id": d.id, "filename": d.filename, "ext": d.ext, "status": d.status,
            "chunk_count": d.chunk_count, "error": d.error, "created_at": d.created_at.isoformat(),
        }
        for d in rows
    ])


@router.delete("/{document_id}")
async def delete_document(document_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, document_id)
    if doc is None or doc.user_id != user.id:
        raise HTTPException(status_code=404, detail="文档不存在")
    await get_milvus().delete_by_filter("documents", f"doc_id == {document_id}")
    try:
        Path(doc.path).unlink(missing_ok=True)
    except OSError:
        pass
    await db.delete(doc)
    await db.commit()
    return ok()
