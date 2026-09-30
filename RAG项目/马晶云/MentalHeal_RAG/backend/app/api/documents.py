from datetime import datetime
from hashlib import sha256
from pathlib import Path
from threading import Thread

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db_session
from app.models.chat import Document, IngestJob, User
from app.schemas.documents import (
    DocumentJobResponse,
    DocumentListResponse,
    DocumentResponse,
    IngestJobListResponse,
    IngestJobResponse,
)
from app.security.auth import get_current_user
from app.services.document_ingestion import (
    create_ingest_job,
    embed_document,
    get_ingestion_config,
    has_active_job,
    process_document,
    register_document,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def require_admin(user: User) -> None:
    if user.account_role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")


def serialize_document(document: Document) -> DocumentResponse:
    return DocumentResponse(
        document_id=document.document_id,
        source_file=document.source_file,
        page_count=document.page_count,
        total_text_chars=document.total_text_chars,
        is_enabled=document.is_enabled,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def serialize_job(job: IngestJob) -> IngestJobResponse:
    return IngestJobResponse(
        job_id=job.id,
        job_name=job.job_name,
        status=job.status,
        document_count=job.document_count,
        page_count=job.page_count,
        chunk_count=job.chunk_count,
        note=job.note,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> DocumentResponse:
    require_admin(current_user)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")
    settings = get_settings()
    max_bytes = settings.upload_max_size_mb * 1024 * 1024
    target_dir = settings.raw_data_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    target = target_dir / safe_name
    content = file.file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="文件超过上传大小限制")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="文件内容不是有效的 PDF")
    if target.exists():
        existing_content = target.read_bytes()
        if existing_content == content:
            document = register_document(target, settings)
            return serialize_document(document)
        digest = sha256(content).hexdigest()[:12]
        target = target_dir / f"{Path(safe_name).stem}-{digest}{Path(safe_name).suffix.lower()}"
        if target.exists() and target.read_bytes() != content:
            raise HTTPException(status_code=409, detail="同名文件已存在且内容冲突")
    if not target.exists():
        target.write_bytes(content)
    document = register_document(target, settings)
    return serialize_document(document)


@router.get("", response_model=DocumentListResponse)
def list_documents(
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> DocumentListResponse:
    require_admin(current_user)
    documents = db.scalars(select(Document).order_by(Document.updated_at.desc())).all()
    return DocumentListResponse(documents=[serialize_document(item) for item in documents])


@router.post("/{document_id}/parse", response_model=DocumentJobResponse, status_code=status.HTTP_202_ACCEPTED)
def parse_document(
    document_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> DocumentJobResponse:
    require_admin(current_user)
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    if has_active_job(document_id, "parse"):
        raise HTTPException(status_code=409, detail="该文档已有解析任务正在执行")
    job_id = create_ingest_job(document_id, "parse", get_ingestion_config(validate=False))
    thread = Thread(
        target=_run_document_task,
        args=(process_document, document_id, job_id),
        daemon=True,
    )
    thread.start()
    return DocumentJobResponse(
        document_id=document_id,
        job_id=job_id,
        status="queued",
        message="文档解析任务已启动",
    )


@router.post("/{document_id}/embed", response_model=DocumentJobResponse, status_code=status.HTTP_202_ACCEPTED)
def embed_document_route(
    document_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> DocumentJobResponse:
    require_admin(current_user)
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    if has_active_job(document_id, "embed"):
        raise HTTPException(status_code=409, detail="该文档已有向量化任务正在执行")
    job_id = create_ingest_job(document_id, "embed", get_ingestion_config(validate=False))
    thread = Thread(
        target=_run_document_task,
        args=(embed_document, document_id, job_id),
        daemon=True,
    )
    thread.start()
    return DocumentJobResponse(
        document_id=document_id,
        job_id=job_id,
        status="queued",
        message="文档向量化任务已启动",
    )


@router.get("/jobs", response_model=IngestJobListResponse)
def list_ingest_jobs(
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> IngestJobListResponse:
    require_admin(current_user)
    jobs = db.scalars(select(IngestJob).order_by(IngestJob.created_at.desc()).limit(50)).all()
    return IngestJobListResponse(jobs=[serialize_job(item) for item in jobs])


def _run_document_task(task, document_id: str, job_id: int) -> None:
    try:
        task(document_id, job_id)
    except Exception:
        return
