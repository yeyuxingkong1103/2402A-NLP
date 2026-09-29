"""知识库 API：上传、两阶段索引、列表、删除、重建和独立检索。"""

import json
import logging
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import admin_user, current_user
from app.config import get_settings
from app.database import SessionLocal, get_db
from app.models import Document, DocumentCatalog, Role, User
from app.schemas import CatalogItem, DocumentResponse, SearchHit, SearchRequest
from app.rag_main import build_index, delete_document, hybrid_search

router = APIRouter(prefix="/knowledge", tags=["知识库"])
settings = get_settings()
logger = logging.getLogger(__name__)
_visual_index_lock = threading.Lock()
_active_index_lock = threading.Lock()
_active_index_ids: set[int] = set()


def _document_or_404(document_id: int, database: Session) -> Document:
    document = database.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    return document


def _index_in_background(document_id: int) -> None:
    """第一阶段快速索引，成功后启动不阻塞检索的视觉增强。"""
    try:
        with SessionLocal() as database:
            document = database.get(Document, document_id)
            if not document:
                return
            try:
                build_index(document, database, use_visual=False)
            except Exception:
                logger.exception("文档 %s 基础索引失败", document_id)
                return
        if settings.paddleocr_vl_enabled:
            threading.Thread(
                target=_enhance_in_background, args=(document_id,), daemon=True,
                name=f"visual-index-{document_id}",
            ).start()
    finally:
        with _active_index_lock:
            _active_index_ids.discard(document_id)


def schedule_index(document_id: int) -> bool:
    """提交单个基础索引任务，并避免同一文档被重复处理。"""
    with _active_index_lock:
        if document_id in _active_index_ids:
            return False
        _active_index_ids.add(document_id)
    threading.Thread(
        target=_index_in_background, args=(document_id,), daemon=True,
        name=f"base-index-{document_id}",
    ).start()
    return True


def recover_processing_documents() -> int:
    """应用重启后重新提交未完成任务，防止文档永久卡在处理中。"""
    with SessionLocal() as database:
        document_ids = list(database.scalars(
            select(Document.id).where(Document.status == "processing")
        ))
    recovered = sum(schedule_index(document_id) for document_id in document_ids)
    if recovered:
        logger.warning("已恢复 %s 个中断的文档索引任务", recovered)
    return recovered


def _enhance_in_background(document_id: int) -> None:
    """第二阶段串行执行 PaddleOCR-VL；失败时保留基础索引。"""
    with _visual_index_lock, SessionLocal() as database:
        document = database.get(Document, document_id)
        if not document or document.status != "ready":
            return
        try:
            build_index(document, database, use_visual=True, mark_processing=False)
        except Exception:
            logger.exception("文档 %s 视觉增强失败，保留基础索引", document_id)


@router.get("/catalog", response_model=list[CatalogItem])
def public_catalog(database: Session = Depends(get_db)) -> list[dict]:
    statement = (
        select(DocumentCatalog, Document)
        .join(Document, Document.id == DocumentCatalog.document_id)
        .where(Document.status == "ready", Document.is_public.is_(True))
        .order_by(DocumentCatalog.category, DocumentCatalog.title)
    )
    items = []
    for catalog, document in database.execute(statement).all():
        stored = Path(document.stored_path)
        bundled = Path("output/pdf") / document.filename
        if not stored.exists() and not bundled.exists():
            continue
        items.append({
            "document_id": document.id,
            "category": catalog.category,
            "title": catalog.title,
            "summary": catalog.summary,
            "questions": json.loads(catalog.questions_json),
            "filename": document.filename,
        })
    return items


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(
    _: User = Depends(admin_user), database: Session = Depends(get_db)
) -> list[Document]:
    statement = select(Document).order_by(Document.created_at.desc())
    return list(database.scalars(statement))


@router.post("/documents", response_model=DocumentResponse, status_code=201)
def upload_document(
    role_id: int = Form(...),
    source_url: str = Form(default=""),
    file: UploadFile = File(...),
    user: User = Depends(admin_user),
    database: Session = Depends(get_db),
) -> Document:
    if not database.get(Role, role_id):
        raise HTTPException(status_code=404, detail="角色不存在")
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="首版只支持 PDF")
    safe_name = f"{uuid.uuid4().hex}.pdf"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = settings.upload_dir / safe_name
    with stored_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    if stored_path.stat().st_size > settings.max_upload_mb * 1024 * 1024:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="文件过大")
    if stored_path.read_bytes()[:5] != b"%PDF-":
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=415, detail="文件内容不是 PDF")
    document = Document(
        owner_id=user.id,
        role_id=role_id,
        filename=Path(file.filename).name[:255],
        stored_path=str(stored_path.resolve()),
        source_url=source_url[:1000],
        is_public=True,
    )
    database.add(document)
    database.commit()
    database.refresh(document)
    schedule_index(document.id)
    return document


@router.post("/documents/{document_id}/reindex", response_model=DocumentResponse)
def reindex_document(
    document_id: int,
    _: User = Depends(admin_user),
    database: Session = Depends(get_db),
) -> Document:
    document = _document_or_404(document_id, database)
    document.status, document.error_message = "processing", ""
    database.commit()
    schedule_index(document.id)
    return document


@router.delete("/documents/{document_id}", status_code=204)
def remove_document(
    document_id: int,
    _: User = Depends(admin_user),
    database: Session = Depends(get_db),
) -> None:
    document = _document_or_404(document_id, database)
    delete_document(document.id)
    Path(document.stored_path).unlink(missing_ok=True)
    database.delete(document)
    database.commit()


@router.post("/search", response_model=list[SearchHit])
def search_knowledge(payload: SearchRequest, user: User = Depends(current_user)) -> list[dict]:
    return hybrid_search(payload.query, payload.role_id, user.id, payload.top_k)
