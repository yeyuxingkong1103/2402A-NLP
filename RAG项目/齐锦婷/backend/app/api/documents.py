from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from backend.app.api.knowledge import require_kb
from backend.app.core.config import get_settings
from backend.app.core.database import get_db
from backend.app.core.security import get_current_user
from backend.app.models.entities import Document, IngestTask, User
from backend.app.models.schemas import DocumentOut, TaskOut


router = APIRouter(prefix="/documents", tags=["文档"])
settings = get_settings()


@router.get("", response_model=list[DocumentOut])
def list_documents(knowledge_base_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[Document]:
    require_kb(db, current_user.id, knowledge_base_id)
    return (
        db.query(Document)
        .filter(Document.user_id == current_user.id, Document.knowledge_base_id == knowledge_base_id)
        .order_by(Document.id.desc())
        .all()
    )


@router.post("/upload", response_model=TaskOut)
async def upload_document(
    background_tasks: BackgroundTasks,
    knowledge_base_id: int = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IngestTask:
    require_kb(db, current_user.id, knowledge_base_id)
    upload_dir = settings.data_root / "uploads" / str(current_user.id) / str(knowledge_base_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "document.pdf").name
    target_path = upload_dir / safe_name
    target_path.write_bytes(await file.read())
    document = Document(user_id=current_user.id, knowledge_base_id=knowledge_base_id, filename=safe_name, file_path=str(target_path))
    db.add(document)
    db.commit()
    db.refresh(document)
    task = IngestTask(document_id=document.id, user_id=current_user.id, knowledge_base_id=knowledge_base_id)
    db.add(task)
    db.commit()
    db.refresh(task)
    background_tasks.add_task(run_ingest_task, task.id)
    return task


async def run_ingest_task(task_id: int) -> None:
    from ingestion.ingest_pipeline import IngestService

    await IngestService().run(task_id)


@router.get("/tasks", response_model=list[TaskOut])
def list_tasks(knowledge_base_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[IngestTask]:
    require_kb(db, current_user.id, knowledge_base_id)
    return (
        db.query(IngestTask)
        .filter(IngestTask.user_id == current_user.id, IngestTask.knowledge_base_id == knowledge_base_id)
        .order_by(IngestTask.id.desc())
        .limit(20)
        .all()
    )
