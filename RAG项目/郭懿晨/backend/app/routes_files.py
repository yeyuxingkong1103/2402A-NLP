from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.files import pdf_file_sha256, pdf_sha256, save_pdf_bytes
from backend.app.models import BuildTask, Document
from backend.app.storage import JsonStateStore


router = APIRouter(prefix="/api/files", tags=["files"])


def get_file_dependencies() -> tuple[Path, JsonStateStore]:
    """提供上传接口依赖。"""
    data_dir = Path("data")
    store = JsonStateStore(Path("data/state.json"))
    return data_dir, store


def find_duplicate_pdf_document(
    documents: list[Document], content: bytes
) -> Document | None:
    """在已有文档中查找内容完全相同（sha256）的 PDF。"""
    content_hash = pdf_sha256(content)
    for document in documents:
        if document.content_hash:
            if document.content_hash == content_hash:
                return document
            continue
        if pdf_file_sha256(Path(document.file_path)) == content_hash:
            return document
    return None


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, str]:
    """上传 PDF 并创建构建任务；相同内容的 PDF 拒绝重复入库。"""
    data_dir, store = get_file_dependencies()
    try:
        content = await file.read()
    except OSError as exc:
        raise HTTPException(status_code=400, detail="读取上传文件失败") from exc

    duplicate = find_duplicate_pdf_document(store.list_documents(), content)
    if duplicate is not None:
        raise HTTPException(
            status_code=409,
            detail=f"档案库已存在相同内容的 PDF（{duplicate.file_name}），未重复入库",
        )

    try:
        saved_path = save_pdf_bytes(data_dir, file.filename or "upload.pdf", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document = Document.new(
        file_name=file.filename or saved_path.name,
        file_path=str(saved_path),
        content_hash=pdf_sha256(content),
    )
    task = BuildTask.new(document.document_id)
    store.save_document(document)
    store.save_task(task)

    return {"document_id": document.document_id, "task_id": task.task_id}
