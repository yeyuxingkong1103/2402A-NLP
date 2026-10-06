import json
import shutil
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import Document, Feedback, QueryLog, get_db, init_db
from .llm import DeepSeekClient
from .parsing import sha256_file
from .retrieval import RetrievalService
from .schemas import AskRequest, AskResponse, DocumentOut, FeedbackRequest, TaskOut
from .tasks import ingest_document

settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def require_admin(x_admin_token: str | None = Header(default=None)):
    if x_admin_token != settings.admin_token:
        raise HTTPException(status_code=401, detail="管理员口令无效")


def enqueue_document(document: Document, db: Session) -> None:
    task = ingest_document.delay(document.id)
    document.error = json.dumps({"task_id": task.id}, ensure_ascii=False)
    db.commit()


@app.on_event("startup")
def startup():
    try:
        init_db()
        default_pdf = Path("/data/default/招股说明书1-无水印.pdf")
        if default_pdf.exists():
            db = next(get_db())
            digest = sha256_file(default_pdf)
            existing = db.scalar(select(Document).where(Document.sha256 == digest))
            if not existing:
                target = settings.upload_dir / default_pdf.name
                if not target.exists():
                    shutil.copyfile(default_pdf, target)
                document = Document(display_name="招股说明书1.pdf", original_name=default_pdf.name, sha256=digest, file_path=str(target), status="pending")
                db.add(document)
                db.commit()
                db.refresh(document)
                enqueue_document(document, db)
            elif existing.status in {"failed", "pending", "indexing"}:
                existing.status = "pending"
                existing.error = None
                db.commit()
                enqueue_document(existing, db)
            db.close()
    except Exception:
        pass


@app.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@app.get("/ready")
def ready(db: Session = Depends(get_db)):
    try:
        db.execute(select(Document).limit(1))
        return {"status": "ready"}
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error))


@app.get("/api/documents", response_model=list[DocumentOut])
def documents(db: Session = Depends(get_db)):
    return list(db.scalars(select(Document).order_by(Document.created_at.desc())))


@app.post("/api/documents/upload", response_model=DocumentOut, dependencies=[Depends(require_admin)])
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")
    safe_name = Path(file.filename).name
    target = settings.upload_dir / f"{uuid.uuid4()}-{safe_name}"
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"文件不能超过 {settings.max_upload_mb} MB")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="文件头不是有效 PDF")
    target.write_bytes(content)
    digest = sha256_file(target)
    existing = db.scalar(select(Document).where(Document.sha256 == digest))
    if existing:
        target.unlink(missing_ok=True)
        return existing
    document = Document(display_name="招股说明书1.pdf" if "招股说明书" in safe_name else safe_name, original_name=safe_name, sha256=digest, file_path=str(target), status="pending")
    db.add(document)
    db.commit()
    db.refresh(document)
    enqueue_document(document, db)
    return document


@app.get("/api/documents/{document_id}/task", response_model=TaskOut)
def document_task(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    task_id = ""
    try:
        task_id = json.loads(document.error or "{}").get("task_id", "")
    except json.JSONDecodeError:
        pass
    result = ingest_document.AsyncResult(task_id) if task_id else None
    meta = result.info if result and isinstance(result.info, dict) else {}
    return TaskOut(task_id=task_id, document_id=document_id, status=document.status, stage=meta.get("stage", document.status), progress=meta.get("progress", 100 if document.status == "completed" else 0), message=meta.get("message", document.error if document.status == "failed" else None))


@app.delete("/api/documents/{document_id}", dependencies=[Depends(require_admin)])
def delete_document(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    try:
        RetrievalService(settings).store.delete_document(document_id)
    except Exception:
        pass
    Path(document.file_path).unlink(missing_ok=True)
    (settings.parsed_dir / f"{document_id}.json").unlink(missing_ok=True)
    db.delete(document)
    db.commit()
    return {"deleted": document_id}


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest, db: Session = Depends(get_db)):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    plan_started = time.perf_counter()
    llm = DeepSeekClient(settings)
    plan = await llm.plan(request.question)
    retrieval = RetrievalService(settings)
    query = " ".join(plan.search_queries) if plan.search_queries else request.question
    hits = retrieval.search(query, request.document_id)
    retrieval_ms = (time.perf_counter() - plan_started) * 1000
    language = request.language if request.language != "auto" else plan.language
    answer, refusal = await llm.answer(request.question, hits, language)
    total_ms = (time.perf_counter() - started) * 1000
    citations = [{"chunk_id": hit.chunk_id, "page_number": hit.page_number, "section_title": hit.section_title, "content": hit.content, "score": hit.score, "document_name": hit.document_name} for hit in hits]
    response = AskResponse(request_id=request_id, answer=answer, language=language, citations=citations, confidence="insufficient" if refusal else ("high" if hits else "low"), refusal=refusal, latency={"retrieval_ms": round(retrieval_ms, 2), "total_ms": round(total_ms, 2)}, model=settings.deepseek_model, document_id=request.document_id)
    db.add(QueryLog(request_id=request_id, question=request.question, answer=answer, document_id=request.document_id, response_json=response.model_dump_json()))
    db.commit()
    return response


@app.post("/api/ask/stream")
async def ask_stream(request: AskRequest, db: Session = Depends(get_db)):
    response = await ask(request.model_copy(update={"stream": False}), db)

    async def event_stream():
        yield f"event: answer_delta\ndata: {json.dumps({'text': response.answer}, ensure_ascii=False)}\n\n"
        yield f"event: complete\ndata: {response.model_dump_json()}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/feedback/{request_id}")
def feedback(request_id: str, payload: FeedbackRequest, db: Session = Depends(get_db)):
    db.add(Feedback(request_id=request_id, helpful=1 if payload.helpful else 0, reason=payload.reason, note=payload.note))
    db.commit()
    return {"saved": True}
