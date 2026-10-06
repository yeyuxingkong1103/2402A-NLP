import asyncio
import json
import time
from pathlib import Path

from celery import Celery
from sqlalchemy import select

from .config import get_settings
from .db import Document, SessionLocal
from .parsing import MinerUClient, parse_with_pymupdf, write_parsed
from .retrieval import RetrievalService

settings = get_settings()
celery_app = Celery("prospectus_rag", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(task_track_started=True, result_expires=86400)


def _progress(task, status: str, stage: str, progress: int, message: str = ""):
    task.update_state(state=status.upper(), meta={"stage": stage, "progress": progress, "message": message})


@celery_app.task(bind=True, autoretry_for=(ConnectionError,), retry_backoff=True, max_retries=2)
def ingest_document(self, document_id: str):
    db = SessionLocal()
    document = db.get(Document, document_id)
    if not document:
        return {"status": "failed", "message": "document not found"}
    path = Path(document.file_path)
    try:
        document.status = "parsing"
        db.commit()
        _progress(self, "started", "parsing", 10, "正在调用远程 MinerU")
        try:
            chunks, pages = asyncio.run(MinerUClient(settings).parse(path))
            parser = "mineru_remote"
            if not chunks:
                raise RuntimeError("MinerU returned no chunks")
        except Exception as error:
            chunks, pages = parse_with_pymupdf(path)
            parser = f"pymupdf_fallback:{type(error).__name__}"
        _progress(self, "started", "chunking", 40, f"解析器：{parser}")
        write_parsed(chunks, settings.parsed_dir / f"{document_id}.json")
        document.status = "indexing"
        document.pages = pages
        db.commit()
        _progress(self, "started", "embedding", 60, f"共 {len(chunks)} 个切片")
        retrieval = RetrievalService(settings)
        count = retrieval.index(document.id, document.display_name, document.sha256, chunks)
        document.status = "completed"
        document.chunks = count
        document.error = None
        db.commit()
        _progress(self, "success", "completed", 100, "知识库入库完成")
        return {"status": "completed", "chunks": count, "pages": pages}
    except Exception as error:
        document.status = "failed"
        document.error = str(error)
        db.commit()
        _progress(self, "failure", "failed", 100, str(error))
        raise
    finally:
        db.close()
