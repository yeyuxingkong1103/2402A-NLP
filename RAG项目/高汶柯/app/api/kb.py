"""知识库接口：上传入库、批量入库、文档列表、删除。 知识库管理"""
from __future__ import annotations

import os
import shutil

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.db.mysql_store import list_kb_docs
from app.ingest.pipeline import delete_document, ingest_directory, ingest_file
from app.logging_conf import log

router = APIRouter(tags=["knowledge"])


@router.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    domain: str = Form("general"),
    parser: str = Form("auto"),
) -> dict:
    os.makedirs(settings.laws_dir, exist_ok=True)
    path = os.path.join(settings.laws_dir, file.filename)
    with open(path, "wb") as fp:
        shutil.copyfileobj(file.file, fp)
    try:
        result = ingest_file(path, domain=domain, parser=parser)
    except Exception as exc:  # noqa: BLE001
        log.error("上传入库失败: %s", exc)
        raise HTTPException(500, str(exc)) from exc
    return result


@router.post("/kb/ingest_dir")
def kb_ingest_dir(dir_path: str | None = Form(None), domain: str = Form("general"),
                  parser: str = Form("auto")) -> dict:
    target = dir_path or settings.laws_dir
    results = ingest_directory(target, domain=domain, parser=parser)
    return {"ok": True, "count": len(results), "results": results}


@router.get("/kb/docs")
def kb_docs(domain: str | None = None) -> list[dict]:
    return list_kb_docs(domain)


@router.delete("/kb/docs/{doc_id}")
def kb_delete(doc_id: str) -> dict:
    try:
        delete_document(doc_id)
    except Exception as exc:  # noqa: BLE001
        log.error("删除文档失败: %s", exc)
        raise HTTPException(500, str(exc)) from exc
    return {"ok": True, "doc_id": doc_id}
