# -*- coding: utf-8 -*-
"""知识库路由：入库（后台任务）/ 文档列表 / 删除 / 集合统计。

⚠️ Qdrant 嵌入式模式持有独占文件锁，入库必须走本服务的接口，
   不要在服务运行时另起脚本打开同一个 qdrant_storage 目录。
"""
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..core import config
from ..core.db import get_db
from ..core.logging import get_logger
from ..deps import get_current_user
from ..services import ingest as ingest_svc
from .. import models, schemas

router = APIRouter(prefix="/kb", tags=["知识库"])
log = get_logger("kb")

# 简易任务登记表（进程内，够用即可）
JOBS: dict[str, dict] = {}


def _run_ingest_job(job_id: str, path: str, collection: str,
                    strategy: str, ocr: str = "auto") -> None:
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["started_at"] = time.time()
    try:
        result = ingest_svc.ingest_path(path, collection, strategy, ocr=ocr)
        JOBS[job_id]["status"] = "done" if result.get("ok") else "failed"
        JOBS[job_id]["result"] = result
    except Exception as e:
        log.exception("入库任务异常")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        JOBS[job_id]["finished_at"] = time.time()


@router.post("/ingest", summary="提交入库（后台执行，立即返回 job_id）")
def submit_ingest(body: schemas.IngestIn, background: BackgroundTasks,
                  _user: models.User = Depends(get_current_user)):
    import os
    if not os.path.exists(body.path):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"路径不存在: {body.path}")

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "job_id": job_id,
        "path": body.path,
        "collection": body.collection,
        "strategy": body.strategy,
        "status": "pending",
    }
    background.add_task(_run_ingest_job, job_id, body.path, body.collection,
                        body.strategy, body.ocr)
    return {"job_id": job_id, "status": "pending",
            "hint": f"轮询 GET /api/kb/jobs/{job_id} 查看进度"}


@router.get("/jobs/{job_id}", summary="查询入库任务进度")
def get_job(job_id: str, _user: models.User = Depends(get_current_user)):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")
    return job


@router.get("/jobs", summary="所有入库任务")
def list_jobs(_user: models.User = Depends(get_current_user)):
    return list(JOBS.values())


@router.get("/documents", response_model=list[schemas.KbDocumentOut], summary="文档列表")
def list_documents(collection: str | None = None, limit: int = 300,
                   db: Session = Depends(get_db),
                   _user: models.User = Depends(get_current_user)):
    q = db.query(models.KbDocument)
    if collection:
        q = q.filter_by(collection=collection)
    return q.order_by(models.KbDocument.id.desc()).limit(limit).all()


@router.delete("/documents/{doc_id}", summary="删除文档（连带删向量）")
def delete_document(doc_id: int, _user: models.User = Depends(get_current_user)):
    result = ingest_svc.delete_document(doc_id)
    if not result.get("ok"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, result.get("error", "删除失败"))
    return result


@router.get("/collections", response_model=list[schemas.CollectionOut], summary="集合统计")
def list_collections(db: Session = Depends(get_db),
                     _user: models.User = Depends(get_current_user)):
    stats = ingest_svc.collection_stats()
    # 标注每个集合被哪些角色使用
    bindings: dict[str, list[str]] = {}
    for ch in db.query(models.Character).all():
        bindings.setdefault(ch.kb_collection, []).append(ch.name)

    return [
        schemas.CollectionOut(
            name=s["name"], points=s["points"],
            characters=bindings.get(s["name"], []),
        )
        for s in stats
    ]


@router.post("/collections", summary="创建空集合")
def create_collection(name: str, _user: models.User = Depends(get_current_user)):
    ingest_svc.ensure_collection(name)
    return {"ok": True, "collection": name}
