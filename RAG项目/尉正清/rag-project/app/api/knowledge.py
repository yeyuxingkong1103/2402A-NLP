# app/api/knowledge.py
"""知识库管理接口：入库（新增）与更新（增量/删除）。

入库与更新都是对同一份 Milvus 知识的写操作，共用角色校验与错误处理，
因此合并在一个模块里。
"""
import os
import uuid

from fastapi import (APIRouter, Depends, File, Form, HTTPException, UploadFile)

from sqlalchemy.orm import Session

from app.config import settings
from app.core.ingest_service import get_ingest_service
from app.core.role_service import RoleService
from app.core.update_service import get_update_service
from app.db import get_db
from app.db.milvus_conn import expr_eq
from app.schemas import (DeleteDocumentRequest, IngestDatasetRequest, Resp,
                         UpdateDocumentRequest)
import logging

logger = logging.getLogger(__name__)

ingest_router = APIRouter(prefix="/api/ingest", tags=["知识入库"])
update_router = APIRouter(prefix="/api/update", tags=["知识更新"])

ALLOWED_EXT = {".pdf", ".txt", ".md"}


def _check_role(db: Session, role_key: str) -> None:
    if RoleService.get_by_key(db, role_key) is None:
        raise HTTPException(status_code=400, detail="角色不存在: %s" % role_key)


# ==================== 入库 ====================
@ingest_router.post("/dataset", response_model=Resp, summary="按角色批量导入内置数据集")
async def ingest_dataset(req: IngestDatasetRequest, db: Session = Depends(get_db)):
    """把 data/{role_key}/ 下的全部 JSONL 灌入 Milvus。"""
    _check_role(db, req.role_key)
    try:
        if req.force:
            get_ingest_service().milvus.delete_by_expr(
                settings.MILVUS_COLLECTION, expr_eq(role_id=req.role_key))
            logger.warning("已清空角色 [%s] 的旧知识", req.role_key)
        result = get_ingest_service().ingest_role_dir(req.role_key)
        return Resp(msg="入库完成", data=result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("数据集入库失败")
        raise HTTPException(status_code=500, detail="入库失败: %s" % e)


@ingest_router.post("/upload", response_model=Resp,
                    summary="上传文档入库（支持去水印与图表 OCR）")
async def upload_document(
    file: UploadFile = File(...),
    role_key: str = Form(...),
    remove_watermark: bool = Form(True),
    use_ocr: bool = Form(True),
    use_tables: bool = Form(True),
    db: Session = Depends(get_db),
):
    _check_role(db, role_key)

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400,
                            detail="仅支持 %s 格式" % "/".join(sorted(ALLOWED_EXT)))

    saved = os.path.join(settings.UPLOADS_DIR, "%s%s" % (uuid.uuid4().hex, ext))
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")
    with open(saved, "wb") as f:
        f.write(content)

    try:
        ingest = get_ingest_service()
        if ext == ".pdf":
            result = ingest.ingest_pdf(saved, role_key,
                                       remove_watermark=remove_watermark,
                                       use_ocr=use_ocr, use_tables=use_tables)
        else:
            with open(saved, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            n = ingest.ingest_text(text, role_key, file.filename,
                                   title=os.path.splitext(file.filename)[0])
            result = {"chunks": n}
        result["filename"] = file.filename
        return Resp(msg="入库完成", data=result)
    except Exception as e:
        logger.exception("上传入库失败")
        raise HTTPException(status_code=500, detail="入库失败: %s" % e)


# ==================== 更新 ====================
@update_router.post("/document", response_model=Resp, summary="增量更新单个文档")
async def update_document(req: UpdateDocumentRequest, db: Session = Depends(get_db)):
    """按文件哈希判断是否需要重建；内容没变直接跳过。"""
    _check_role(db, req.role_key)
    try:
        result = get_update_service().update_document(
            req.file_path, req.role_key, force=req.force)
    except Exception as e:
        logger.exception("文档更新失败")
        raise HTTPException(status_code=500, detail="更新失败: %s" % e)
    if result.get("code") != 200:
        raise HTTPException(status_code=400, detail=result.get("msg"))
    return Resp(msg=result.get("msg", "ok"), data=result)


@update_router.post("/dataset/{role_key}", response_model=Resp,
                    summary="批量更新角色的全部数据集")
async def update_dataset(role_key: str, force: bool = False,
                         db: Session = Depends(get_db)):
    _check_role(db, role_key)
    result = get_update_service().update_role_dataset(role_key, force=force)
    if result.get("code") != 200:
        raise HTTPException(status_code=400, detail=result.get("msg"))
    return Resp(msg="批量更新完成", data=result)


@update_router.delete("/document", response_model=Resp, summary="删除某文档的全部知识")
async def delete_document(req: DeleteDocumentRequest, db: Session = Depends(get_db)):
    _check_role(db, req.role_key)
    return Resp(data=get_update_service().delete_document(req.source, req.role_key))


ROUTERS = [ingest_router, update_router]
