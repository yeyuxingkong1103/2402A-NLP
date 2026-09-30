# -*- coding: utf-8 -*-
"""
PDF 上传接口

提供：
    POST /api/kb/upload —— 上传 PDF 文件并纳入知识库

处理方式：
    1. 校验文件（扩展名、大小、是否重复）
    2. 落盘到知识库源目录
    3. 默认**不**立即触发解析，由调用方决定何时点击「加载知识库」

    这样设计的原因：解析与向量化在 CPU 环境下耗时较长（约 10 秒/页），
    若在上传请求内同步完成会直接触发 HTTP 超时；
    而上传多个文件时逐个触发解析也会造成资源争抢。
    统一走「上传 -> 点加载 -> 后台构建」，交互更可控。
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.config import settings
from backend.db import mysql
from backend.logging_config import get_logger, new_request_id

logger = get_logger(__name__)

router = APIRouter(prefix="/api/kb", tags=["知识库"])

# 单文件大小上限：50 MB（国家标准类 PDF 通常远小于此）
MAX_FILE_SIZE = 50 * 1024 * 1024

ALLOWED_SUFFIXES = {".pdf"}


class UploadResponse(BaseModel):
    """上传响应"""

    accepted: bool = Field(..., description="是否接收成功")
    file_name: str = Field(..., description="保存后的文件名")
    file_size: int = Field(..., description="文件大小（字节）")
    duplicated: bool = Field(False, description="该文件是否已存在于知识库中")
    message: str = Field(..., description="提示信息")


def _sha256_of(path: Path) -> str:
    """计算文件 SHA256"""
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


@router.post("/upload", response_model=UploadResponse, summary="上传 PDF 加入知识库")
async def upload_pdf(file: UploadFile = File(..., description="待上传的 PDF 文件")) -> UploadResponse:
    """
    上传 PDF 文件到知识库源目录。

    上传完成后，调用 POST /api/kb/build 即可将其解析并纳入检索范围。
    """
    new_request_id()
    settings.ensure_directories()

    raw_name = file.filename or "unnamed.pdf"
    # 只取文件名，防止路径穿越
    file_name = Path(raw_name).name

    suffix = Path(file_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持 PDF 文件，当前文件类型：{suffix or '未知'}",
        )

    target = settings.source_docs_path / file_name
    total = 0  # 必须在 try 之前初始化：文件打开失败时下方日志仍会引用它

    # 分块写入，同时统计大小，避免一次性读入内存
    try:
        with target.open("wb") as fh:
            total = 0
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_FILE_SIZE:
                    fh.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件过大，上限为 {MAX_FILE_SIZE // 1024 // 1024} MB",
                    )
                fh.write(chunk)
    finally:
        await file.close()

    logger.info("PDF 上传完成：%s | %.2f MB", file_name, total / 1024 / 1024)

    # 重复检测：按内容哈希比对已入库文档
    duplicated = False
    try:
        file_hash = _sha256_of(target)
        existing = mysql.get_document_by_hash(file_hash)
        duplicated = bool(existing)
    except Exception as exc:
        logger.warning("重复检测失败（不影响上传）：%s", exc)

    if duplicated:
        message = f"「{file_name}」内容已在知识库中，无需重复添加"
    else:
        message = f"「{file_name}」上传成功，点击「加载知识库」后即可提问"

    return UploadResponse(
        accepted=True,
        file_name=file_name,
        file_size=total,
        duplicated=duplicated,
        message=message,
    )


@router.get("/documents", summary="查看知识库文档清单")
def list_documents() -> Dict[str, Any]:
    """
    返回已入库的文档清单（供前端「知识库加载」区域展示）。
    """
    try:
        documents = mysql.list_documents()
    except Exception as exc:
        logger.warning("读取文档清单失败：%s", exc)
        documents = []

    return {
        "total": len(documents),
        "documents": [
            {
                "doc_id": d.get("doc_id"),
                "file_name": d.get("file_name"),
                "page_count": d.get("page_count"),
                "chunk_count": d.get("chunk_count"),
                "status": d.get("status"),
            }
            for d in documents
        ],
    }
