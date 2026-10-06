# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""入库接口：后台任务 + 进度轮询（548 页全程 1–3 分钟，不能同步阻塞）。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.pipeline import get_pipeline
from app.schemas import IngestRequest, IngestStatus

router = APIRouter(tags=["ingest"])


@router.post("/api/ingest", response_model=IngestStatus)
async def start_ingest(req: IngestRequest) -> IngestStatus:
    pipe = get_pipeline()
    if pipe.state.running:
        raise HTTPException(409, "已有入库任务在运行，请等待完成")

    pdf = req.pdf_path or str(settings.data_path / "raw" / "招股说明书1.pdf")
    # 后台跑，接口立刻返回，前端轮询 /api/ingest/status
    asyncio.create_task(pipe.run(pdf, rebuild=req.rebuild,
                                 limit_pages=req.limit_pages))
    await asyncio.sleep(0.1)   # 让任务进入 running 状态后再返回
    return IngestStatus(**pipe.status().as_dict())


@router.get("/api/ingest/status", response_model=IngestStatus)
async def ingest_status() -> IngestStatus:
    return IngestStatus(**get_pipeline().status().as_dict())
