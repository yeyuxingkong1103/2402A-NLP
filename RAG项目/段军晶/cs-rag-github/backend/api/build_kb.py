# -*- coding: utf-8 -*-
"""
知识库构建接口

提供两个接口：
    POST /api/kb/build   —— 触发知识库构建/重建（后台执行，不阻塞请求）
    GET  /api/kb/status  —— 查询构建进度与知识库当前状态

为什么要后台执行：
    本项目运行在 CPU 环境，3 份 PDF（74 页）的完整构建包含版面解析、
    分块与向量化，耗时可达数分钟。若同步执行会直接触发 HTTP 超时。
    因此接口立即返回，由前端轮询 /status 获取进度。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config import settings
from backend.db import milvus_client, mysql, redis_client
from backend.logging_config import get_logger, new_request_id

logger = get_logger(__name__)

router = APIRouter(prefix="/api/kb", tags=["知识库"])


# ===========================================================================
# 构建状态（进程内单例，供前端轮询）
# ===========================================================================

_build_lock = threading.Lock()
_build_state: Dict[str, Any] = {
    "running": False,
    "stage": "idle",          # idle / parsing / chunking / embedding / done / failed
    "message": "尚未开始构建",
    "current_file": "",
    "total_files": 0,
    "finished_files": 0,
    "started_at": None,
    "finished_at": None,
    "summary": None,
    "error": None,
}

_STAGE_TEXT = {
    "idle": "空闲",
    "parsing": "正在解析 PDF 版面",
    "chunking": "正在切分文本块",
    "embedding": "正在向量化并写入数据库",
    "done": "构建完成",
    "failed": "构建失败",
}


def _update_state(**kwargs: Any) -> None:
    """线程安全地更新构建状态"""
    with _build_lock:
        _build_state.update(kwargs)


def _get_state() -> Dict[str, Any]:
    """线程安全地读取构建状态副本"""
    with _build_lock:
        return dict(_build_state)


# ===========================================================================
# 构建流程（后台线程执行）
# ===========================================================================

def _run_build(force: bool = False) -> None:
    """
    完整构建流程：解析 -> 分块 -> 向量化入库。

    三个阶段分别复用离线脚本的函数，保证命令行与服务接口行为完全一致。
    """
    from scripts import chunk_split, embedding_store, pdf_parse

    started = time.time()
    summary: Dict[str, Any] = {"documents": [], "total_chunks": 0}

    try:
        settings.ensure_directories()

        # 准备表结构与向量集合
        mysql.init_schema()
        milvus_client.create_collection(drop_existing=force)

        pdf_files = sorted(settings.source_docs_path.glob("*.pdf"))
        if not pdf_files:
            raise RuntimeError(f"源 PDF 目录为空：{settings.source_docs_path}")

        _update_state(total_files=len(pdf_files), finished_files=0)

        for index, pdf_path in enumerate(pdf_files, start=1):
            _update_state(
                current_file=pdf_path.name,
                stage="parsing",
                message=f"[{index}/{len(pdf_files)}] 正在解析：{pdf_path.name}",
            )
            logger.info("[%d/%d] 开始处理：%s", index, len(pdf_files), pdf_path.name)

            # ---- 阶段一：MinerU 版面解析（页码元数据在此产生）----
            parsed = pdf_parse.parse_one(pdf_path, force=force)
            if not parsed:
                logger.error("解析失败，跳过：%s", pdf_path.name)
                continue

            # ---- 阶段二：LangChain 分块（页码元数据在此存活）----
            _update_state(
                stage="chunking",
                message=f"[{index}/{len(pdf_files)}] 正在分块：{pdf_path.name}",
            )
            parsed_file = settings.parsed_path / (
                pdf_parse.safe_stem(pdf_path.name) + ".parsed.json"
            )
            chunked = chunk_split.process_one(parsed_file, force=force)

            # ---- 阶段三：向量化并写入 Milvus + MySQL（页码在此落库）----
            _update_state(
                stage="embedding",
                message=f"[{index}/{len(pdf_files)}] 正在向量化：{pdf_path.name}"
                        f"（CPU 推理，请耐心等待）",
            )
            chunks_file = parsed_file.with_name(
                parsed_file.name.replace(".parsed.json", ".chunks.json"))
            stored = embedding_store.store_one(chunks_file)

            summary["documents"].append({
                "file_name": stored.get("file_name"),
                "page_count": stored.get("page_count"),
                "chunk_count": stored.get("chunk_count"),
            })
            summary["total_chunks"] += stored.get("chunk_count", 0)
            _update_state(finished_files=index)

        # 知识库变更：版本号自增，历史查询缓存失效
        redis_client.bump_kb_version()

        summary["elapsed_seconds"] = round(time.time() - started, 1)
        summary["stats"] = mysql.get_kb_stats()

        _update_state(
            stage="done",
            running=False,
            message=f"构建完成：{len(summary['documents'])} 份文档，"
                    f"{summary['total_chunks']} 个知识块，"
                    f"耗时 {summary['elapsed_seconds']} 秒",
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            summary=summary,
            error=None,
            current_file="",
        )
        logger.info("知识库构建完成 | %s", summary)

    except Exception as exc:
        logger.exception("知识库构建失败：%s", exc)
        _update_state(
            stage="failed",
            running=False,
            message=f"构建失败：{exc}",
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            error=str(exc),
        )


# ===========================================================================
# 请求 / 响应模型
# ===========================================================================

class BuildRequest(BaseModel):
    """构建请求"""

    force: bool = Field(
        False,
        description="是否强制重新构建。false 时已处理过的文档会跳过（增量）",
    )


class BuildResponse(BaseModel):
    """构建触发响应"""

    accepted: bool = Field(..., description="是否成功受理")
    message: str = Field(..., description="提示信息")
    stage: str = Field(..., description="当前阶段")


class KbStatusResponse(BaseModel):
    """知识库状态响应"""

    running: bool = Field(..., description="构建是否进行中")
    stage: str = Field(..., description="当前阶段标识")
    stage_text: str = Field(..., description="当前阶段中文说明")
    message: str = Field(..., description="进度描述")
    current_file: str = Field("", description="正在处理的文件")
    total_files: int = Field(0, description="待处理文件总数")
    finished_files: int = Field(0, description="已处理文件数")
    started_at: Optional[str] = Field(None, description="开始时间")
    finished_at: Optional[str] = Field(None, description="结束时间")
    error: Optional[str] = Field(None, description="失败原因")

    doc_count: int = Field(0, description="已入库文档数")
    page_count: int = Field(0, description="已入库总页数")
    chunk_count: int = Field(0, description="已入库知识块数")
    documents: list = Field(default_factory=list, description="文档清单")


# ===========================================================================
# 接口实现
# ===========================================================================

@router.post("/build", response_model=BuildResponse, summary="构建知识库")
def build_knowledge_base(request: BuildRequest) -> BuildResponse:
    """
    触发知识库构建（异步执行）。

    构建包含：MinerU 版面解析 → LangChain 分块 → BGE-M3 向量化 → 写入 Milvus 与 MySQL。
    接口立即返回，请通过 GET /api/kb/status 轮询进度。
    """
    new_request_id()
    state = _get_state()
    if state["running"]:
        raise HTTPException(status_code=409, detail="知识库正在构建中，请等待当前任务完成")

    _update_state(
        running=True,
        stage="parsing",
        message="构建任务已受理，正在准备…",
        current_file="",
        finished_files=0,
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        finished_at=None,
        summary=None,
        error=None,
    )

    thread = threading.Thread(
        target=_run_build, kwargs={"force": request.force},
        name="kb-build", daemon=True,
    )
    thread.start()

    logger.info("知识库构建任务已启动 | force=%s", request.force)
    return BuildResponse(
        accepted=True,
        message="构建任务已启动，请轮询 /api/kb/status 查看进度",
        stage="parsing",
    )


@router.get("/status", response_model=KbStatusResponse, summary="查询知识库状态")
def kb_status() -> KbStatusResponse:
    """查询构建进度与当前知识库规模"""
    state = _get_state()

    # 已入库规模以 MySQL 为准（页码溯源的真相源）
    documents: list = []
    stats: Dict[str, Any] = {"doc_count": 0, "page_count": 0, "chunk_count": 0}
    try:
        documents = mysql.list_documents()
        stats = mysql.get_kb_stats()
    except Exception as exc:
        logger.warning("读取知识库统计失败（可能尚未初始化）：%s", exc)

    return KbStatusResponse(
        running=state["running"],
        stage=state["stage"],
        stage_text=_STAGE_TEXT.get(state["stage"], state["stage"]),
        message=state["message"],
        current_file=state["current_file"],
        total_files=state["total_files"],
        finished_files=state["finished_files"],
        started_at=state["started_at"],
        finished_at=state["finished_at"],
        error=state["error"],
        doc_count=int(stats.get("doc_count") or 0),
        page_count=int(stats.get("page_count") or 0),
        chunk_count=int(stats.get("chunk_count") or 0),
        documents=[
            {
                "doc_id": d.get("doc_id"),
                "file_name": d.get("file_name"),
                "page_count": d.get("page_count"),
                "chunk_count": d.get("chunk_count"),
                "status": d.get("status"),
                "created_at": str(d.get("created_at") or ""),
            }
            for d in documents
        ],
    )
