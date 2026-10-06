# -*- coding: utf-8 -*-
"""server/routes_build.py —— 上传与构建进度相关接口。

在链路中的位置：
    浏览器 → 【本文件】 → backend/pipeline 的 build_pipeline（后台线程） → Milvus

三个接口：
    POST /api/upload          上传 PDF 并立刻返回，构建在后台线程里跑
    GET  /api/build/status    查询构建进度（前端轮询它）
    GET  /api/documents       已登记文档列表

为什么上传是"立刻返回 + 轮询"而不是同步等构建完：
    一份标准 PDF 的解析 + 向量化要几分钟，同步返回会让浏览器请求超时、
    用户以为上传失败。改成回执后轮询，用户能看到每一步进度。
"""
from __future__ import annotations

import threading
import time

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

try:
    from ..pipeline import PDF_DIR, build_pipeline
    from ..retrieval import invalidate_cache
except ImportError:
    from pipeline import PDF_DIR, build_pipeline
    from retrieval import invalidate_cache

from .config import logger
from .registry import load_docs, update_doc
from .security import pdf_name
from .state import build_state, update_step

router = APIRouter()

@router.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """上传 PDF 并触发后台构建。

    流程：
        校验 -> 落盘到 data/pdfs/ -> 起后台线程跑构建 -> 立即返回

    为什么是后台线程 + 立即返回，而不是同步等构建完：
        一份标准 PDF 的解析 + 向量化要几分钟，同步返回会让浏览器请求超时、
        用户以为上传失败。改成"立刻回执 + 前端轮询 /api/build/status"，
        用户能看到每一步进度，体验完全不同。

    返回：
        {"ok": True, "filename": 文件名}；被拒时返回 400/409 及错误说明。
    """
    # 409 Conflict：已有任务在跑。放在最前面，避免白读一遍上传内容
    if build_state["running"]:
        return JSONResponse({"ok": False, "error": "已有构建任务进行中，请稍候"}, status_code=409)

    filename = pdf_name(file.filename or "upload.pdf")
    if not filename:
        return JSONResponse({"ok": False, "error": "只支持不含目录的 PDF 文件名"}, status_code=400)

    pdf_path = PDF_DIR / filename
    pdf_path.write_bytes(await file.read())
    # 重置构建状态：清空上轮的步骤列表，否则新任务会带着上轮的进度条一起显示
    build_state.update(
        running=True,
        steps=[],
        error=None,
        doc={"filename": filename, "size": pdf_path.stat().st_size},
    )

    def run_build() -> None:
        """后台线程体：跑完整构建并更新登记表与缓存。"""
        try:
            stats = build_pipeline(filename, update_step)
            update_doc({
                "filename": filename,
                "size": pdf_path.stat().st_size,
                **stats,  # 展开 pages/chars/chunks/vectors/seconds
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            # 必须让检索侧的语料/BM25 缓存失效，否则新上传的内容在检索时"看不见"
            invalidate_cache()
            update_step("完成", "done", f"构建成功：{stats['chunks']} 块 / {stats['vectors']} 向量")
        except Exception as exc:
            logger.exception("PDF build failed")  # 打完整堆栈到日志，便于排查解析失败
            build_state["error"] = str(exc)
            update_step("失败", "error", str(exc))
        finally:
            # 无论成功失败都要复位，否则一次失败会让服务永远拒绝后续上传
            build_state["running"] = False

    # daemon=True：主进程退出时不让构建线程拖住服务关闭
    threading.Thread(target=run_build, daemon=True).start()
    return {"ok": True, "filename": filename}

@router.get("/api/build/status")
def build_status():
    """查询构建进度（前端轮询这个接口）。

    返回：
        直接返回 build_state 全局对象：running / steps / error / doc。
    """
    return build_state

@router.get("/api/documents")
def documents():
    """列出已登记文档（页数、块数、构建时间等）。

    注意：
        这里读的是 documents.json 登记表，不是实时扫 Milvus。
        想看 Milvus 里的真实数据量请用 /api/kb/overview。
    """
    return {"documents": load_docs()}
