# -*- coding: utf-8 -*-
"""文档端点：上传能力 / 入库前判定 / 上传入库 / 列表 / job / 删除 —— 从 ``api/app.py`` 拆出。

四条**产品语义**（改动前务必读）：
1. **入库前判定**（`GET/POST /documents/fit-check`）只做判定、**不写向量库**；
   结论三档（相符 / 不符+建议角色 / 无法判定），最终是否入库由用户在网页决定
   —— 服务端不替用户拍板。
2. **上传必须归属角色**：检索侧按 role_id 严格过滤，未标注的知识**任何角色都搜不到**。
   因此未指定时回退默认角色并在响应里写 `note` 说明，而不是让用户面对"上传成功但问不出东西"。
3. **删除权限默认关闭**（用户裁决「用户没有删除的权力即可」）：普通用户 403 +
   明确原因；本地维护者置 `ALLOW_DOCUMENT_DELETE=true` 才能删。
4. **`rows` 与 `docs_total` 不是同一口径**：前者是 Gauge 存量（`store.count()`），
   后者是 Counter 进程内累计 —— `GET /documents` 的 `note` 显式说明这一点。
"""
from __future__ import annotations

from fastapi import (APIRouter, Depends, File, HTTPException, Query, Request,
                     UploadFile, status)

from ...logging_setup import get_logger
from ...roles import DEFAULT_ROLE_ID, ROLE_LIBRARY, get_role
from ..concurrency import run_blocking
from ..deps import AppState
from ..dependencies import auth_required_only, get_app_state, require_documents
from ..fitcheck import RoleFitJudge
from ..schemas import IngestRequest
from ..services.uploads import prepare_uploads_sync, sample_text
from ..uploads import UploadRejected, summarize_supported

router = APIRouter()

logger = get_logger("legal_rag.api.routers.documents")


async def _prepare_uploads(app_state: AppState, files: list[UploadFile]) -> list[dict]:
    """校验 + 落盘（卸载到线程池，避免读盘/写盘卡住事件循环）。

    处理逻辑在 ``api/services/uploads.py``（**同步函数**，故此处必须走线程池）。
    这里只做两件事：把 ``app_state`` 解包成服务层真正需要的两个值
    （``documents`` / ``config``），以及把阻塞调用卸载出事件循环。
    """
    return await run_blocking(prepare_uploads_sync, app_state.documents,
                              app_state.config, files,
                              limiter=app_state.blocking_limiter())


def _rejected(exc: UploadRejected) -> HTTPException:
    """上传被拒 → 400（reason 分类 + detail，供前端分档提示）。"""
    return HTTPException(status_code=400,
                         detail={"reason": exc.reason, "detail": exc.detail})


@router.post("/ingest")
async def ingest(req: IngestRequest,
                 app_state: AppState = Depends(get_app_state)) -> dict:
    """重建/增量更新知识库。

    走全局并发闸门（``limiter.slot("ingest")``）：入库是重活，不能把线程池占满
    而拖垮并发问答；真正的入库在 ``AppState.ensure_index`` 里（**只在库为空或
    rebuild=True 时才真跑**，避免每次启动都全量重嵌入）。
    """
    if req.sources:
        app_state.sources = req.sources
    assert app_state.limiter is not None
    async with app_state.limiter.slot("ingest"):
        return await run_blocking(app_state.ensure_index, req.rebuild,
                                  limiter=app_state.blocking_limiter())


@router.get("/documents/upload")
async def upload_capabilities(app_state: AppState = Depends(get_app_state)) -> dict:
    """上传能力自描述（允许的扩展名、体积/数量上限、落盘目录）。"""
    return {"supported": summarize_supported(app_state.config)}


@router.get("/documents/fit-check")
async def fit_check_status(app_state: AppState = Depends(get_app_state)) -> dict:
    """判定器状态：当前裁判是谁、能不能用 —— 前端据此说明"判定是否可信"。"""
    judge = RoleFitJudge(app_state.config, app_state.engine)
    return judge.provider_status()


@router.post("/documents/fit-check")
async def fit_check_documents(
    files: list[UploadFile] = File(..., description="待判定的资料"),
    role_id: str = Query(..., min_length=1, description="准备归入的角色 role_id"),
    app_state: AppState = Depends(get_app_state),
) -> dict:
    """**先判定、后入库**：只做判定，不写向量库。

    判定结论三档（见 `api/fitcheck.py`）：相符 / 不符（给建议角色）/ **无法判定**。
    无论哪一档，最终是否入库、入到哪个角色，都由用户在网页上决定 —— 服务端不替用户拍板。
    """
    try:
        payload = await _prepare_uploads(app_state, files)
    except UploadRejected as exc:
        raise _rejected(exc) from exc
    judge = RoleFitJudge(app_state.config, app_state.engine)
    results: list[dict] = []
    for item in payload:
        text, note = await run_blocking(sample_text, item["stored_path"],
                                        limiter=app_state.blocking_limiter())
        verdict = await run_blocking(judge.judge, text, role_id,
                                     filename=item["filename"],
                                     limiter=app_state.blocking_limiter())
        row = {
            "filename": item["filename"],
            "stored_name": item["stored_path"].name,
            "md5": item["md5"],
            "size_bytes": item["size"],
            "extract_note": note,
        }
        row.update(verdict.to_dict())
        results.append(row)
    body = {
        "role_id": role_id,
        "role_known": role_id in ROLE_LIBRARY,
        "role_name": get_role(role_id).name,
        "judge": judge.provider_status(),
        "results": results,
    }
    logger.info("入库前判定：role=%s 文件 %d 个 -> %s", role_id, len(results),
                [(r["fit"], r["suggested_role"]) for r in results])
    return body


@router.post("/documents/upload")
async def upload_documents(
    request: Request,
    files: list[UploadFile] = File(..., description="PDF 为主，兼容 txt/md/json"),
    role_id: str = Query("", description="把这些知识归属到某个角色（分区隔离用）"),
    app_state: AppState = Depends(get_app_state),
) -> dict:
    # 上传会**写入共享知识库**（影响所有角色的检索结果），开关打开时必须登录
    auth_required_only(request)
    documents = require_documents(request)
    assert app_state.limiter is not None
    # 检索侧按 role_id 严格过滤：不指定角色的知识**任何角色都搜不到**。
    # 与其让用户面对「上传成功但问不出东西」，这里明确回退默认角色并把话说清楚。
    requested_role = (role_id or "").strip()
    effective_role = requested_role or DEFAULT_ROLE_ID
    role_note = ""
    if not requested_role:
        role_note = (
            f"未指定 role_id，已按默认角色 role_id={effective_role}"
            f"（{get_role(effective_role).name}）入库；"
            f"若这份资料属于别的角色，请带 ?role_id=<角色ID> 重新上传")
        logger.warning("上传未指定角色：%s", role_note)
    try:
        payload = await _prepare_uploads(app_state, files)
    except UploadRejected as exc:
        raise _rejected(exc) from exc
    async with app_state.limiter.slot("upload"):
        job = documents.create_job()
        # 入库同步阻塞（嵌入 + Milvus 写入）—— 卸载线程池；返回时 job 已是终态，
        # 因此「上传完立刻能检索」成立（也仍可用 /documents/jobs/{id} 轮询）。
        job = await run_blocking(documents.run_job, job, payload, role_id=effective_role,
                                 limiter=app_state.blocking_limiter())
    result = job.to_dict()
    result["accepted"] = [
        {"filename": item["filename"], "doc_id": item["doc_id"],
         "stored_name": item["stored_path"].name, "md5": item["md5"],
         "size_bytes": item["size"], "idempotent_hit": not item["created"]}
        for item in payload
    ]
    result["query"] = {"role_id": effective_role,
                       "role_id_requested": requested_role or None}
    if role_note:
        result["note"] = role_note
    return result


@router.get("/documents")
async def list_documents(request: Request,
                         app_state: AppState = Depends(get_app_state)) -> dict:
    auth_required_only(request)
    documents = require_documents(request)
    rows = await run_blocking(documents.list_documents,
                              limiter=app_state.blocking_limiter())
    return {
        "total": len(rows),
        "documents": rows,
        "rows": documents.rows_snapshot(),
        "note": ("rows 是**当前存量**（Gauge，取自 store.count()）；"
                 "metrics 里的 docs_total/chunks_total 是**进程内累计写入**（Counter），"
                 "两者不是同一个口径"),
    }


@router.get("/documents/jobs")
async def list_jobs(request: Request, limit: int = Query(20, ge=1, le=200)) -> dict:
    documents = require_documents(request)
    return {"jobs": documents.jobs.list(limit)}


@router.get("/documents/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> dict:
    documents = require_documents(request)
    job = documents.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"未知 job_id: {job_id}")
    return job.to_dict()


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request,
                          app_state: AppState = Depends(get_app_state)) -> dict:
    documents = require_documents(request)
    # 知识库删除权限（用户裁决：「用户没有删除的权力即可」）：**默认关闭** ——
    # 普通用户拿到 403 + 明确原因；本地维护者把 ALLOW_DOCUMENT_DELETE=true 打开
    # 才能删（上传与三档判定不受影响；资料列表也仍然共享可见，不按用户过滤）。
    if not bool(getattr(app_state.config, "allow_document_delete", False)):
        logger.warning("拒绝删除资料：doc_id=%s（用户删除权限默认关闭）", doc_id)
        raise HTTPException(status_code=403, detail={
            "error": "document_delete_forbidden",
            "message": "资料只能由维护者删除；如需删除请联系管理员",
        })
    assert app_state.limiter is not None
    async with app_state.limiter.slot("delete"):
        try:
            return await run_blocking(documents.delete_document, doc_id,
                                      limiter=app_state.blocking_limiter())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
