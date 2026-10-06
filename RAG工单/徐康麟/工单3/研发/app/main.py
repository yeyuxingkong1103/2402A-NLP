# -*- coding: utf-8 -*-
"""工单3 FastAPI 服务入口（设计/接口设计.md §3.24 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

端点（与设计一致）：
    ``GET  /api/health``  ``GET  /api/files``  ``GET  /api/sessions``
    ``GET  /api/sessions/{session_id}/messages``
    ``POST /api/ask``（非流式）  ``POST /api/ask/stream``（SSE）
    ``POST /api/feedback``（点赞/点踩，T7 新增）  ``POST /api/clear``（清空会话，T7 新增）
启动：``startup`` 事件里 ``setup_logging`` + ``build_engine(warmup=True)``（验收项 4 的硬要求）。

运行：
    pwsh -NoProfile -File run_py.ps1 -m uvicorn app.main:app --host 127.0.0.1 --port 8600
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import get_config  # noqa: E402
from app.core.errors import RagError  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.qa_engine import QAEngine, build_engine  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

app = FastAPI(title="工单3 RAG 服务", version="1.0.0")
_state: dict[str, Any] = {"engine": None, "started_at": None, "warmup": {}}


def _engine() -> QAEngine:
    """取引擎（未初始化时构造但不重复预热，避免阻塞首个请求）。"""
    engine = _state.get("engine")
    if engine is None:
        log = get_logger("main")
        log.log_event("main.engine_lazy_build", level="WARNING", reason="startup 未完成即收到请求")
        engine = build_engine(warmup=False)
        _state["engine"] = engine
    return engine


@app.on_event("startup")
def _startup() -> None:
    """启动：日志初始化 + 引擎预热。"""
    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("main")
    started = time.perf_counter()
    with log.enter("app.startup", {"host": cfg.server_host, "port": cfg.server_port}) as span:
        engine = build_engine(cfg=cfg, warmup=True, logger=log)
        _state["engine"] = engine
        _state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _state["warmup"] = engine.warmup_info
        log.log_event("main.startup_done", warmup=engine.warmup_info,
                      ms=round((time.perf_counter() - started) * 1000, 2))
        span.set_output({"warmup_ms": engine.warmup_info.get("total_ms")})


@app.on_event("shutdown")
def _shutdown() -> None:
    """关闭：写运行记录 + 关闭日志。"""
    log = get_logger("main")
    engine = _state.get("engine")
    if engine is not None:
        engine.sqlite.record_run(run_id=f"api-{int(time.time())}", kind="app.shutdown",
                                 stats={"started_at": _state.get("started_at")}, ok=True)
    log.log_event("main.shutdown")
    shutdown_logging()


@app.get("/api/health")
def health() -> dict[str, Any]:
    """健康检查（检索/生成/存储/文件/预热）。"""
    return _engine().health()


@app.get("/api/files")
def files() -> dict[str, Any]:
    """自动发现的 PDF 列表。"""
    return {"files": _engine().files()}


@app.get("/api/sessions")
def sessions() -> dict[str, Any]:
    """会话列表。"""
    return {"sessions": _engine().sessions()}


@app.get("/api/sessions/{session_id}/messages")
def messages(session_id: str) -> dict[str, Any]:
    """某会话的消息（多轮历史）。"""
    return {"session_id": session_id, "messages": _engine().messages(session_id)}


@app.post("/api/ask")
def ask(payload: dict[str, Any]) -> dict[str, Any]:
    """非流式问答（响应体由 ``QAEngine.answer_payload`` 统一构造，§7 契约字段）。

    §7 状态码：``400`` = 入参非法（空问题、``file_names`` 含不存在的文件）。
    """
    log = get_logger("main")
    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")
    engine = _engine()
    try:
        file_names = engine.validate_files(list(payload.get("file_names") or []) or None)
    except RagError as exc:
        log.log_event("api.ask_bad_request", level="WARNING", code=exc.code, message=str(exc))
        raise HTTPException(status_code=400, detail={"code": exc.code, "message": str(exc)}) from exc
    started = time.perf_counter()
    try:
        answer = engine.ask(question, session_id=payload.get("session_id"), file_names=file_names,
                            top_k=int(payload.get("top_k") or engine.cfg.retrieval.top_k), logger=log)
    except RagError as exc:
        log.log_event("api.ask_failed", level="ERROR", code=exc.code, message=str(exc))
        raise HTTPException(status_code=500, detail={"code": exc.code, "message": str(exc)}) from exc
    return engine.answer_payload(answer, session_id=payload.get("session_id"),
                                 wall_ms=round((time.perf_counter() - started) * 1000, 2))


@app.post("/api/ask/stream")
def ask_stream(payload: dict[str, Any]) -> StreamingResponse:
    """SSE 流式问答（§7：delta 事件 ``{"text","index"}``，末事件为完整答案对象）。"""
    log = get_logger("main")
    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")
    engine = _engine()
    try:
        file_names = engine.validate_files(list(payload.get("file_names") or []) or None)
    except RagError as exc:
        log.log_event("api.stream_bad_request", level="WARNING", code=exc.code, message=str(exc))
        raise HTTPException(status_code=400, detail={"code": exc.code, "message": str(exc)}) from exc

    def _events() -> Iterator[str]:
        index = 0
        for delta in engine.ask(question, session_id=payload.get("session_id"), file_names=file_names,
                                top_k=int(payload.get("top_k") or engine.cfg.retrieval.top_k),
                                stream=True, logger=log):
            event = engine.stream_event(delta, index=index)      # 契约事件体（唯一实现）
            index += 1
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(_events(), media_type="text/event-stream")


@app.post("/api/feedback")
def feedback(payload: dict[str, Any]) -> dict[str, Any]:
    """点赞/点踩。"""
    rating = str(payload.get("rating") or "")
    if rating not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="rating 只接受 up/down")
    fid = _engine().feedback(rating=rating, session_id=payload.get("session_id"),
                             message_id=payload.get("message_id"), answer_id=payload.get("answer_id"),
                             trace_id=payload.get("trace_id"), comment=payload.get("comment"))
    return {"feedback_id": fid}


@app.post("/api/clear")
def clear(payload: dict[str, Any]) -> dict[str, Any]:
    """清空会话。"""
    return {"removed": _engine().clear_conversation(str(payload.get("session_id") or ""))}


if __name__ == "__main__":
    import uvicorn

    cfg = get_config()
    uvicorn.run(app, host=cfg.server_host, port=cfg.server_port)
