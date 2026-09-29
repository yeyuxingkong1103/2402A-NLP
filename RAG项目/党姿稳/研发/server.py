"""
server.py — FastAPI 服务入口

提供 7 个接口（对应需求文档 4.11）：

    POST /chat              普通对话
    POST /chat/stream       流式对话（SSE）
    POST /kb/import         导入知识库 PDF
    POST /kb/import/upload  上传并导入 PDF
    POST /kb/search         知识库检索
    GET  /memory/{user_id}  查看某用户的记忆
    GET  /roles             查看所有角色
    GET  /                  健康检查

启动：python -m uvicorn server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import config
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

import chat
import domains
import memory
from knowledge_base import KnowledgeBase
from llm_client import get_client

app = FastAPI(
    title="多角色 RAG 智能问答系统",
    description="法律 / 医疗 / 英语三领域知识库问答，自动识别意图并路由到对应角色。",
    version="1.0.0",
)

_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


# ---------------------------------------------------------------- 请求模型


class ChatRequest(BaseModel):
    user_id: str = Field(default="default", description="用户标识，用于隔离记忆")
    message: str = Field(..., min_length=1, description="用户问题")


class ImportRequest(BaseModel):
    domain: str = Field(..., description="领域：legal / medical / english")
    pdf_path: str = Field(..., description="PDF 路径，必须位于 data 目录下")
    strategy: str = Field(default="parent_child", description="分块策略")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    domain: str = Field(...)
    top_k: int = Field(default=5, ge=1, le=50)


# ---------------------------------------------------------------- 生命周期


@app.on_event("startup")
async def on_startup() -> None:
    config.ensure_dirs()
    logger = config.setup_logger()
    logger.info(
        f"服务启动 模式={'本地' if config.LOCAL_MODE else '生产'} "
        f"provider={config.LLM_PROVIDER} model={config.get_llm_config()['model']}"
    )
    for problem in config.validate():
        logger.warning(f"配置告警：{problem}")


# ---------------------------------------------------------------- 对话接口


@app.post("/chat", summary="普通对话")
async def chat_endpoint(req: ChatRequest) -> dict:
    try:
        return chat.chat(req.user_id, req.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/stream", summary="流式对话（SSE）")
async def chat_stream_endpoint(req: ChatRequest):
    async def event_generator():
        # chat.chat_stream 是同步生成器，放到线程池里迭代，避免阻塞事件循环
        async for event in iterate_in_threadpool(chat.chat_stream(req.user_id, req.message)):
            yield {"event": event.get("type", "message"), "data": json.dumps(event, ensure_ascii=False)}

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------- 知识库接口


def _safe_path(raw: str) -> Path:
    """限定只能访问 data 目录下的文件，防止路径穿越。"""
    path = Path(raw)
    if not path.is_absolute():
        path = config.DATA_DIR / path

    resolved = path.resolve()
    if not resolved.is_relative_to(config.DATA_DIR.resolve()):
        raise HTTPException(status_code=400, detail="pdf_path 必须位于项目的 data 目录下")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在：{resolved}")
    return resolved


@app.post("/kb/import", summary="导入知识库 PDF")
async def kb_import(req: ImportRequest) -> dict:
    if req.domain not in config.KB_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"未知领域：{req.domain}")

    path = _safe_path(req.pdf_path)
    try:
        return get_kb().import_pdf(path, req.domain, strategy=req.strategy)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"导入失败：{exc}") from exc


@app.post("/kb/import/upload", summary="上传并导入 PDF")
async def kb_import_upload(
    domain: str = Form(...), file: UploadFile = File(...), strategy: str = Form("parent_child")
) -> dict:
    if domain not in config.KB_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"未知领域：{domain}")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    target_dir = config.DATA_DIR / domain
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(file.filename).name

    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        return get_kb().import_pdf(target, domain, strategy=strategy)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"导入失败：{exc}") from exc


@app.post("/kb/search", summary="知识库检索")
async def kb_search(req: SearchRequest) -> dict:
    if req.domain not in config.KB_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"未知领域：{req.domain}")

    docs = get_kb().search(req.query, req.domain, top_k=req.top_k)
    return {
        "query": req.query,
        "domain": req.domain,
        "count": len(docs),
        "results": [
            {
                "text": doc.get("text", ""),
                "source": doc.get("source", ""),
                "page": int(doc.get("page") or 0),
                "chunk_type": doc.get("chunk_type", ""),
                "score": round(float(doc.get("score", 0.0) or 0.0), 4),
                "rerank_score": round(float(doc.get("rerank_score", 0.0) or 0.0), 4),
            }
            for doc in docs
        ],
    }


@app.get("/kb/stats", summary="知识库统计")
async def kb_stats() -> dict:
    return get_kb().stats()


# ---------------------------------------------------------------- 记忆与角色


@app.get("/memory/{user_id}", summary="查看用户记忆")
async def get_memory(user_id: str) -> dict:
    return memory.get_all_memory(user_id)


@app.delete("/memory/{user_id}", summary="清空用户记忆")
async def clear_memory(user_id: str) -> dict:
    memory.clear_all(user_id)
    return {"user_id": user_id, "cleared": True}


@app.get("/roles", summary="查看所有角色")
async def list_roles() -> dict:
    return {"roles": domains.list_roles(), "chat_role": domains.CHAT_ROLE_NAME}


# ---------------------------------------------------------------- 健康检查


@app.get("/", summary="健康检查")
async def health() -> dict:
    cfg = config.get_llm_config()
    return {
        "status": "ok",
        "local_mode": config.LOCAL_MODE,
        "embedding_backend": config.EMBEDDING_BACKEND,
        "llm": {
            "provider": cfg["provider"],
            "model": cfg["model"],
            "base_url": cfg["base_url"],
            "api_key_configured": bool(cfg["api_key"]),
        },
        "kb": get_kb().stats(),
        "config_warnings": config.validate(),
    }


@app.get("/health/llm", summary="模型连通性检查")
async def health_llm() -> dict:
    return get_client().health()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)
