# 工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失工单
"""工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失。独立扩展验证API，非冒充完整RAGFlow。"""

import asyncio
import json
import os
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field
from .pipeline import Pipeline, CONFIG, ROOT, DATA, DOCS, parse_document

app = FastAPI(title=f"工单{CONFIG['number']} 工业文档验证")
pipeline = Pipeline()
sem = asyncio.Semaphore(4)
metrics = {
    "requests": 0,
    "errors": 0,
    "active": 0,
    "parse_tasks": 0,
    "completed_parse": 0,
}


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document: str | None = None
    optimized: bool = True


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "project": CONFIG,
        "parsed_pages": len(pipeline.records),
        "service": "独立验证扩展API",
        "metrics": metrics,
    }


@app.post("/api/ask")
async def ask(q: Query):
    metrics["requests"] += 1
    metrics["active"] += 1
    try:
        async with sem:
            return await asyncio.to_thread(
                pipeline.ask, q.question, q.document, q.optimized
            )
    except Exception as exc:
        metrics["errors"] += 1
        raise HTTPException(503, str(exc))
    finally:
        metrics["active"] -= 1


@app.get("/pages/{document}/{page}")
def image(document: str, page: int):
    candidate = (DATA / "pages" / document / f"{page:03}.png").resolve()
    if (
        not candidate.is_relative_to((DATA / "pages").resolve())
        or not candidate.exists()
    ):
        raise HTTPException(404)
    return FileResponse(candidate)


@app.get("/api/report")
def report():
    p = ROOT.parent / "测试/实测结果.json"
    return (
        json.loads(p.read_text(encoding="utf-8"))
        if p.exists()
        else {"status": "尚未执行"}
    )


@app.get("/metrics")
def prometheus():
    return PlainTextResponse(
        "\n".join(f"industrial_{k} {v}" for k, v in metrics.items()) + "\n"
    )


if CONFIG["number"] == 18:
    from .workflow import document_ingestion_workflow

    @app.post("/v1/document/quality-inspection")
    async def inspect_quality(request: Request):
        body = await request.json()
        names = body.get("paths") or [str(DOCS)]
        allowed = ROOT.resolve()
        paths = [Path(p).resolve() for p in names]
        if any(not p.is_relative_to(allowed) for p in paths):
            raise HTTPException(403, "只能质检当前工单研发目录内的文件")
        result = await asyncio.to_thread(
            document_ingestion_workflow, paths, None, ROOT.parent / "测试/API质检"
        )
        return result

    @app.get("/quality-report")
    def quality_report():
        p = ROOT.parent / "测试/全量质检/quality_report.html"
        return FileResponse(p)
