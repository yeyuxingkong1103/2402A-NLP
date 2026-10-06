# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：server —— FastAPI 服务（REST + SSE）
# 说明：对外暴露问答、检索、知识库管理等接口；前端 static/index.html 直连。

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Query  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import config           # noqa: E402
import engine           # noqa: E402
import kb as kb_mod     # noqa: E402
import llm              # noqa: E402

app = FastAPI(title="招股说明书 RAG 问答系统",
              description="工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化")

FEEDBACK_PATH = os.path.join(config.DATA_DIR, "feedback.jsonl")


def ok(data=None, message=""):
    return {"code": 0, "data": data, "message": message}


@app.get("/api/health")
def health():
    return ok({"llm": llm.health(), "index_ready": os.path.exists(config.CHUNKS_PATH)})


@app.get("/api/kb/overview")
def kb_overview():
    km = kb_mod.get_kb()
    pages = [c["page"] for c in km.chunks] or [0]
    return ok({
        "chunks": len(km.chunks),
        "dim": int(km.emb.shape[1]) if km.emb is not None else 0,
        "page_min": min(pages), "page_max": max(pages),
        "documents": 1, "doc_name": os.path.basename(config.DEFAULT_PDF),
    })


@app.get("/api/kb/chunks")
def kb_chunks(offset: int = 0, limit: int = 20):
    km = kb_mod.get_kb()
    items = km.chunks[offset:offset + limit]
    return ok({"total": len(km.chunks), "items": items})


@app.post("/api/search")
async def search(payload: dict):
    q = (payload or {}).get("query", "").strip()
    k = int((payload or {}).get("top_k", config.TOP_K))
    if not q:
        return JSONResponse(ok(None, "query 不能为空"))
    t0 = time.time()
    hits, conf = engine.retrieve(q, top_k=k)
    return ok({"query": q, "confidence": conf, "cost_ms": int((time.time() - t0) * 1000),
               "hits": [{"page": h["page"], "section": h.get("section"),
                         "score": h.get("rrf"), "sim": h.get("dense_sim"),
                         "text": h["text"]} for h in hits]})


@app.post("/api/ask")
async def ask(payload: dict):
    q = (payload or {}).get("query", "").strip()
    if not q:
        return JSONResponse(ok(None, "query 不能为空"))
    t0 = time.time()
    r = engine.answer(q)
    return ok({"query": q, "answer": r["answer"], "refused": r["refused"],
               "confidence": r["confidence"], "cost_ms": int((time.time() - t0) * 1000),
               "query_understanding": r.get("query_understanding"),
               "hits": [{"page": h["page"], "page_end": h.get("page_end"),
                         "section": h.get("section"), "sim": h.get("dense_sim"),
                         "text": h["text"]} for h in r["hits"]]})


@app.post("/api/ask/llm")
async def ask_llm(payload: dict):
    """纯 LLM 对照链路（不做检索）。"""
    q = (payload or {}).get("query", "").strip()
    if not q:
        return JSONResponse(ok(None, "query 不能为空"))
    t0 = time.time()
    r = engine.answer_plain_llm(q)
    return ok({"query": q, "answer": r["answer"], "cost_ms": int((time.time() - t0) * 1000)})


@app.post("/api/ask/stream")
async def ask_stream(payload: dict):
    q = (payload or {}).get("query", "").strip()

    def gen():
        if not q:
            yield "data: " + json.dumps({"type": "error", "message": "query 不能为空"},
                                        ensure_ascii=False) + "\n\n"
            return
        for ev in engine.answer_stream(q):
            yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
        yield "data: " + json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/feedback")
async def feedback(payload: dict):
    rec = {"ts": time.time(), **(payload or {})}
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return ok({"saved": True})


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    dst = os.path.join(config.PDF_DIR, file.filename)
    os.makedirs(config.PDF_DIR, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(await file.read())
    return ok({"saved": dst, "hint": "请运行 python app/build_kb.py --parse 重建知识库"})


@app.get("/")
def index():
    return FileResponse(os.path.join(config.STATIC_DIR, "index.html"))


if os.path.isdir(config.STATIC_DIR):
    app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
