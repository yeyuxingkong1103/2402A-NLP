# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务
# 模块：server —— FastAPI 服务（REST + SSE）
# 说明：对外暴露问答、检索、知识库管理、**多轮对话（工单 05）**等接口；前端 static/index.html 直连。

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
import dialog           # noqa: E402
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
    docs = {}
    for c in km.chunks:
        d = c.get("doc") or "?"
        it = docs.setdefault(d, {"chunks": 0, "page_min": 10 ** 6, "page_max": 0})
        it["chunks"] += 1
        it["page_min"] = min(it["page_min"], c["page"])
        it["page_max"] = max(it["page_max"], c["page"])
    return ok({
        "chunks": len(km.chunks),
        "dim": int(km.emb.shape[1]) if km.emb is not None else 0,
        "page_min": min(pages), "page_max": max(pages),
        "documents": len(docs) or 1,
        "docs": [{"doc": d, **v} for d, v in sorted(docs.items())],
        "doc_name": os.path.basename(config.DEFAULT_PDF),
    })


@app.get("/api/kb/chunks")
def kb_chunks(offset: int = 0, limit: int = 20):
    km = kb_mod.get_kb()
    items = km.chunks[offset:offset + limit]
    return ok({"total": len(km.chunks), "items": items})


@app.post("/api/search")
async def search(payload: dict):
    p = payload or {}
    q = p.get("query", "").strip()
    k = int(p.get("top_k", config.TOP_K))
    doc = p.get("doc") or None
    if not q:
        return JSONResponse(ok(None, "query 不能为空"))
    strategy = p.get("strategy") or None
    reranker = p.get("reranker") or "none"
    weights = p.get("weights") or None
    fusion = p.get("fusion") or None
    t0 = time.time()
    hits, conf = engine.retrieve(q, top_k=k, doc=doc, strategy=strategy, reranker=reranker,
                                 weights=tuple(weights) if weights else None, fusion=fusion)
    return ok({"query": q, "confidence": conf, "doc": doc or engine.detect_doc(q),
               "strategy": strategy or "rrf", "reranker": reranker,
               "cost_ms": int((time.time() - t0) * 1000),
               "hits": [{"doc": h.get("doc"), "page": h["page"], "section": h.get("section"),
                         "type": h.get("type"), "score": h.get("rrf"), "sim": h.get("dense_sim"),
                         "text": h["text"]} for h in hits]})


@app.post("/api/ask")
async def ask(payload: dict):
    p = payload or {}
    q = p.get("query", "").strip()
    if not q:
        return JSONResponse(ok(None, "query 不能为空"))
    t0 = time.time()
    r = engine.answer(q, doc=p.get("doc") or None,
                      strategy=p.get("strategy") or None,
                      reranker=p.get("reranker") or "none")
    return ok({"query": q, "answer": r["answer"], "refused": r["refused"],
               "confidence": r["confidence"], "cost_ms": int((time.time() - t0) * 1000),
               "doc": r.get("doc"),
               "query_understanding": r.get("query_understanding"),
               "hits": [{"doc": h.get("doc"), "page": h["page"], "page_end": h.get("page_end"),
                         "section": h.get("section"), "type": h.get("type"),
                         "sim": h.get("dense_sim"), "text": h["text"]} for h in r["hits"]]})


@app.post("/api/ask/llm")
async def ask_llm(payload: dict):
    """纯 LLM 对照链路（不做检索）。"""
    q = (payload or {}).get("query", "").strip()
    if not q:
        return JSONResponse(ok(None, "query 不能为空"))
    t0 = time.time()
    r = engine.answer_plain_llm(q)
    return ok({"query": q, "answer": r["answer"], "cost_ms": int((time.time() - t0) * 1000)})


# ---------- 多轮对话（工单 05：人工智能NLP-RAG-Query理解优化任务） ----------
@app.post("/api/session/new")
async def session_new():
    return ok({"session_id": dialog.new_session()})


@app.get("/api/session/{sid}")
async def session_get(sid: str):
    return ok(dialog.get_state(sid).as_dict())


@app.post("/api/session/reset")
async def session_reset(payload: dict):
    sid = (payload or {}).get("session_id", "")
    dialog.reset_session(sid)
    return ok({"session_id": dialog.new_session()})


@app.get("/api/sessions")
async def sessions():
    return ok({"sessions": dialog.list_sessions()})


@app.post("/api/chat")
async def chat(payload: dict):
    """多轮对话：带 session_id 时先做指代消解/追问改写，再检索生成。"""
    p = payload or {}
    q = p.get("query", "").strip()
    if not q:
        return JSONResponse(ok(None, "query 不能为空"))
    r = dialog.chat(p.get("session_id") or dialog.new_session(), q, doc=p.get("doc") or None)
    return ok(r)


@app.post("/api/ask/stream")
async def ask_stream(payload: dict):
    p = payload or {}
    q = p.get("query", "").strip()
    sid = p.get("session_id")
    rewrite_ev = None
    if q and sid:
        # 多轮：先做指代消解，把改写后的完整问题交给检索链路
        try:
            rq, info = dialog.resolve(q, dialog.get_state(sid))
            rewrite_ev = {"type": "rewrite", "rewritten_query": rq, "resolution": info}
            q = rq
        except Exception as e:  # noqa: BLE001
            rewrite_ev = {"type": "rewrite", "error": str(e)}

    def gen():
        if not q:
            yield "data: " + json.dumps({"type": "error", "message": "query 不能为空"},
                                        ensure_ascii=False) + "\n\n"
            return
        if rewrite_ev:
            yield "data: " + json.dumps(rewrite_ev, ensure_ascii=False) + "\n\n"
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


@app.get("/api/perf/stats")
def perf_stats():
    """工单 13：运行时性能观测（嵌入缓存命中率、tokens 等）。"""
    return ok({"embed_cache": llm.cache_stat(), "refuse_score": config.REFUSE_SCORE})


@app.get("/")
def index():
    return FileResponse(os.path.join(config.STATIC_DIR, "index.html"))


# ---------- 工单 13：启动预热（消除首问冷启动，检索 P95 ≤3s） ----------
@app.on_event("startup")
def _startup_warmup():
    """服务启动后后台预热：加载索引/BM25 + 预热嵌入与生成模型。"""
    import threading

    def _w():
        try:
            t0 = time.time()
            kb_mod.get_kb()                     # 载入索引/向量/BM25
            info = llm.warmup(gen=True)         # 预热嵌入 + 生成
            print("[warmup] %.2fs %s" % (time.time() - t0, info))
        except Exception as e:  # noqa: BLE001
            print("[warmup] skipped:", e)

    threading.Thread(target=_w, daemon=True).start()


# ---------- 工单 12：LightRAG 知识库检索（可选 RAG / LightRAG） ----------
_LR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LR_PY = os.path.join(_LR_ROOT, ".venv_lightrag", "Scripts", "python.exe")
LR_SCRIPT = os.path.join(_LR_ROOT, "lightrag_rag", "query_lightrag.py")


def lightrag_ask(q, mode="mix", timeout=600):
    """调用独立 venv（装了 lightrag-hku）子进程做 LightRAG 检索+生成；主环境保持轻依赖。"""
    import subprocess
    cmd = [LR_PY, LR_SCRIPT, "--q", q, "--mode", mode]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run(cmd, cwd=_LR_ROOT, capture_output=True, env=env,
                       timeout=timeout)
    out = (p.stdout or b"").decode("utf-8", "ignore")
    ans, chunks = "", []
    for line in out.splitlines():
        if line.startswith("A: "):
            ans = line[3:].strip()
        elif line.strip().startswith("· p"):
            chunks.append(line.strip()[2:])
    return {"answer": ans, "contexts": chunks, "raw_tail": out[-600:],
            "ok": bool(ans), "rc": p.returncode}


@app.post("/api/lightrag")
async def lightrag(payload: dict):
    """工单 12：用 LightRAG 知识库（图 + 向量双层检索）回答问题。"""
    import asyncio
    p = payload or {}
    q = (p.get("query") or "").strip()
    if not q:
        return JSONResponse(ok(None, "query 不能为空"))
    if not os.path.isfile(LR_PY):
        return JSONResponse(ok(None, "未找到 LightRAG 环境（%s），请先按 docs/LightRAG优化方案.md 安装" % LR_PY))
    t0 = time.time()
    r = await asyncio.to_thread(lightrag_ask, q, p.get("mode") or "mix")
    return ok({"query": q, "backend": "lightrag", "mode": p.get("mode") or "mix",
               "answer": r["answer"], "hits": r["contexts"],
               "ok": r["ok"], "cost_ms": int((time.time() - t0) * 1000),
               "note": "LightRAG 子进程（.venv_lightrag）：实体-关系图 + 向量双层检索"})


@app.get("/api/perf")
def perf():
    """工单 13：运行时性能自检（嵌入缓存命中率等）。"""
    return ok({"embed_cache": llm.cache_stat(), "kb_chunks": len(kb_mod.get_kb().chunks)})


if os.path.isdir(config.STATIC_DIR):
    app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


# ---------- 工单 13：启动预热（消除首问冷启动，检索 P95 ≤3s） ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
