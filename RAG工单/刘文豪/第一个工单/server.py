# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
后端服务：FastAPI 封装 RAG 问答（复用 rag.py 与已构建的 chroma_db 索引）
运行（在本目录下）：
  D:\anaconda3\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
浏览器打开 http://127.0.0.1:8000 即为问答界面
"""
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import rag

WEB_DIR = Path("web")

app = FastAPI(title="工单01 基于 PDF 文档的问答系统")
col = rag.get_collection()


class AskReq(BaseModel):
    question: str


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "index_docs": col.count(), "gen_model": rag.GEN_MODEL}


@app.post("/api/ask")
def ask(req: AskReq):
    t0 = time.time()
    ans, hits = rag.answer(col, req.question.strip())
    return {
        "answer": ans,
        "elapsed": round(time.time() - t0, 2),
        "sources": [
            {"page": h["page"], "score": h["score"], "snippet": h["text"][:150]}
            for h in hits
        ],
    }
