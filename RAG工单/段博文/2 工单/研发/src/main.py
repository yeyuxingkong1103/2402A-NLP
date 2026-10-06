# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""
RAG-PDF 问答系统 - FastAPI 主应用（优化版）

完整链路：
    文档上传 → PDF 解析(清洗) → 切分(400/50) → 向量化 → 写入 Milvus(批25)
    用户提问 → 混合检索(召回6) → 重排(保留3) → 缓存 → LLM生成(768) → 返回

接口一览：
    POST /api/upload   上传 PDF
    POST /api/ingest   解析+切分+入库+重建BM25+清缓存
    POST /api/chat     问答（检索+LLM），含计时指标
    GET  /api/search   纯检索
    GET  /api/health   健康检查 + 缓存统计
"""

import os
import json
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import API_HOST, API_PORT, TOP_K_RERANK
from logger import get_logger
from db_milvus import (
    get_embedding, get_vectorstore, get_milvus_client,
    ingest_documents, get_collection_count,
)
from pdf_parser import parse_pdf_to_document
from text_splitter import split_documents
from retriever import hybrid_search, refresh_bm25, get_reranker, get_cache_info, clear_cache
from llm_client import generate_answer, stream_answer

logger = get_logger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时一次性加载全部模型。"""
    logger.info("[启动] 开始初始化 RAG-PDF 问答系统（优化版） ...")
    os.makedirs(DATA_DIR, exist_ok=True)

    get_embedding()
    logger.info("[启动] 本地 bge-m3 向量模型就绪")

    get_milvus_client()
    get_vectorstore()
    logger.info("[启动] Milvus 集合连接就绪")

    try:
        get_reranker()
        logger.info("[启动] bge-reranker 重排模型就绪")
    except Exception as e:
        logger.warning(f"[启动] 重排模型加载失败（{e}），后续检索将只用向量召回")

    try:
        refresh_bm25()
        logger.info("[启动] BM25 索引就绪")
    except Exception as e:
        logger.warning(f"[启动] BM25 重建失败（{e}）")

    logger.info(f"[启动] 完成！打开 http://127.0.0.1:{API_PORT}/docs 测试")
    yield
    logger.info("[关闭] 服务退出")


app = FastAPI(
    title="RAG-PDF 问答系统（优化版）",
    description="基于PDF文档的RAG问答优化：分块400/50 + 召回6 + 重排3 + 结果缓存 + LLM 768token",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
if os.path.isdir(_frontend_dir):
    app.mount("/web", StaticFiles(directory=_frontend_dir), name="frontend")


class IngestRequest(BaseModel):
    files: Optional[List[str]] = Field(default=None, description="要处理的文件名列表")


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户提问")
    top_k: int = Field(default=TOP_K_RERANK, ge=1, le=20, description="引用资料条数")
    stream: bool = Field(default=False, description="是否流式返回")


@app.get("/", tags=["系统"])
def root():
    index_html = os.path.join(_frontend_dir, "index.html") if os.path.isdir(_frontend_dir) else ""
    if index_html and os.path.isfile(index_html):
        return HTMLResponse('<meta http-equiv="refresh" content="0;url=/web/index.html">')
    return {"message": "RAG-PDF 问答系统（优化版）", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health", tags=["系统"])
def health():
    """健康检查 + 缓存统计。"""
    count = get_collection_count()
    return {
        "status": "ok",
        "collection_count": count,
        "data_dir": DATA_DIR,
        "cache": get_cache_info(),
        "version": "2.0",
    }


@app.post("/api/upload", tags=["入库"])
async def upload_pdf(file: UploadFile = File(..., description="PDF 文件")):
    """上传 PDF 文件到 data 目录。"""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 .pdf 文件")

    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, os.path.basename(file.filename))
    try:
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"上传文件失败：{file.filename}（{e}）")
        raise HTTPException(status_code=500, detail=f"保存文件失败：{e}")

    logger.info(f"上传成功：{file.filename} -> {save_path}（{len(content)} bytes）")
    return {"filename": file.filename, "saved_path": save_path, "size": len(content)}


@app.post("/api/ingest", tags=["入库"])
def ingest(req: IngestRequest):
    """解析+切分+入库+重建BM25+清缓存。"""
    if req.files:
        pdf_files = [os.path.join(DATA_DIR, os.path.basename(n)) for n in req.files]
    else:
        if not os.path.isdir(DATA_DIR):
            raise HTTPException(status_code=400, detail=f"data 目录不存在：{DATA_DIR}")
        pdf_files = [
            os.path.join(DATA_DIR, f)
            for f in os.listdir(DATA_DIR)
            if f.lower().endswith(".pdf")
        ]

    if not pdf_files:
        raise HTTPException(status_code=400, detail="data 目录下没有可处理的 PDF 文件")

    logger.info(f"开始入库：共 {len(pdf_files)} 个 PDF")
    ingested = 0
    for i, pdf_path in enumerate(pdf_files, start=1):
        logger.info(f"[{i}/{len(pdf_files)}] 处理：{os.path.basename(pdf_path)}")
        if not os.path.isfile(pdf_path):
            logger.error(f"文件不存在：{pdf_path}")
            continue
        try:
            docs = parse_pdf_to_document(pdf_path)
            if not docs:
                logger.warning(f"{os.path.basename(pdf_path)} 解析为空，跳过")
                continue
            chunks = split_documents(docs)
            if not chunks:
                logger.warning(f"{os.path.basename(pdf_path)} 切分为空，跳过")
                continue
            ingest_documents(chunks)
            ingested += 1
        except Exception as e:
            logger.error(f"处理 {os.path.basename(pdf_path)} 失败：{e}")

    try:
        refresh_bm25()
        clear_cache()  # 入库后清缓存，确保新文档可被检索
    except Exception as e:
        logger.error(f"重建 BM25/清缓存失败：{e}")

    count = get_collection_count()
    logger.info(f"入库完成：成功 {ingested}/{len(pdf_files)}，集合总数 {count}")
    return {
        "processed": len(pdf_files),
        "ingested": ingested,
        "collection_count": count,
    }


@app.get("/api/search", tags=["检索"])
def search(query: str, top_k: int = TOP_K_RERANK):
    """纯检索接口：不调 LLM。"""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k 必须在 1~20 之间")

    t0 = time.time()
    try:
        docs = hybrid_search(query, top_k=top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    elapsed = round(time.time() - t0, 3)
    results = []
    for doc in docs:
        results.append({
            "page_content": doc.page_content,
            "score": doc.metadata.get("score", 0.0),
            "source": doc.metadata.get("source", ""),
            "page_number": doc.metadata.get("page_number", 0),
        })
    logger.info(f"检索：query='{query[:30]}'，命中 {len(results)} 条，耗时 {elapsed}s")
    return {"query": query, "total": len(results), "results": results, "elapsed_sec": elapsed}


@app.post("/api/chat", tags=["问答"])
def chat(req: ChatRequest):
    """问答接口：检索 + LLM 生成（带计时与缓存）。"""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    t_total = time.time()

    # 1. 检索
    try:
        docs = hybrid_search(req.query, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    t_search = time.time() - t_total
    logger.info(f"问答：query='{req.query[:30]}'，引用 {len(docs)} 条资料，检索 {t_search:.3f}s，stream={req.stream}")

    # 2. 流式
    if req.stream:
        def sse_generator():
            try:
                for chunk in stream_answer(req.query, docs):
                    yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"流式生成失败：{e}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 3. 非流式
    t_llm_start = time.time()
    try:
        answer = generate_answer(req.query, docs)
    except Exception as e:
        logger.error(f"LLM 生成失败：{e}")
        raise HTTPException(status_code=500, detail=f"LLM 生成失败：{e}")
    t_llm = time.time() - t_llm_start
    t_all = time.time() - t_total

    references = [
        {
            "page_content": doc.page_content,
            "score": doc.metadata.get("score", 0.0),
            "source": doc.metadata.get("source", ""),
            "page_number": doc.metadata.get("page_number", 0),
        }
        for doc in docs
    ]
    return {
        "query": req.query,
        "answer": answer,
        "references": references,
        "timing": {
            "search_sec": round(t_search, 3),
            "llm_sec": round(t_llm, 3),
            "total_sec": round(t_all, 3),
            "cached": t_llm < 0.05,
        },
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=False)
