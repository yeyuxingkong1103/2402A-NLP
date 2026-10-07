# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化任务
"""
RAG-PDF 问答系统 - FastAPI 主应用（Query 理解优化版 / 多轮对话）

完整链路：
    文档上传 → PDF解析(文本+表格+图像语义) → 切分(400/50) → 向量化 → 写入 Milvus
    用户提问 → 会话历史加载 → Query改写(指代消解/主语继承切换) →
               混合检索(召回6) → 重排(保留3) → 缓存 → LLM生成(768) → 写回历史 → 返回

接口一览：
    POST /api/upload           上传 PDF
    POST /api/ingest           解析+切分+入库+重建BM25+清缓存
    POST /api/chat             问答（检索+LLM），支持 session_id 多轮对话
    GET  /api/search           纯检索
    GET  /api/health           健康检查 + 缓存统计
    GET  /api/stats            系统统计（文档数、表格数、图像数、缓存等）
    GET  /api/session/{sid}    获取会话历史
    DELETE /api/session/{sid}  清空会话历史
    GET  /api/session_stats    会话统计
"""

import os
import json
import time
import uuid
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
    ingest_documents, get_collection_count, clear_collection,
)
from pdf_parser import parse_pdf_to_document, get_pdf_metadata
from text_splitter import split_documents, get_chunk_stats
from retriever import hybrid_search, refresh_bm25, get_reranker, get_cache_info, clear_cache
from llm_client import generate_answer, stream_answer, get_llm_cache_info, clear_llm_cache
from session_manager import get_session_manager
from query_rewriter import rewrite_query, get_rewrite_cache_info

logger = get_logger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时一次性加载全部模型。"""
    logger.info("[启动] 开始初始化 RAG-PDF 问答系统（Query理解优化版 / 多轮对话） ...")
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
    title="RAG-PDF 问答系统（Query理解优化版）",
    description="基于PDF文档的RAG问答系统，支持多轮对话、指代消解、主语继承与切换",
    version="5.0",
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
    session_id: Optional[str] = Field(default=None, description="会话ID，多轮对话时必传；不传则为单轮")
    use_rewrite: bool = Field(default=True, description="是否启用 Query 改写（多轮时建议开启）")


@app.get("/", tags=["系统"])
def root():
    index_html = os.path.join(_frontend_dir, "index.html") if os.path.isdir(_frontend_dir) else ""
    if index_html and os.path.isfile(index_html):
        return HTMLResponse('<meta http-equiv="refresh" content="0;url=/web/index.html">')
    return {"message": "RAG-PDF 问答系统（Query理解优化版）", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health", tags=["系统"])
def health():
    """健康检查 + 缓存统计。"""
    count = get_collection_count()
    return {
        "status": "ok",
        "collection_count": count,
        "data_dir": DATA_DIR,
        "cache": get_cache_info(),
        "llm_cache": get_llm_cache_info(),
        "rewrite_cache": get_rewrite_cache_info(),
        "sessions": get_session_manager().stats(),
        "version": "5.0",
        "features": ["Query理解优化", "多轮对话", "指代消解", "主语继承与切换",
                     "图像语义解析", "表格解析", "混合检索", "缓存优化"],
    }


@app.get("/api/stats", tags=["系统"])
def stats():
    """系统统计信息。"""
    count = get_collection_count()
    return {
        "collection_count": count,
        "cache": get_cache_info(),
        "llm_cache": get_llm_cache_info(),
        "rewrite_cache": get_rewrite_cache_info(),
        "sessions": get_session_manager().stats(),
    }


@app.get("/api/session/{session_id}", tags=["会话"])
def get_session(session_id: str):
    """获取指定会话的历史记录。"""
    sm = get_session_manager()
    sess = sm.get_or_create(session_id)
    return {
        "session_id": session_id,
        "history": sess.get_history(),
        "last_entity": sess.last_entity,
        "history_length": len(sess.history),
        "created_at": sess.created_at,
        "updated_at": sess.updated_at,
    }


@app.delete("/api/session/{session_id}", tags=["会话"])
def reset_session(session_id: str):
    """清空指定会话的历史。"""
    sm = get_session_manager()
    ok = sm.reset(session_id)
    return {"session_id": session_id, "reset": ok}


@app.get("/api/session_stats", tags=["会话"])
def session_stats():
    """会话全局统计。"""
    return get_session_manager().stats()


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
    total_chunks = 0

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
            total_chunks += len(chunks)
            ingested += 1

            stats = get_chunk_stats(chunks)
            logger.info(f"  统计：{stats}")

        except Exception as e:
            logger.error(f"处理 {os.path.basename(pdf_path)} 失败：{e}")

    try:
        refresh_bm25()
        clear_cache()
        clear_llm_cache()
    except Exception as e:
        logger.error(f"重建 BM25/清缓存失败：{e}")

    count = get_collection_count()
    logger.info(f"入库完成：成功 {ingested}/{len(pdf_files)}，总块数 {total_chunks}，集合总数 {count}")
    return {
        "processed": len(pdf_files),
        "ingested": ingested,
        "total_chunks": total_chunks,
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
            "rerank_score": doc.metadata.get("rerank_score", 0.0),
            "source": doc.metadata.get("source", ""),
            "page_number": doc.metadata.get("page_number", 0),
            "block_type": doc.metadata.get("block_type", "text"),
        })
    logger.info(f"检索：query='{query[:30]}'，命中 {len(results)} 条，耗时 {elapsed}s")
    return {"query": query, "total": len(results), "results": results, "elapsed_sec": elapsed}


@app.post("/api/chat", tags=["问答"])
def chat(req: ChatRequest):
    """问答接口：会话加载 + Query改写 + 检索 + LLM 生成（带计时与缓存）。

    多轮对话流程：
        1. 根据 session_id 加载会话历史；
        2. 调用 Query 改写模块，把当前问题结合历史改写成独立问题；
        3. 用改写后的问题做混合检索与生成；
        4. 把本轮问答写回会话历史。
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    t_total = time.time()

    # 1. 会话管理：获取或创建 session
    session_id = req.session_id or f"anon-{uuid.uuid4().hex[:12]}"
    sm = get_session_manager()
    session = sm.get_or_create(session_id)

    # 2. Query 改写（核心新增）
    rewrite_info = {
        "original": req.query,
        "rewritten": req.query,
        "query_type": "standalone",
        "rewritten_by": "none",
        "elapsed_sec": 0.0,
    }
    search_query = req.query
    if req.use_rewrite:
        try:
            rewrite_info = rewrite_query(req.query, session, use_llm=True)
            search_query = rewrite_info["rewritten"]
            if rewrite_info["rewritten_by"] != "none":
                logger.info(
                    f"[Query改写] '{req.query[:30]}' -> '{search_query[:50]}' "
                    f"(type={rewrite_info['query_type']}, by={rewrite_info['rewritten_by']}, "
                    f"{rewrite_info['elapsed_sec']:.3f}s)"
                )
        except Exception as e:
            logger.warning(f"[Query改写] 失败（{e}），使用原问题")
            search_query = req.query

    # 3. 检索（用改写后的问题）
    try:
        docs = hybrid_search(search_query, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    t_search = time.time() - t_total - rewrite_info.get("elapsed_sec", 0)
    logger.info(
        f"问答：session={session_id}，原问题='{req.query[:30]}'，改写后='{search_query[:50]}'，"
        f"引用 {len(docs)} 条资料，检索 {t_search:.3f}s，stream={req.stream}"
    )

    # 4. 流式
    if req.stream:
        # 先在 SSE 开头下发改写元信息，让前端可展示"理解后的问题"
        def sse_generator():
            try:
                # 元信息包：session_id + 改写结果
                meta = {
                    "meta": True,
                    "session_id": session_id,
                    "original_query": rewrite_info["original"],
                    "rewritten_query": rewrite_info["rewritten"],
                    "query_type": rewrite_info["query_type"],
                    "rewritten_by": rewrite_info["rewritten_by"],
                }
                yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

                full_answer_chunks = []
                for chunk in stream_answer(search_query, docs):
                    full_answer_chunks.append(chunk)
                    yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"

                # 写回会话历史
                full_answer = "".join(full_answer_chunks)
                session.append("user", req.query)
                session.append("assistant", full_answer)

                yield f"data: {json.dumps({'done': True, 'session_id': session_id}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"流式生成失败：{e}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 5. 非流式
    t_llm_start = time.time()
    try:
        answer = generate_answer(search_query, docs)
    except Exception as e:
        logger.error(f"LLM 生成失败：{e}")
        raise HTTPException(status_code=500, detail=f"LLM 生成失败：{e}")
    t_llm = time.time() - t_llm_start
    t_all = time.time() - t_total

    # 写回会话历史
    session.append("user", req.query)
    session.append("assistant", answer)

    references = [
        {
            "page_content": doc.page_content,
            "score": doc.metadata.get("score", 0.0),
            "rerank_score": doc.metadata.get("rerank_score", 0.0),
            "source": doc.metadata.get("source", ""),
            "page_number": doc.metadata.get("page_number", 0),
            "block_type": doc.metadata.get("block_type", "text"),
        }
        for doc in docs
    ]

    return {
        "query": req.query,
        "rewritten_query": search_query,
        "query_type": rewrite_info["query_type"],
        "rewritten_by": rewrite_info["rewritten_by"],
        "session_id": session_id,
        "history_length": len(session.history),
        "answer": answer,
        "references": references,
        "timing": {
            "rewrite": round(rewrite_info.get("elapsed_sec", 0), 3),
            "search": round(t_search, 3),
            "llm": round(t_llm, 3),
            "total": round(t_all, 3),
            "cached": False,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
