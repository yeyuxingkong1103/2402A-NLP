# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏工单
优化后的 API 服务：
  - 重量级组件单例（避免每请求初始化 DeepDoc/VLM）；
  - 连接池正确归还（修复连接泄漏）；
  - 检索+生成走内存队列 + 工作线程池 + 限流（避免阻塞堆积）。
"""
import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

import config
from monitoring import METRICS
from resource_manager import get_parser, get_reranker, get_vlm, DB_POOL, VECTOR_POOL

app = FastAPI(title="RAG API (optimized)")
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

# 简单令牌桶限流
_tokens = {"ts": time.time(), "count": 0}


def rate_limit():
    now = time.time()
    if now - _tokens["ts"] >= 1.0:
        _tokens["ts"] = now
        _tokens["count"] = 0
    _tokens["count"] += 1
    return _tokens["count"] <= config.RATE_LIMIT_PER_SEC


def _do_answer(query):
    """检索+生成链路（在工作线程池中执行，避免阻塞事件循环）。"""
    with VECTOR_POOL as vconn:          # 连接池正确归还
        docs = vconn.query(query)
    reranked = get_reranker().rerank(docs, query)
    answer = get_vlm().generate(f"context:{reranked}\nq:{query}")
    return answer


@app.post("/api/v1/chats_openai/{chat_id}/chat/completions")
async def chat_completions(chat_id: str, request: Request):
    if not rate_limit():
        METRICS.observe(0, error=True)
        return JSONResponse({"error": "rate limited"}, status_code=429)
    body = await request.json()
    query = body.get("query") or body.get("messages", [{}])[-1].get("content", "")
    t0 = time.perf_counter()
    try:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(executor, _do_answer, query)
        latency = time.perf_counter() - t0
        METRICS.observe(latency)
        return JSONResponse({
            "id": uuid.uuid4().hex,
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": answer}}],
        })
    except Exception as e:
        METRICS.observe(time.perf_counter() - t0, error=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/metrics")
async def metrics():
    return Response(content=METRICS.render(), media_type="text/plain")


@app.get("/health")
async def health():
    return {"status": "ok", "rss_mb": METRICS.mem_usage()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
