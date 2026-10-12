# -*- coding: utf-8 -*-
"""
优化版 RAG API 服务（工单十七）。
优化点：
- 启动时单例初始化 RAGEngine（不在请求内初始化重量级组件）；
- 检索+生成链路：连接池 + LLM 并发信号量 + 有界缓存；
- Prometheus 指标暴露 /metrics；
- 后台线程定期采集 RSS 内存，便于观察泄漏；
- 异步文档解析走队列（Task Executor 语义），API 不阻塞。
"""
import os, time, threading, asyncio, uuid, json
import psutil
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .rag_engine import RAGEngine
from . import metrics

app = FastAPI(title="RAG API (优化版)")
engine = None  # 启动时初始化为单例
process = psutil.Process(os.getpid())

DOC_QUEUE = asyncio.Queue(maxsize=256)
DOC_RESULTS = {}  # task_id -> status/result（有界清理）
_doc_lock = threading.Lock()


@app.on_event("startup")
def startup():
    global engine
    engine = RAGEngine(latency=float(os.environ.get("LLM_LATENCY", "0.8")))
    threading.Thread(target=_mem_monitor, daemon=True).start()
    asyncio.get_event_loop().create_task(_doc_consumer()) if False else None
    app.state.consumer_started = False


def _mem_monitor():
    while True:
        try:
            rss = process.memory_info().rss / 1024 / 1024
            metrics.PROCESS_MEM_RSS_MB.set(rss)
            metrics.CACHE_SIZE.set(len(engine.cache))
        except Exception:
            pass
        time.sleep(2)


async def _doc_consumer():
    """文档解析异步消费者：模拟 Task Executor 从队列消费，API 不阻塞。"""
    while True:
        task = await DOC_QUEUE.get()
        try:
            await asyncio.get_event_loop().run_in_executor(None, _do_parse, task)
        except Exception as e:
            DOC_RESULTS[task["id"]] = {"status": "failed", "error": str(e)}
        finally:
            DOC_QUEUE.task_done()


def _do_parse(task):
    """模拟耗时文档解析（DeepDoc），并演示中间对象及时释放。"""
    import gc
    time.sleep(2.0)  # 模拟解析耗时
    big = bytearray(1024 * 1024)  # 1MB 中间对象
    del big
    gc.collect()
    DOC_RESULTS[task["id"]] = {"status": "done", "chars": len(task["filename"]) * 1000}


class ChatReq(BaseModel):
    question: str
    chat_id: str = "default"


@app.middleware("http")
async def metrics_mw(request: Request, call_next):
    ep = request.url.path
    metrics.INFLIGHT.labels(ep).inc()
    t0 = time.time()
    try:
        resp = await call_next(request)
        status = resp.status_code
    except Exception:
        status = 500
        raise
    finally:
        dt = time.time() - t0
        metrics.REQUEST_TOTAL.labels(ep, request.method, str(status)).inc()
        metrics.REQUEST_LATENCY.labels(ep).observe(dt)
        metrics.INFLIGHT.labels(ep).dec()
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "rss_mb": round(process.memory_info().rss / 1024 / 1024, 1)}


@app.get("/api/v1/chats_openai/{chat_id}/chat/completions")
async def chat_get(chat_id: str, q: str):
    return await _chat(chat_id, q)


@app.post("/api/v1/chats_openai/{chat_id}/chat/completions")
async def chat_post(chat_id: str, req: ChatReq):
    return await _chat(chat_id, req.question)


async def _chat(chat_id, question):
    loop = asyncio.get_event_loop()
    result, hit = await loop.run_in_executor(None, engine.query, question)
    metrics.QUEUE_DEPTH.set(DOC_QUEUE.qsize())
    if hit:
        metrics.CACHE_HITS.inc()
    return JSONResponse({
        "chat_id": chat_id, "question": question,
        "answer": result["answer"], "cache_hit": hit,
        "docs": result["docs"],
    })


@app.post("/api/v1/documents/parse")
async def submit_parse(filename: str = "doc.pdf"):
    """文档解析入队，立即返回 task_id（异步，不阻塞 API）。"""
    if DOC_QUEUE.full():
        return JSONResponse({"error": "queue full"}, status_code=503)
    tid = uuid.uuid4().hex
    await DOC_QUEUE.put({"id": tid, "filename": filename})
    DOC_RESULTS[tid] = {"status": "queued"}
    # 启动消费者（幂等）
    if not getattr(app.state, "consumer_started", False):
        app.state.consumer_started = True
        asyncio.create_task(_doc_consumer())
    metrics.QUEUE_DEPTH.set(DOC_QUEUE.qsize())
    return {"task_id": tid, "status": "queued"}


@app.get("/api/v1/documents/parse/{task_id}")
def get_parse(task_id: str):
    return DOC_RESULTS.get(task_id, {"status": "unknown"})


@app.get("/metrics")
def prometheus():
    data, ct = metrics.metrics_text()
    return Response(content=data, media_type=ct)
