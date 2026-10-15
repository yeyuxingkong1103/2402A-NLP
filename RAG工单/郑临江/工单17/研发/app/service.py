"""工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏。"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import Literal
from .resources import Capacity, TTLCache
import asyncio, psutil, os, json, time, hashlib, uuid
from pathlib import Path
from .pipeline import Pipeline, ROOT, DOCS, DATA, parse_document
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

app = FastAPI()
capacity = Capacity()
cache = TTLCache()
metrics = {"requests": 0, "errors": 0, "cache_hits": 0, "queue_rejects": 0}
pipeline = Pipeline()
parse_queue = asyncio.Queue(maxsize=16)
jobs = {}
worker = None
parse_pool = None


@app.get("/debug/memory")
def memory_debug():
    import tracemalloc, gc, collections

    if not tracemalloc.is_tracing():
        tracemalloc.start(8)
    gc.collect()
    snapshot = tracemalloc.take_snapshot()
    destination = ROOT.parent / "测试/memory_snapshots"
    destination.mkdir(exist_ok=True)
    suffix = str(int(time.time()))
    snapshot.dump(str(destination / (suffix + ".snapshot")))
    counts = collections.Counter(type(o).__name__ for o in gc.get_objects())
    return {
        "rss_mb": psutil.Process().memory_info().rss / 1048576,
        "threads": psutil.Process().num_threads(),
        "object_types": counts.most_common(20),
        "allocation_top": [str(s) for s in snapshot.statistics("lineno")[:15]],
    }


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["retrieval", "generation"] = "retrieval"
    use_cache: bool = True


@app.get("/health")
def health():
    return {
        "ok": True,
        "project": 17,
        "build": "process-isolated",
        "project_root": str(ROOT),
        "metrics": metrics,
        "rss_mb": psutil.Process(os.getpid()).memory_info().rss / 1048576,
    }


@app.get("/metrics")
def prometheus():
    return PlainTextResponse(
        "\n".join(f"api_{k} {v}" for k, v in metrics.items())
        + f"\napi_parse_queue {parse_queue.qsize()}\napi_rss_bytes {psutil.Process().memory_info().rss}\n"
    )


@app.get("/")
def index():
    return HTMLResponse(
        """<meta charset="utf-8"><style>body{font:16px/1.7 Microsoft YaHei;margin:35px;color:#203b45;background:#f0f5f6}.card{padding:20px;background:white;border-radius:10px;margin:15px 0}pre{white-space:pre-wrap}img{max-width:100%;max-height:400px}</style><h1>工单17 · 并发与资源观测</h1><p>设计 · 研发 · 测试 · 优化 · 部署</p><div class="card"><h2>当前API资源</h2><pre id="m"></pre></div><div class="card"><h2>真实JMeter前后对照</h2><p>独立OCR证据检索/缓存接口，未把检索耗时当作每请求LLM生成耗时。B为5检索+5上传，原始日志保留。</p><pre id="b"></pre><img src="/benchmark/chart"></div><script>setInterval(async()=>m.textContent=JSON.stringify(await(await fetch('/health')).json(),null,2),1000);fetch('/benchmark').then(r=>r.json()).then(r=>{const x={};for(const [key,v] of Object.entries(r.scenarios||{}))x[key]={seconds:v.duration_s,requests:v.requests,errors:v.errors,completed_600s:v.complete_600s};b.textContent=JSON.stringify(x,null,2)})</script>"""
    )


@app.get("/benchmark")
def benchmark():
    return json.loads(
        (ROOT.parent / "测试/JMeter汇总.json").read_text(encoding="utf-8")
    )


@app.get("/benchmark/chart")
def benchmark_chart():
    return FileResponse(ROOT.parent / "测试/压测对比.png")


async def consume():
    while True:
        job = await parse_queue.get()
        try:
            jobs[job["id"]]["state"] = "processing"
            await asyncio.get_running_loop().run_in_executor(
                parse_pool, parse_document, job["document"]
            )
            pipeline.reload()
            cache.entries.clear()
            jobs[job["id"]]["state"] = "completed"
        except Exception as exc:
            jobs[job["id"]].update(state="failed", error=str(exc))
        finally:
            parse_queue.task_done()
        while len(jobs) > 128:
            jobs.pop(next(iter(jobs)))


@app.on_event("startup")
async def startup():
    global worker, parse_pool
    parse_pool = ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    )
    worker = asyncio.create_task(consume())


@app.on_event("shutdown")
async def shutdown():
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass
    parse_pool.shutdown(wait=False, cancel_futures=True)


@app.post("/api/upload")
async def upload(request: Request, name: str):
    if Path(name).name != name or not name.endswith(".pdf"):
        raise HTTPException(400, "invalid_pdf_name")
    if int(request.headers.get("Content-Length", "0")) > 5 * 1024 * 1024:
        raise HTTPException(413, "file_too_large")
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(413, "file_too_large")
    content = bytes(content)
    if len(content) > 5 * 1024 * 1024 or not content.startswith(b"%PDF"):
        raise HTTPException(400, "invalid_pdf")
    uploads = DATA / "uploads"
    uploads.mkdir(exist_ok=True)
    path = uploads / name
    if (
        path.exists()
        and hashlib.sha256(path.read_bytes()).digest()
        != hashlib.sha256(content).digest()
    ):
        raise HTTPException(409, "same_name_different_document")
    digest = hashlib.sha256(content).hexdigest()
    existing = next(
        (
            j
            for j in jobs.values()
            if j.get("sha256") == digest
            and j["state"] in {"queued", "processing", "completed"}
        ),
        None,
    )
    if existing:
        return {
            "job_id": existing["id"],
            "state": existing["state"],
            "deduplicated": True,
        }
    if parse_queue.full():
        raise HTTPException(429, "parse_queue_full")
    path.write_bytes(content)
    job = {"id": uuid.uuid4().hex, "document": name}
    jobs[job["id"]] = {**job, "sha256": digest, "state": "queued"}
    parse_queue.put_nowait(job)
    return {"job_id": job["id"], "state": "queued"}


@app.get("/api/jobs")
def listjobs():
    return list(jobs.values())


@app.post("/ask")
async def ask(item: Ask):
    metrics["requests"] += 1
    key = (item.mode, item.question.strip())
    old = cache.get(key) if item.use_cache else None
    if old is not None:
        metrics["cache_hits"] += 1
        return {"answer": old, "cached": True}
    try:
        async with capacity:
            if item.mode == "generation":
                result = await asyncio.to_thread(
                    pipeline.ask, item.question, "CN100342976C.pdf", True
                )
                answer = result["answer"]
            else:
                _, hits = (
                    None,
                    await asyncio.to_thread(
                        pipeline.retrieve, item.question, "CN100342976C.pdf", True
                    ),
                )
                answer = "\n".join(h["text"] for h in hits)
                result = {"mode": "retrieval_evidence", "sources": hits}
            cache.put(key, answer)
            return {"answer": answer, "cached": False, "result": result}
    except OverflowError:
        metrics["queue_rejects"] += 1
        raise HTTPException(429, "request queue full")
