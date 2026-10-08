# -*- coding: utf-8 -*-
"""
工单17 压测用 API 服务
用 FastAPI 包装 RAGQASystem，暴露 /api/chat，供 locust 压测
"""
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_server")

app = FastAPI(title="RAG QA API")

# ===== Prometheus 指标 =====
REQUEST_COUNT = Counter("rag_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds", "Request latency", ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
)
MEMORY_USAGE = Gauge("rag_process_memory_bytes", "Process memory bytes")
GPU_MEMORY = Gauge("rag_gpu_memory_bytes", "GPU memory bytes")

# ===== 请求模型 =====
class ChatRequest(BaseModel):
    query: str
    chat_id: str = "default"
    stream: bool = False

# ===== 全局单例 RAG 系统（工单17优化：避免每次请求 new） =====
_rag_system = None

def get_rag_system():
    global _rag_system
    if _rag_system is None:
        from rag_qa_system import RAGQASystem
        import glob
        logger.info("Initializing RAGQASystem (singleton)...")
        # 工单17压测用：加载 ccf_competition 里的年度报告 PDF
        pdf_paths = sorted(glob.glob("/root/autodl-tmp/data/ccf_competition/pdf/*.pdf"))[:1]
        if not pdf_paths:
            pdf_paths = sorted(glob.glob("/root/autodl-tmp/data/*.pdf"))
        logger.info(f"Found {len(pdf_paths)} PDFs")
        for p in pdf_paths:
            logger.info(f"  - {p}")
        if not pdf_paths:
            raise RuntimeError("No PDF found")
        _rag_system = RAGQASystem(pdf_paths=pdf_paths)
        logger.info("RAGQASystem initialized")
    return _rag_system

# ===== 启动时预加载 =====
@app.on_event("startup")
async def startup_event():
    logger.info("Preloading RAG system...")
    get_rag_system()
    logger.info("RAG system ready")

# ===== 路由 =====
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    import psutil, torch
    try:
        MEMORY_USAGE.set(psutil.Process().memory_info().rss)
        if torch.cuda.is_available():
            GPU_MEMORY.set(torch.cuda.memory_allocated())
    except Exception as e:
        logger.warning(f"metrics collect error: {e}")
    return generate_latest()

@app.post("/api/chat")
def chat(req: ChatRequest):
    endpoint = "/api/chat"
    start = time.time()
    try:
        system = get_rag_system()
        result = system.answer(req.query, use_rag=True)
        elapsed = time.time() - start
        REQUEST_LATENCY.labels(endpoint).observe(elapsed)
        REQUEST_COUNT.labels(endpoint, "200").inc()
        return {
            "answer": result.get("answer", ""),
            "response_time": result.get("response_time", elapsed),
            "query": req.query,
        }
    except Exception as e:
        REQUEST_COUNT.labels(endpoint, "500").inc()
        logger.exception("chat error")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
