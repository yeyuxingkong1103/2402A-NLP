# -*- coding: utf-8 -*-
"""
基线（问题）版 RAG API —— 复现工单描述的故障，用于优化前后对比。
故意保留的问题：
1) 每个请求都“重新初始化”重量级组件（模拟 DeepDoc/VLM 初始化延迟与内存）；
2) 全局缓存无上限，随运行持续增长（内存泄漏）；
3) 向量检索每次新建连接且不归还（连接池泄漏）；
4) LLM 推理无限流，高并发时线程堆积、延迟飙升；
5) 中间大对象不释放。
运行：PYTHONPATH=. uvicorn src.api_baseline:app --port 8018
"""
import os, time, uuid
import psutil
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="RAG API (基线版)")
process = psutil.Process(os.getpid())

# 问题点2：无界全局缓存（会无限增长）
LEAKY_CACHE = {}
# 问题点3：模拟“连接泄漏”计数
LEAKED_CONNECTIONS = []


def heavy_init():
    """问题点1：模拟每请求初始化重量级组件（耗时+分配内存）。"""
    time.sleep(0.3)
    return bytearray(2 * 1024 * 1024)  # 2MB 中间对象，且不释放（被列表持有）


def vector_search(q):
    # 问题点3：每次新建“连接”且不归还到池
    conn = {"id": uuid.uuid4().hex, "buf": bytearray(512 * 1024)}
    LEAKED_CONNECTIONS.append(conn)  # 永不释放 → 连接/内存泄漏
    return [f"doc-about-{q[:8]}"]


def llm_generate(prompt):
    # 问题点4：无并发限流，直接 sleep 模拟推理（高并发下堆积）
    time.sleep(0.8)
    return "基线回答: " + prompt[:20]


class ChatReq(BaseModel):
    question: str
    chat_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok", "rss_mb": round(process.memory_info().rss / 1024 / 1024, 1),
            "leaked_connections": len(LEAKED_CONNECTIONS), "cache_size": len(LEAKY_CACHE)}


@app.post("/api/v1/chats_openai/{chat_id}/chat/completions")
def chat(chat_id: str, req: ChatReq):
    heavy_init()                      # 每请求重复初始化
    docs = vector_search(req.question)
    answer = llm_generate(req.question + "|" + " ".join(docs))
    LEAKY_CACHE[req.question + uuid.uuid4().hex] = answer  # 无界缓存（key 带 uuid 永不去重）
    return JSONResponse({"chat_id": chat_id, "question": req.question,
                         "answer": answer, "docs": docs})
