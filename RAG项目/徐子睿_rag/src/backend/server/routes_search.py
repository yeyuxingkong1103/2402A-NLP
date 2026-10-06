# -*- coding: utf-8 -*-
"""server/routes_search.py —— 检索与问答接口。

在链路中的位置：
    浏览器 → 【本文件】 → backend/retrieval 的检索链路 → backend/server/answer 生成

三个接口：
    POST /api/search       纯检索，返回 trace / candidates / results（不生成答案）
    POST /api/ask          非流式问答，一次返回完整答案
    POST /api/ask/stream   流式问答（SSE），事件序列 start → trace… → retrieval_done → answer → done

流式接口为什么用"队列 + 后台线程"：
    检索链路是同步阻塞的（要调 Milvus 和 Ollama），而 SSE 生成器必须是可迭代的。
    把检索放到后台线程跑、用 queue 把事件传回生成器，每一段 trace 才能在产生时立刻推给前端，
    做出"链路逐步点亮"的效果；同步跑完再 yield 会让中间长时间无输出，被代理判定超时断连。
"""
from __future__ import annotations

import json
import queue
import threading
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

try:
    from ..retrieval import retrieve_with_trace
except ImportError:
    from retrieval import retrieve_with_trace

from .answer import generate_answer, retrieve_and_answer
from .config import LLM_MODEL, logger

router = APIRouter()

class SearchRequest(BaseModel):
    """检索请求体：q 问题，k 返回条数。"""

    q: str
    k: int = 5

@router.post("/api/search")
def search(request: SearchRequest):
    """纯检索接口：只做检索、不生成答案。

    用途：
        前端"检索"页签用它展示两路召回、RRF 融合和精排的全过程。

    为什么 candidate_k 用 max(request.k, 15)：
        精排是从融合后的候选里挑出最终 k 条。如果用户要 5 条却只融合出 5 个候选，
        精排没有挑选空间、等于没起作用。所以候选池至少要 15 条。
    """
    result = retrieve_with_trace(request.q, candidate_k=max(request.k, 15), final_k=request.k)
    return {
        "query": request.q,
        "results": result["results"],
        "note": result["note"],
        "trace": result["trace"],
        "rewritten_query": result["rewritten_query"],
        "candidates": result["candidates"],
    }

class AskRequest(BaseModel):
    """问答请求体：只有 q 一个问题字段。"""

    q: str

@router.post("/api/ask")
def ask(request: AskRequest):
    """非流式问答：一次返回完整答案。

    返回：
        answer / citations / trace / rewritten_query / note

    注意 trace 末尾手工补了一条 "LLM 生成"：
        retrieve_with_trace 只记录到"上下文组织"，但用户看到的时间线里
        "生成答案"也是耗时的一步，补上这条前端展示的链路才完整。
    """
    question = (request.q or "").strip()
    if not question:
        return {"answer": "请输入问题", "citations": [], "trace": [], "rewritten_query": ""}

    bundle, answer = retrieve_and_answer(question)
    return {
        "answer": answer["answer"],
        "citations": answer["citations"],
        "trace": bundle["trace"] + [{"step": "LLM 生成", "status": "done", "detail": LLM_MODEL, "ms": 0}],
        "rewritten_query": bundle["rewritten_query"],
        "note": bundle["note"],
    }

@router.post("/api/ask/stream")
def ask_stream(request: AskRequest):
    """流式问答：用 SSE 逐事件推送处理过程。

    返回：
        StreamingResponse，事件顺序为 start → trace×N → retrieval_done → answer → done
        （出错时是 error）。

    为什么用"队列 + 后台线程"这种写法：
        检索链路是同步阻塞的（要调 Milvus 和 Ollama），而 SSE 生成器必须是可迭代的。
        于是把检索放到后台线程跑，用 queue 把事件传回生成器 ——
        这样每一段 trace 都能在它产生的那一刻立刻推给前端，
        前端才能做出"链路一步步点亮"的效果，而不是等全部算完再一次性吐出来。
        若直接同步跑完再 yield，中间会有长时间无输出，浏览器和中间代理都可能判定超时断连。
    """
    question = (request.q or "").strip()
    if not question:
        return JSONResponse({"answer": "请输入问题", "citations": [], "trace": [], "rewritten_query": ""}, status_code=400)

    def event_stream():
        """SSE 生成器：不断从队列取事件并格式化成 SSE 帧。"""
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def send(event_type: str, data: dict[str, Any]) -> None:
            """把一个事件放进队列（供后台线程调用）。"""
            events.put({"type": event_type, "data": data})

        def run() -> None:
            """后台线程体：跑检索与生成，把每步结果塞进队列。"""
            try:
                send("start", {"question": question})
                # trace_callback 让检索链路的每一步都能即时推给前端
                bundle = retrieve_with_trace(
                    question,
                    candidate_k=15,
                    final_k=5,
                    char_budget=3600,
                    trace_callback=lambda event: send("trace", event),
                )
                send("retrieval_done", {"rewritten_query": bundle["rewritten_query"], "note": bundle["note"]})
                send("answer", generate_answer(question, bundle))
                send("done", {"rewritten_query": bundle["rewritten_query"], "note": bundle["note"]})
            except Exception as exc:
                logger.exception("stream request failed")
                send("error", {"message": str(exc)})
            finally:
                # 用 None 当结束哨兵：主循环收到它就关闭连接。
                # 放在 finally 里，保证异常路径也能正常收尾、不留下悬挂连接
                events.put(None)

        threading.Thread(target=run, daemon=True).start()
        while True:
            try:
                event = events.get(timeout=15)
            except queue.Empty:
                # 注释事件不会被前端处理，但可让 Cloudflare/Nginx 知道连接仍然存活。
                yield ": keep-alive\n\n"
                continue
            if event is None:
                return
            # ensure_ascii=False 让中文以原文推送（体积更小、前端无需额外解码）；
            # 每条 data 后必须跟两个换行，这是 SSE 协议的帧分隔要求
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # X-Accel-Buffering: no 是关键 —— 否则 Nginx 会缓冲整个响应，
        # 流式效果会被吃掉，变成"等全部结束才一次性显示"
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
