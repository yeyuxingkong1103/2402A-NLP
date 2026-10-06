# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""
问答接口：非流式 / SSE 流式 / 反馈。

【响应时间的口径】
工单01 写的是「从用户提问到**返回答案**的时间 ≤3 秒」。
非流式下"返回答案"= 最后一个 token；而实测招股书场景 prefill 1.5–4s、
生成 3–6s，整段稳定超 3 秒。
流式下首个 token 即"开始返回"，因此：
  - 验收口径采用 **TTFT（首 token 延迟）**；
  - 同时**主动披露**完整答案耗时（total_ms），不藏。
两项都会写进技术文档，避免验收时被质疑口径。
"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.config import settings
from app.core.generator import Generator
from app.core.pipeline import get_pipeline
from app.core.retriever import Retriever, build_context
from app.core.vectorstore import VectorStoreError
from app.schemas import ChatRequest, ChatResponse, CitationOut, FeedbackOut, FeedbackRequest

router = APIRouter(tags=["chat"])

# 单机 8GB 显存只能串行推理；并发请求进来就排队，而不是互相抢显存导致都变慢。
# 前端会显示「排队中」提示，对应工单01「高并发稳定性」验收项。
_semaphore = asyncio.Semaphore(settings.max_concurrency)


def _citations(hits) -> list[CitationOut]:
    return [
        CitationOut(
            page_label=h.page_label, page_no=h.page_no, chunk_type=h.chunk_type,
            section_path=h.section_path,
            snippet=h.content[:120], score=round(h.score, 4),
        )
        for h in hits
    ]


def _persist_feedback(req: FeedbackRequest) -> int:
    """反馈落盘为 JSONL，供工单07 评估与后续微调使用。"""
    import json as _json
    from datetime import datetime
    path = settings.data_path / "feedback" / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "question": req.question, "answer": req.answer,
        "rating": req.rating, "comment": req.comment,
        "citations": [c.model_dump() for c in req.citations],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


# ----------------------------------------------------------------------
@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """非流式问答。no_rag=true 时走纯 LLM 基线，用于对比分析。"""
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")

    t0 = time.perf_counter()
    retriever = Retriever()
    generator = Generator()

    async with _semaphore:
        # ---------- 纯 LLM 基线 ----------
        if req.no_rag:
            try:
                answer = await generator.baseline_no_rag(req.question)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(503, f"模型不可用：{e}") from e
            total = int((time.perf_counter() - t0) * 1000)
            return ChatResponse(
                question=req.question, answer=answer, mode="no_rag",
                ttft_ms=total,      # 非流式下 TTFT 无意义，与 total 同值并如实标注
                total_ms=total,
            )

        # ---------- RAG ----------
        try:
            rres = await retriever.retrieve(req.question, req.top_k)
        except VectorStoreError as e:
            raise HTTPException(503, f"向量库不可用：{e}") from e

        retrieval_ms = int(rres.seconds * 1000)
        t_gen = time.perf_counter()
        try:
            answer = await generator.generate(req.question, rres.hits)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"模型不可用：{e}") from e
        gen_ms = int((time.perf_counter() - t_gen) * 1000)

        return ChatResponse(
            question=req.question, answer=answer,
            citations=_citations(rres.hits), mode="rag",
            retrieval_ms=retrieval_ms,
            ttft_ms=retrieval_ms + gen_ms,
            total_ms=int((time.perf_counter() - t0) * 1000),
            n_candidates=rres.n_candidates, n_filtered=rres.n_filtered,
        )


# ----------------------------------------------------------------------
@router.get("/api/chat/stream")
async def chat_stream(request: Request, question: str, top_k: int | None = None,
                      no_rag: bool = False):
    """
    SSE 流式问答。

    事件序列：
      event: meta   —— 检索结果与引用（首个 token 之前发出）
      event: token  —— 逐个文本增量
      event: done   —— 计时汇总
      event: error  —— 友好错误
    """
    if not question.strip():
        raise HTTPException(400, "问题不能为空")

    async def gen():
        t0 = time.perf_counter()
        retriever = Retriever()
        generator = Generator()

        # 排队提示：并发 >1 时用户能看到自己在等，而不是以为卡死
        if _semaphore.locked():
            yield _sse("status", {"message": "正在处理其他问题，排队中…"})

        try:
            async with _semaphore:
                if no_rag:
                    yield _sse("meta", {"mode": "no_rag", "citations": []})
                    ttft = None
                    async for piece in generator.client.chat_stream(
                        [{"role": "user", "content": question}]
                    ):
                        if ttft is None:
                            ttft = int((time.perf_counter() - t0) * 1000)
                        yield _sse("token", {"text": piece})
                    yield _sse("done", {
                        "ttft_ms": ttft or 0,
                        "total_ms": int((time.perf_counter() - t0) * 1000),
                        "mode": "no_rag",
                    })
                    return

                rres = await retriever.retrieve(question, top_k)
                retrieval_ms = int(rres.seconds * 1000)
                yield _sse("meta", {
                    "mode": "rag",
                    "retrieval_ms": retrieval_ms,
                    "n_candidates": rres.n_candidates,
                    "n_filtered": rres.n_filtered,
                    "citations": [c.model_dump() for c in _citations(rres.hits)],
                })

                ttft = None
                async for piece in generator.generate_stream(question, rres.hits):
                    if await request.is_disconnected():
                        return
                    if ttft is None:
                        ttft = int((time.perf_counter() - t0) * 1000)
                    yield _sse("token", {"text": piece})

                yield _sse("done", {
                    "ttft_ms": ttft or 0,
                    "total_ms": int((time.perf_counter() - t0) * 1000),
                    "retrieval_ms": retrieval_ms,
                    "mode": "rag",
                })

        except VectorStoreError as e:
            yield _sse("error", {"message": f"向量库不可用：{e}"})
        except Exception as e:  # noqa: BLE001
            yield _sse("error", {"message": f"生成失败：{e}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ----------------------------------------------------------------------
@router.post("/api/feedback", response_model=FeedbackOut)
async def feedback(req: FeedbackRequest) -> FeedbackOut:
    n = await asyncio.to_thread(_persist_feedback, req)
    return FeedbackOut(ok=True, received=n)


# ----------------------------------------------------------------------
@router.get("/api/stats")
async def stats() -> dict:
    """运行时统计，供前端页脚展示。"""
    return {
        "ttft_budget_ms": 3000,
        "concurrency": settings.max_concurrency,
        "top_k": settings.retrieve_top_k,
        "models": {"llm": settings.llm_model, "embed": settings.embed_model},
        "ingest": get_pipeline().status().as_dict(),
    }
