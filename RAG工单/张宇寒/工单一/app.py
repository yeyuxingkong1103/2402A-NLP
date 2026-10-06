"""FastAPI 问答服务。"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag import CHUNKS_PATH, INDEX_PATH, MODEL_PATH, DeepSeek, RAGSystem, normalize_question


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

app = FastAPI(title="招股说明书 RAG 问答")
_rag: RAGSystem | None = None
_feedback: dict[str, str] = {}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class FeedbackRequest(BaseModel):
    answer_id: str
    helpful: bool


def get_rag() -> RAGSystem:
    global _rag
    if _rag is None:
        _rag = RAGSystem(os.getenv("DEEPSEEK_API_KEY", "").strip())
    return _rag


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/api/health")
def health():
    return {
        "deepseek_key": bool(os.getenv("DEEPSEEK_API_KEY", "").strip()),
        "mineru_key": bool(os.getenv("MINERU_API_KEY", "").strip()),
        "model": MODEL_PATH.exists(),
        "index": INDEX_PATH.exists() and CHUNKS_PATH.exists(),
    }


@app.post("/api/ask")
async def ask(payload: AskRequest):
    try:
        question = normalize_question(payload.question)
        result = await get_rag().ask(question)
        result["answer_id"] = uuid.uuid4().hex
        return result
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None


def _comparison(rag_result: dict, llm_answer: str) -> str:
    if not rag_result.get("citations"):
        return "RAG 未检索到足够证据，不应相信无文档依据的猜测。"
    if rag_result["answer"].strip() == llm_answer.strip():
        return "两个答案基本一致，但 RAG 答案额外提供了 PDF 页码和原文依据。"
    return "RAG 答案来自 PDF 检索片段并有页码依据；纯 LLM 答案未使用 PDF，存在知识缺失或编造风险。"


async def _run_one(item: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        system = get_rag()
        baseline_start = time.perf_counter()
        rag_result, llm_answer = await asyncio.gather(
            system.ask(item["question"]),
            system.llm.pure_answer(item["question"]),
        )
        baseline_ms = (time.perf_counter() - baseline_start) * 1000
        return {
            "id": item["id"],
            "question": item["question"],
            "rag": rag_result,
            "llm_answer": llm_answer,
            "llm_elapsed_ms": round(baseline_ms, 1),
            "comparison": _comparison(rag_result, llm_answer),
        }


@app.post("/api/benchmark")
async def benchmark():
    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    semaphore = asyncio.Semaphore(3)
    jobs = [_run_one(item, semaphore) for item in questions]
    results = await asyncio.gather(*jobs, return_exceptions=True)
    output = []
    for item, result in zip(questions, results):
        if isinstance(result, Exception):
            output.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "error": f"评测失败：{type(result).__name__}",
                }
            )
        else:
            output.append(result)
    return {"results": output}


@app.post("/api/feedback")
def feedback(payload: FeedbackRequest):
    _feedback[payload.answer_id] = "helpful" if payload.helpful else "not_helpful"
    return {"saved": True, "count": len(_feedback)}

