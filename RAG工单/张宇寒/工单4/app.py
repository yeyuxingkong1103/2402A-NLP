"""FastAPI 问答服务。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag import CHUNKS_PATH, INDEX_PATH, LEXICAL_PATH, MODEL_PATH, RAGSystem, normalize_question
from prepare import DOCUMENTS


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

app = FastAPI(title="双招股说明书 RAG 问答")
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


@app.on_event("startup")
async def warm_up():
    """索引已存在时在启动期加载模型，避免首个用户请求超时。"""
    if (
        os.getenv("DEEPSEEK_API_KEY", "").strip()
        and INDEX_PATH.exists()
        and CHUNKS_PATH.exists()
    ):
        await asyncio.to_thread(get_rag)


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/api/pdf/{document}")
def source_pdf(document: str):
    source = next((item["pdf_path"] for item in DOCUMENTS if item["document"] == document), None)
    if source is None or not source.exists():
        raise HTTPException(status_code=404, detail="PDF 不存在")
    return FileResponse(source, media_type="application/pdf")


@app.get("/api/health")
def health():
    return {
        "deepseek_key": bool(os.getenv("DEEPSEEK_API_KEY", "").strip()),
        "mineru_key": bool(os.getenv("MINERU_API_KEY", "").strip()),
        "model": MODEL_PATH.exists(),
        "index": INDEX_PATH.exists() and CHUNKS_PATH.exists(),
        "hybrid_index": LEXICAL_PATH.exists(),
        "documents": 2,
        "ready": bool(os.getenv("DEEPSEEK_API_KEY", "").strip())
        and MODEL_PATH.exists()
        and INDEX_PATH.exists()
        and CHUNKS_PATH.exists(),
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


def _score_answer(answer: str, expected_facts: list[list[str]]) -> dict:
    """按关键事实命中数评分，避免用主观文本相似度冒充准确率。"""
    normalized = re.sub(r"[\s,，。；;：:、《》()（）]", "", answer.lower())
    matched = 0
    for alternatives in expected_facts:
        if any(
            re.sub(r"[\s,，。；;：:、《》()（）]", "", fact.lower()) in normalized
            for fact in alternatives
        ):
            matched += 1
    total = len(expected_facts)
    return {
        "score": round(matched * 100 / total, 1) if total else 0.0,
        "matched": matched,
        "total": total,
    }


async def _run_one(item: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        system = get_rag()
        baseline_start = time.perf_counter()

        async def safe_baseline() -> str:
            try:
                return await system.llm.pure_answer(item["question"])
            except RuntimeError:
                return "纯 LLM 请求超时或失败。"

        rag_result, llm_answer = await asyncio.gather(
            system.ask(item["question"]), safe_baseline()
        )
        baseline_ms = (time.perf_counter() - baseline_start) * 1000
        return {
            "case_id": item["case_id"],
            "id": item["id"],
            "document": item["document"],
            "question": item["question"],
            "rag": rag_result,
            "llm_answer": llm_answer,
            "llm_elapsed_ms": round(baseline_ms, 1),
            "comparison": _comparison(rag_result, llm_answer),
            "reference_answer": item["reference_answer"],
            "rag_score": _score_answer(rag_result["answer"], item["expected_facts"]),
            "llm_score": _score_answer(llm_answer, item["expected_facts"]),
        }


@app.post("/api/benchmark")
async def benchmark():
    started = time.perf_counter()
    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    semaphore = asyncio.Semaphore(3)
    jobs = [_run_one(item, semaphore) for item in questions]
    results = await asyncio.gather(*jobs, return_exceptions=True)
    output = []
    for item, result in zip(questions, results):
        if isinstance(result, Exception):
            output.append(
                {
                    "case_id": item["case_id"],
                    "id": item["id"],
                    "document": item["document"],
                    "question": item["question"],
                    "error": f"评测失败：{type(result).__name__}",
                    "reference_answer": item["reference_answer"],
                    "rag_score": {"score": 0.0, "matched": 0, "total": len(item["expected_facts"])},
                    "llm_score": {"score": 0.0, "matched": 0, "total": len(item["expected_facts"])},
                }
            )
        else:
            output.append(result)
    rag_accuracy = round(sum(x["rag_score"]["score"] for x in output) / len(output), 1)
    llm_accuracy = round(sum(x["llm_score"]["score"] for x in output) / len(output), 1)
    by_document = {}
    for document in ("prospectus_1", "prospectus_2"):
        document_results = [item for item in output if item["document"] == document]
        by_document[document] = round(
            sum(item["rag_score"]["score"] for item in document_results)
            / len(document_results),
            1,
        )
    return {
        "results": output,
        "summary": {
            "rag_accuracy": rag_accuracy,
            "llm_accuracy": llm_accuracy,
            "target": 90,
            "target_passed": rag_accuracy >= 90,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "question_count": len(output),
            "by_document": by_document,
        },
    }


@app.post("/api/feedback")
def feedback(payload: FeedbackRequest):
    _feedback[payload.answer_id] = "helpful" if payload.helpful else "not_helpful"
    return {"saved": True, "count": len(_feedback)}
