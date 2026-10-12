"""FastAPI 问答服务。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from conversation import ConversationManager
from feedback import FeedbackStore
from rag import CHUNKS_PATH, INDEX_PATH, LEXICAL_PATH, MODEL_PATH, RAGSystem, SearchConfig, normalize_question
from prepare import DOCUMENTS


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

_rag: RAGSystem | None = None
_feedback: dict[str, str] = {}
_feedback_store = FeedbackStore(ROOT / "feedback.json")
_conversations = ConversationManager()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    session_id: str | None = Field(default=None, max_length=64)
    retrieval_mode: Literal["vector", "fulltext", "hybrid"] = "hybrid"
    reranker: Literal["none", "tfidf", "llm", "adaptive"] = "tfidf"
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    top_k: int = Field(default=8, ge=3, le=10)


class FeedbackRequest(BaseModel):
    answer_id: str
    helpful: bool
    chunk_ids: list[str] = Field(default_factory=list, max_length=10)


class EvaluationRequest(BaseModel):
    retrieval_mode: Literal["vector", "fulltext", "hybrid"] = "hybrid"
    reranker: Literal["none", "tfidf", "llm", "adaptive"] = "tfidf"
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    top_k: int = Field(default=8, ge=3, le=10)


def get_rag() -> RAGSystem:
    global _rag
    if _rag is None:
        _rag = RAGSystem(os.getenv("DEEPSEEK_API_KEY", "").strip(), _feedback_store)
    return _rag


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """索引已存在时在启动期加载模型，避免首个用户请求超时。"""
    if (
        os.getenv("DEEPSEEK_API_KEY", "").strip()
        and INDEX_PATH.exists()
        and CHUNKS_PATH.exists()
    ):
        await asyncio.to_thread(get_rag)
    yield


app = FastAPI(title="工单六：双招股说明书多策略 RAG 问答", lifespan=lifespan)


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
        "retrieval_modes": ["vector", "fulltext", "hybrid"],
        "rerankers": ["none", "tfidf", "llm", "adaptive"],
        "ready": bool(os.getenv("DEEPSEEK_API_KEY", "").strip())
        and MODEL_PATH.exists()
        and INDEX_PATH.exists()
        and CHUNKS_PATH.exists(),
    }


@app.post("/api/ask")
async def ask(payload: AskRequest):
    try:
        question = normalize_question(payload.question)
        session_id = payload.session_id or uuid.uuid4().hex
        resolved = _conversations.resolve(question, session_id)
        config = SearchConfig(
            payload.retrieval_mode, payload.reranker, payload.vector_weight, payload.top_k
        )
        result = await get_rag().ask(resolved, config)
        _conversations.remember(session_id, resolved)
        result["session_id"] = session_id
        result["original_question"] = question
        result["resolved_question"] = resolved
        result["answer_id"] = uuid.uuid4().hex
        result.setdefault(
            "search_config",
            {
                "mode": config.mode,
                "reranker": config.reranker,
                "vector_weight": config.vector_weight,
                "top_k": config.top_k,
            },
        )
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


def _score_retrieval(evidence: str, expected_facts: list[list[str]]) -> bool:
    """召回内容命中任一预期事实组即视为该题被召回。"""
    return _score_answer(evidence, expected_facts)["matched"] > 0


def _summarize_strategy(rows: list[dict]) -> dict:
    if not rows:
        return {"accuracy": 0.0, "recall": 0.0, "average_ms": 0.0, "question_count": 0}
    count = len(rows)
    return {
        "accuracy": round(sum(row["answer_score"] for row in rows) / count, 1),
        "recall": round(sum(bool(row["recalled"]) for row in rows) * 100 / count, 1),
        "average_ms": round(sum(row["elapsed_ms"] for row in rows) / count, 1),
        "question_count": count,
    }


async def _evaluate_strategy(questions: list[dict], config: SearchConfig) -> tuple[list[dict], dict]:
    semaphore = asyncio.Semaphore(3)

    async def run(item: dict) -> dict:
        async with semaphore:
            try:
                result = await get_rag().ask(item["question"], config)
                evidence = " ".join(citation.get("text", "") for citation in result.get("citations", []))
                return {
                    "case_id": item["case_id"],
                    "answer_score": _score_answer(result.get("answer", ""), item["expected_facts"])["score"],
                    "recalled": _score_retrieval(evidence, item["expected_facts"]),
                    "elapsed_ms": float(result.get("total_ms", 0.0)),
                    "error": "",
                }
            except Exception as exc:
                return {
                    "case_id": item["case_id"], "answer_score": 0.0,
                    "recalled": False, "elapsed_ms": 0.0,
                    "error": f"{type(exc).__name__}",
                }

    rows = await asyncio.gather(*(run(item) for item in questions))
    return rows, _summarize_strategy(rows)


@app.post("/api/evaluate-retrieval")
async def evaluate_retrieval(payload: EvaluationRequest):
    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    baseline_config = SearchConfig("hybrid", "none", 0.72, payload.top_k)
    optimized_config = SearchConfig(
        payload.retrieval_mode, payload.reranker, payload.vector_weight, payload.top_k
    )
    baseline_rows, baseline = await _evaluate_strategy(questions, baseline_config)
    optimized_rows, optimized = await _evaluate_strategy(questions, optimized_config)
    return {
        "baseline": baseline,
        "optimized": optimized,
        "delta": {
            "accuracy": round(optimized["accuracy"] - baseline["accuracy"], 1),
            "recall": round(optimized["recall"] - baseline["recall"], 1),
            "average_ms": round(optimized["average_ms"] - baseline["average_ms"], 1),
        },
        "targets": {"accuracy": 90, "recall": 95},
        "results": {"baseline": baseline_rows, "optimized": optimized_rows},
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
    _feedback_store.record(payload.chunk_ids, payload.helpful)
    return {"saved": True, "count": len(_feedback)}


def run_server() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8017)


if __name__ == "__main__":
    run_server()
