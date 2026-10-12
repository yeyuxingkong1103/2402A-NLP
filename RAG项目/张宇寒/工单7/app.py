from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from evaluation import evaluate_all, load_questions
from retrieval import DATA_DIR as RETRIEVAL_DATA_DIR, RAGSystem


ROOT = Path(__file__).resolve().parent
DATA_DIR = RETRIEVAL_DATA_DIR
REPORT_PATH = ROOT / "evaluation_report.json"
DEFAULT_CORPUS_DIR = Path(r"C:\Users\lenovo\Desktop\RAG 工单 (1)\RAG 工单\附件\ccf_competition")
load_dotenv(ROOT / ".env")


class EvaluationRequest(BaseModel):
    top_k: int = Field(default=5, ge=3, le=10)
    mode: str = Field(default="hybrid", pattern="^(vector|fulltext|hybrid)$")


class AskRequest(EvaluationRequest):
    question: str = Field(min_length=2, max_length=500)


@lru_cache(maxsize=1)
def get_rag() -> RAGSystem:
    return RAGSystem()


app = FastAPI(title="工单7：CCF年报RAG测试与检索评估")


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/api/health")
def health():
    required = [DATA_DIR / "chunks.json", DATA_DIR / "lexical.pkl", DATA_DIR / "index.faiss"]
    ready = all(path.exists() for path in required)
    corpus_dir = Path(os.getenv("CCF_DATA_DIR", str(DEFAULT_CORPUS_DIR)))
    return {
        "ok": True,
        "index_ready": ready,
        "corpus_ready": (corpus_dir / "txt").is_dir(),
        "llm_ready": bool(os.getenv("DEEPSEEK_API_KEY", "").strip()),
        "reports": 9,
        "questions": 10,
        "message": "索引已就绪" if ready else "尚未建立索引，请先运行 prepare.py",
    }


@app.get("/api/evaluation/questions")
def evaluation_questions():
    questions = load_questions()
    return {"count": len(questions), "questions": questions}


@app.post("/api/evaluation/run")
async def run_evaluation(payload: EvaluationRequest):
    try:
        report = await evaluate_all(get_rag(), top_k=payload.top_k, mode=payload.mode)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


@app.get("/api/evaluation/export")
def export_evaluation():
    if not REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="尚无评测报告，请先运行10题评测")
    return FileResponse(REPORT_PATH, media_type="application/json", filename="工单7_RAG评测报告.json")


@app.post("/api/ask")
async def ask(payload: AskRequest):
    try:
        return await get_rag().answer(payload.question, top_k=payload.top_k, mode=payload.mode)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/pdf/{stock_code}")
def source_pdf(stock_code: str):
    corpus_dir = Path(os.getenv("CCF_DATA_DIR", str(DEFAULT_CORPUS_DIR)))
    matches = list((corpus_dir / "pdf").glob(f"*{stock_code}*.pdf"))
    if not matches:
        raise HTTPException(status_code=404, detail="未找到对应PDF")
    return FileResponse(matches[0], media_type="application/pdf", filename=f"{stock_code}.pdf")


def run_server() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8018)


if __name__ == "__main__":
    run_server()
