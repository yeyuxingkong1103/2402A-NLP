from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dataset import load_jsonl
from embedding_model import DomainEmbeddingModel
from train import TrainingConfig, run_training


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
PORT = 8022

app = FastAPI(title="工单11：金融领域Embedding微调", version="1.0.0")


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=10)
    mode: str = "after"


@lru_cache(maxsize=1)
def _model() -> DomainEmbeddingModel:
    return DomainEmbeddingModel.load(MODEL_DIR / "domain_embedding.pkl")


@lru_cache(maxsize=1)
def _chunks() -> list[dict]:
    return json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def _document_vectors(adapted: bool) -> np.ndarray:
    return _model().encode([row["text"] for row in _chunks()], adapted=adapted)


def _read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


@app.get("/")
async def home():
    return FileResponse(ROOT / "index.html")


@app.get("/api/status")
async def status():
    summary = _read_json(DATA_DIR / "dataset_summary.json", {})
    report = _read_json(MODEL_DIR / "evaluation_report.json", {})
    ready = (MODEL_DIR / "domain_embedding.pkl").is_file() and bool(report)
    steps = [
        {"id": 1, "name": "数据集生成", "done": bool(summary)},
        {"id": 2, "name": "问答对与三元组生成", "done": (DATA_DIR / "qa_pairs.jsonl").is_file()},
        {"id": 3, "name": "数据集与基础模型加载", "done": ready},
        {"id": 4, "name": "Triplet Loss与训练参数", "done": bool(report.get("training"))},
        {"id": 5, "name": "创建检索评估器", "done": bool(report.get("before"))},
        {"id": 6, "name": "微调前后评估", "done": bool(report.get("after"))},
    ]
    return {"ready": ready, "message": "领域Embedding模型已就绪" if ready else "请先运行prepare.py和train.py", "dataset": summary, "evaluation": report, "steps": steps}


@app.get("/api/examples")
async def examples():
    path = DATA_DIR / "qa_pairs.jsonl"
    return {"items": load_jsonl(path)[:5]} if path.is_file() else {"items": []}


@app.post("/api/search")
async def search(payload: SearchRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="查询不能为空")
    if payload.mode not in {"before", "after"}:
        raise HTTPException(status_code=400, detail="mode只能是before或after")
    if not (MODEL_DIR / "domain_embedding.pkl").is_file():
        raise HTTPException(status_code=503, detail="模型尚未训练")
    started = time.perf_counter()
    adapted = payload.mode == "after"
    query_vector = _model().encode([query], adapted=adapted)[0]
    scores = _document_vectors(adapted) @ query_vector
    order = np.argsort(-scores, kind="stable")[: payload.top_k]
    hits = []
    chunks = _chunks()
    for index in order:
        row = chunks[int(index)]
        hits.append(
            {
                "chunk_id": row["chunk_id"],
                "document": row.get("document", ""),
                "company": row.get("company", ""),
                "page": row.get("page"),
                "text": row["text"][:700],
                "score": round(float(scores[int(index)]), 4),
            }
        )
    return {"query": query, "mode": payload.mode, "hits": hits, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}


@app.post("/api/train")
async def train_model(epochs: int = 60):
    if not 1 <= epochs <= 300:
        raise HTTPException(status_code=400, detail="epochs必须在1到300之间")
    train_records = load_jsonl(DATA_DIR / "train.jsonl")
    evaluation_records = load_jsonl(DATA_DIR / "evaluation.jsonl")
    documents = [{"chunk_id": row["chunk_id"], "text": row["text"]} for row in _chunks()]
    report = run_training(
        train_records,
        evaluation_records,
        documents,
        MODEL_DIR,
        TrainingConfig(epochs=epochs, top_k=3),
    )
    _model.cache_clear()
    _document_vectors.cache_clear()
    return {"status": "trained", "report": report}


def run_server() -> None:
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=False)


if __name__ == "__main__":
    run_server()

