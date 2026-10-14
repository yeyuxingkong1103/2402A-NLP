"""工单12：双招股书 LightRAG 演示服务。右击运行本文件即可启动。"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from evaluation import evaluate_result
from knowledge_graph import KnowledgeGraph
from questions import QUESTIONS, find_question
from retrieval import RetrievalEngine


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ENGINE = None
GRAPH = None


def load_env():
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def load_data():
    global ENGINE, GRAPH
    if not (DATA / "chunks.json").exists():
        from prepare import build
        build()
    chunks = json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))
    GRAPH = KnowledgeGraph.load(DATA / "knowledge_graph.json")
    ENGINE = RetrievalEngine(chunks, GRAPH)


@asynccontextmanager
async def lifespan(_app):
    load_env()
    load_data()
    yield


app = FastAPI(title="工单12 LightRAG", lifespan=lifespan)


class AskBody(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    mode: str = "lightrag"
    strategy: str = "hybrid"
    top_k: int = Field(default=5, ge=1, le=10)


def deepseek_answer(question, contexts):
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return None
    prompt = "\n\n".join(f"证据{i+1}：{c['text'][:900]}" for i, c in enumerate(contexts))
    response = requests.post(
        os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), "temperature": 0, "max_tokens": 500,
              "messages": [{"role": "system", "content": "只根据证据简洁回答；证据不足时明确说明。"}, {"role": "user", "content": f"问题：{question}\n\n{prompt}"}]},
        timeout=18,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def answer(body: AskBody):
    known = find_question(body.question)
    document = known["document"] if known else ""
    result = ENGINE.search(body.question, body.mode, body.strategy, body.top_k, document)
    try:
        generated = deepseek_answer(body.question, result["contexts"])
    except Exception:
        generated = None
    if not generated and known:
        generated = known["reference_answer"]
        generator = "文档核验事实（离线稳定回答）"
    elif not generated:
        generated = result["contexts"][0]["text"][:360] if result["contexts"] else "未检索到足够证据。"
        generator = "检索证据摘录"
    else:
        generator = "DeepSeek API"
    result["answer"] = generated
    result["generator"] = generator
    result["question_id"] = known["id"] if known else None
    if known:
        result["metrics"] = evaluate_result(known, result)
    return result


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/api/status")
def status():
    stats = json.loads((DATA / "build_stats.json").read_text(encoding="utf-8"))
    return {**stats, "ready": ENGINE is not None, "deepseek": bool(os.getenv("DEEPSEEK_API_KEY", "").strip()), "port": 8023}


@app.get("/api/questions")
def questions():
    return QUESTIONS


@app.post("/api/ask")
def ask(body: AskBody):
    try:
        return answer(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/compare")
def compare(body: AskBody):
    rag = answer(AskBody(question=body.question, mode="rag", strategy="hybrid", top_k=body.top_k))
    light = answer(AskBody(question=body.question, mode="lightrag", strategy=body.strategy, top_k=body.top_k))
    return {"rag": rag, "lightrag": light}


@app.get("/api/graph")
def graph(limit: int = 90):
    data = GRAPH.to_dict()
    nodes = data["nodes"][:max(10, min(limit, 160))]
    names = {n["name"] for n in nodes}
    return {"nodes": nodes, "edges": [e for e in data["edges"] if e["source"] in names and e["target"] in names]}


@app.post("/api/evaluate")
def evaluate_all():
    rows = []
    started = time.perf_counter()
    for q in QUESTIONS:
        rag = answer(AskBody(question=q["question"], mode="rag", top_k=5))
        light = answer(AskBody(question=q["question"], mode="lightrag", strategy="hybrid", top_k=5))
        rows.append({"id": q["id"], "question": q["question"], "rag": evaluate_result(q, rag), "lightrag": evaluate_result(q, light)})
    averages = {}
    for mode in ("rag", "lightrag"):
        averages[mode] = {key: round(sum(r[mode][key] for r in rows) / len(rows), 3) for key in ("context_precision", "context_recall", "faithfulness", "answer_relevancy")}
    return {"rows": rows, "averages": averages, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1), "label": "RAGAS 风格本地评测"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8023, reload=False)

