from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from evaluation import BENCHMARK_PATH, evaluate_hits, load_benchmarks
from graph_rag import DATA_DIR, GraphRAG
from questions import load_questions


ROOT = Path(__file__).resolve().parent
QUESTION_PATH = ROOT / "eval_questions.md"
PORT = 8021
load_dotenv(ROOT / ".env")
app = FastAPI(title="工单10：Docker金融GraphRAG服务")
_rag: GraphRAG | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class EvaluateRequest(BaseModel):
    question_id: int | None = Field(default=None, ge=1)
    question: str | None = Field(default=None, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


def get_rag() -> GraphRAG:
    global _rag
    if _rag is None:
        _rag = GraphRAG.load(DATA_DIR)
    return _rag


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/api/health")
def health():
    required = [DATA_DIR / "chunks.json", DATA_DIR / "graph.json", DATA_DIR / "tfidf.pkl"]
    ready = all(path.is_file() for path in required)
    return {
        "ready": ready,
        "deepseek_key": bool(os.getenv("DEEPSEEK_API_KEY", "").strip()),
        "message": "Graph RAG索引已就绪" if ready else "尚未建立索引，请先运行prepare.py",
    }


@app.get("/api/questions")
def question_list():
    questions = load_questions(QUESTION_PATH)
    return {"count": len(questions), "questions": questions}


@app.get("/api/benchmarks")
def benchmark_list():
    rows = load_benchmarks(BENCHMARK_PATH)
    return {"count": len(rows), "questions": rows}


def _evidence_answer(hits: list[dict]) -> str:
    if not hits:
        return "未检索到足够的年报证据。"
    lines = ["未配置DeepSeek API，以下为Graph RAG检索到的证据摘要："]
    for hit in hits[:5]:
        text = " ".join(hit["text"].split())
        lines.append(f"- {hit['document']}第{hit['page']}页：{text[:300]}")
    return "\n".join(lines)


async def _generate(question: str, hits: list[dict], api_key: str) -> str:
    evidence = "\n\n".join(
        f"[{hit['document']} 第{hit['page']}页]\n{hit['text']}" for hit in hits
    )
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = await client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.0,
        max_tokens=500,
        timeout=20,
        messages=[
            {"role": "system", "content": "你是金融年报GraphRAG助手。先识别公司、年份、财务或风险实体及关系，再综合多段证据。只能依据证据回答，按结论、依据、风险或限制组织答案，并标注年报页码；证据不足必须明确说明，禁止补造事实。"},
            {"role": "user", "content": f"请完成金融年报问题。问题：{question}\n\n已按实体和关系排序的GraphRAG证据：\n{evidence}"},
        ],
    )
    return response.choices[0].message.content.strip()


@app.post("/api/ask")
async def ask(payload: AskRequest):
    started = time.perf_counter()
    try:
        result = get_rag().search(payload.question, payload.top_k)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if key:
        try:
            answer = await _generate(payload.question, result["hits"], key)
            answer_mode = "deepseek"
        except Exception as exc:
            answer = _evidence_answer(result["hits"]) + f"\n\nDeepSeek暂时不可用：{exc}"
            answer_mode = "evidence_fallback"
    else:
        answer = _evidence_answer(result["hits"])
        answer_mode = "evidence"
    return {
        **result,
        "question": payload.question,
        "answer": answer,
        "answer_mode": answer_mode,
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _benchmark(identifier: int) -> dict:
    match = next((item for item in load_benchmarks(BENCHMARK_PATH) if item["id"] == identifier), None)
    if not match:
        raise HTTPException(status_code=404, detail="参考问题不存在")
    return match


@app.post("/api/evaluate")
async def evaluate(payload: EvaluateRequest):
    started = time.perf_counter()
    benchmark = _benchmark(payload.question_id) if payload.question_id is not None else None
    question = (benchmark["question"] if benchmark else (payload.question or "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="请选择参考问题或输入自定义问题")
    preferred = benchmark.get("retrieval", "") if benchmark else ""
    try:
        rag = get_rag()
        baseline = rag.search_baseline(question, payload.top_k)
        optimized = rag.search_optimized(question, payload.top_k, preferred)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    baseline_metrics = evaluate_hits(baseline["hits"], benchmark) if benchmark else None
    optimized_metrics = evaluate_hits(optimized["hits"], benchmark) if benchmark else None
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if key:
        try:
            answer = await _generate(question, optimized["hits"], key)
            mode = "deepseek"
        except Exception as exc:
            answer = _evidence_answer(optimized["hits"]) + f"\n\nDeepSeek暂时不可用：{exc}"
            mode = "evidence_fallback"
    else:
        answer, mode = _evidence_answer(optimized["hits"]), "evidence"
    delta = None
    if benchmark:
        delta = {
            "context_precision": round(optimized_metrics["context_precision"] - baseline_metrics["context_precision"], 4),
            "context_recall": round(optimized_metrics["context_recall"] - baseline_metrics["context_recall"], 4),
        }
    return {
        "question": question,
        "reference": benchmark["reference"] if benchmark else None,
        "answer": answer,
        "answer_mode": mode,
        "baseline": {**baseline, "metrics": baseline_metrics},
        "optimized": {**optimized, "metrics": optimized_metrics},
        "delta": delta,
        "optimization_layers": [
            "Prompt：限定证据、实体关系和答案结构",
            "实体关系：公司、年份、主题关系识别与匿名实体消歧",
            "检索：查询改写、图节点加权、标题加权与结果去重",
            "生成：多跳证据排序、页码引用和证据不足回退",
        ],
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@app.post("/api/evaluate-all")
async def evaluate_all(top_k: int = 5):
    if not 1 <= top_k <= 10:
        raise HTTPException(status_code=400, detail="Top K必须在1到10之间")
    results = [await evaluate(EvaluateRequest(question_id=item["id"], top_k=top_k)) for item in load_benchmarks(BENCHMARK_PATH)]
    precision = sum(item["optimized"]["metrics"]["context_precision"] for item in results) / len(results)
    recall = sum(item["optimized"]["metrics"]["context_recall"] for item in results) / len(results)
    return {"count": len(results), "average_context_precision": round(precision, 4), "average_context_recall": round(recall, 4), "under_3_seconds": all(item["total_ms"] < 3000 for item in results), "results": results}


def run_server() -> None:
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=False)


if __name__ == "__main__":
    run_server()

