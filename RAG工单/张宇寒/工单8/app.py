from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from graph_rag import DATA_DIR, GraphRAG
from questions import load_questions


ROOT = Path(__file__).resolve().parent
QUESTION_PATH = ROOT / "eval_questions.md"
load_dotenv(ROOT / ".env")
app = FastAPI(title="工单8：金融年报 Graph RAG")
_rag: GraphRAG | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
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
            {"role": "system", "content": "你是金融年报问答助手。只能依据给定证据回答，答案简洁并标注年报页码；证据不足时明确说明。"},
            {"role": "user", "content": f"问题：{question}\n\nGraph RAG证据：\n{evidence}"},
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


def run_server() -> None:
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8019, reload=False)


if __name__ == "__main__":
    run_server()

