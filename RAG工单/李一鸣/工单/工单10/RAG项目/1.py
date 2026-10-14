"""Work order 10: deploy the financial QA service as an observable API."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class ServiceSettings:
    model: str = os.getenv("LLM_MODEL", "financial-qa")
    max_concurrency: int = int(os.getenv("MAX_CONCURRENCY", "16"))
    timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT", "45"))


class FinancialQAService:
    def __init__(self, rag, settings: ServiceSettings | None = None):
        self.rag = rag
        self.settings = settings or ServiceSettings()
        self.semaphore = asyncio.Semaphore(self.settings.max_concurrency)

    async def ask(self, question: str, user_id: str = "anonymous") -> dict[str, Any]:
        started = time.perf_counter()
        async with self.semaphore:
            result = await asyncio.wait_for(asyncio.to_thread(self.rag.answer, question), self.settings.timeout_seconds)
        return {
            "user_id": user_id,
            "question": question,
            "answer": result,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "model": self.settings.model,
        }


def create_app(rag):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("FastAPI is required by the deployment target") from exc

    class AskRequest(BaseModel):
        question: str
        user_id: str = "anonymous"

    @asynccontextmanager
    async def lifespan(app):
        await rag.startup() if hasattr(rag, "startup") else asyncio.sleep(0)
        yield
        await rag.shutdown() if hasattr(rag, "shutdown") else asyncio.sleep(0)

    app = FastAPI(title="Financial RAG QA", version="1.0.0", lifespan=lifespan)
    service = FinancialQAService(rag)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": service.settings.model}

    @app.post("/v1/ask")
    async def ask(request: AskRequest):
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="question is required")
        return await service.ask(request.question, request.user_id)

    return app
