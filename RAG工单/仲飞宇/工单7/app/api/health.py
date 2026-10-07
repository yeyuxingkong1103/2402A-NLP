# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""健康检查：一眼看清 Ollama / Milvus / collection 状态。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.config import settings
from app.core.ollama_client import OllamaError, get_client
from app.core.vectorstore import VectorStore
from app.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthOut)
async def health() -> HealthOut:
    out = HealthOut(ok=True, llm_model=settings.llm_model,
                    embed_model=settings.embed_model)

    try:
        models = await get_client().list_models()
        out.ollama = f"ok（{len(models)} 个模型）"
    except OllamaError as e:
        out.ok = False
        out.ollama = f"失败：{str(e)[:120]}"

    store = VectorStore()
    ok, msg = await asyncio.to_thread(store.health)
    out.milvus = msg if ok else f"失败：{msg[:120]}"
    if not ok:
        out.ok = False
    else:
        out.collection_rows = await asyncio.to_thread(store.count)

    return out
