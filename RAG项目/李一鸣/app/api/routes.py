import json
import logging
import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.rag.evaluation import EvaluationSample, evaluate_samples
from app.rag.service import RAGService
from app.storage.repositories import create_role, list_documents, list_roles
from app.storage.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentRead,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    RoleCreate,
    RoleRead,
    SearchRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


def get_rag(request: Request) -> RAGService:
    # 统一从 FastAPI application.state 取启动阶段创建的单例 RAGService。
    return request.app.state.rag


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request, rag: RAGService = Depends(get_rag)) -> HealthResponse:
    # 健康检查不仅返回服务状态，还暴露当前实际使用的记忆、向量库、LLM 和 embedding 后端。
    return HealthResponse(
        status="ok",
        app=request.app.state.settings.app_name,
        environment=request.app.state.settings.app_env,
        components={
            "database": "configured",
            "memory": "redis" if rag.memory._redis is not None else "local_fallback",
            "vector_store": rag.vector_store.health(),
            "llm": rag.llm.provider,
            "embedding": rag.embeddings.provider,
        },
    )


@router.get("/roles", response_model=list[RoleRead], tags=["roles"])
async def roles(session: AsyncSession = Depends(get_db)) -> list:
    return await list_roles(session)


@router.post("/roles", response_model=RoleRead, status_code=status.HTTP_201_CREATED, tags=["roles"])
async def create_role_endpoint(
    data: RoleCreate, session: AsyncSession = Depends(get_db)
):
    return await create_role(session, data)


@router.get("/documents", response_model=list[DocumentRead], tags=["knowledge-base"])
async def documents(session: AsyncSession = Depends(get_db)) -> list:
    return await list_documents(session)


@router.post("/documents/upload", response_model=DocumentRead, tags=["knowledge-base"])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    source: str = Form(default="upload"),
    role_id: str | None = Form(default=None),
    session: AsyncSession = Depends(get_db),
    rag: RAGService = Depends(get_rag),
):
    # 文件先保存到 upload_dir，再交给 RAGService 完成解析、分块、向量化和入库。
    filename = Path(file.filename or "document.txt").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".markdown", ".csv", ".json"}:
        raise HTTPException(status_code=400, detail="仅支持 PDF/TXT/MD/CSV/JSON 文件")
    upload_dir = rag.settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{uuid4().hex}{suffix}"
    started = time.perf_counter()
    try:
        target.write_bytes(await file.read())
        result = await rag.ingest_file(session, target, filename, source=source, role_id=role_id)
        logger.info(
            "upload endpoint completed",
            extra={"document_id": result.id if result else "-", "latency_ms": round((time.perf_counter() - started) * 1000, 2)},
        )
        return result
    except Exception as exc:
        logger.exception("upload endpoint failed: %s", filename)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/search", tags=["retrieval"])
async def search(
    data: SearchRequest, rag: RAGService = Depends(get_rag)
) -> dict:
    # 搜索接口用于单独观察召回结果，调试知识库时比直接看模型回答更直观。
    filters = {"role_id": data.role_id} if data.role_id else None
    started = time.perf_counter()
    results = rag.retriever.retrieve(data.query, top_k=data.top_k, filters=filters)
    results = rag.reranker.rerank(data.query, results, data.top_k)
    logger.info("search endpoint completed: %s hits", len(results), extra={"latency_ms": round((time.perf_counter() - started) * 1000, 2)})
    return {
        "query": data.query,
        "results": [
            {
                "chunk_id": item.chunk.id,
                "document_id": item.chunk.document_id,
                "content": item.chunk.text,
                "source": item.chunk.source,
                "score": round(item.score, 6),
                "dense_score": round(item.dense_score, 6),
                "lexical_score": round(item.lexical_score, 6),
                "retrieval_method": item.retrieval_method,
            }
            for item in results
        ],
    }


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    data: ChatRequest,
    session: AsyncSession = Depends(get_db),
    rag: RAGService = Depends(get_rag),
):
    # 同一个入口兼容非流式 JSON 和流式 SSE 两种响应模式。
    if data.stream:
        async def event_stream():
            async for event in rag.stream_chat(
                session,
                data.user_id,
                data.role_id,
                data.conversation_id,
                data.message,
                data.top_k,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await rag.chat(
            session,
            data.user_id,
            data.role_id,
            data.conversation_id,
            data.message,
            data.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatResponse(
        answer=result.answer,
        role_id=data.role_id,
        conversation_id=data.conversation_id,
        citations=[
            Citation(
                chunk_id=item.chunk.id,
                document_id=item.chunk.document_id,
                filename=item.chunk.source,
                content=item.chunk.text[:1000],
                score=round(item.score, 6),
                retrieval_method=item.retrieval_method,
            )
            for item in result.citations
        ],
        query=data.message,
        trace_id=result.trace_id,
    )


@router.post("/evaluate", response_model=EvaluationResponse, tags=["evaluation"])
async def evaluate(
    data: EvaluationRequest, rag: RAGService = Depends(get_rag)
) -> dict:
    samples = [EvaluationSample(**sample.model_dump()) for sample in data.samples]
    return await evaluate_samples(rag, samples)
