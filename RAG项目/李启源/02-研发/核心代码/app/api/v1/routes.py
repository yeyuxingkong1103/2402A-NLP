"""API routes for document ingestion and backward-compatible queries."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.api.v1 import access
from app.api.v1 import access
from app.api.v1.chat_schemas import ChatRequest
from app.api.v1.schemas import (
    IndexStatsResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    UploadDocumentResponse,
)
from app.ingestion.chunker import ChunkingConfig, chunk_document
from app.ingestion.embedding import build_embedding_client_from_env
from app.ingestion.indexer import (
    IngestionIndexer,
    SqlAlchemyMetadataStore,
    build_milvus_store_from_env,
)
from app.ingestion.parser import parse_document

logger = logging.getLogger(__name__)
router = APIRouter()
_embedding_client: Any = None
_vector_store: Any = None
_indexer: IngestionIndexer | None = None


def _get_embedding_client() -> Any:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = build_embedding_client_from_env()
    return _embedding_client


def _get_vector_store() -> Any:
    global _vector_store
    if _vector_store is None:
        _vector_store = build_milvus_store_from_env(_get_embedding_client().dimension)
    return _vector_store


def _get_indexer() -> IngestionIndexer:
    global _indexer
    if _indexer is None:
        metadata_store = None
        database_url = os.getenv("DATABASE_URL", "").strip()
        if database_url:
            metadata_store = SqlAlchemyMetadataStore(database_url)
        _indexer = IngestionIndexer(
            embedding_client=_get_embedding_client(),
            vector_store=_get_vector_store(),
            metadata_store=metadata_store,
            collection_name=os.getenv("MILVUS_COLLECTION", "kf_chunks"),
        )
    return _indexer


async def _upload_document(
    file: UploadFile,
    knowledge_base_id: int,
    tenant_id: int | None,
    role_id: int,
    parser: str,
    chunking_strategy: str,
    chunk_size: int,
    force_reindex: bool,
) -> UploadDocumentResponse:
    """Upload and index one document after route parameters are resolved."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    safe_name = Path(file.filename).name
    if safe_name != file.filename or safe_name in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    max_size = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")) * 1024 * 1024
    file_path: Path | None = None

    try:
        document_id = str(uuid.uuid4())
        storage_dir = Path(os.getenv("DOCUMENT_STORAGE_PATH", "data/documents"))
        storage_dir.mkdir(parents=True, exist_ok=True)
        file_path = storage_dir / f"{document_id}-{safe_name}"
        total = 0
        with file_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > max_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds {max_size // (1024 * 1024)} MB limit",
                    )
                output.write(chunk)
        document = parse_document(file_path, parser=parser)
        document = document.__class__(
            source_path=document.source_path,
            file_name=safe_name,
            file_sha256=document.file_sha256,
            mime_type=document.mime_type,
            pages=document.pages,
            metadata={
                **document.metadata,
                "knowledge_base_id": knowledge_base_id,
                "tenant_id": tenant_id,
                "role_id": role_id,
            },
        )
        result = chunk_document(
            document,
            config=ChunkingConfig(
                strategy=chunking_strategy,  # type: ignore[arg-type]
                chunk_size=chunk_size,
            ),
            document_id=document_id,
        )
        indexed = _get_indexer().index(
            document,
            result.chunks,
            document_id=document_id,
            tenant_id=tenant_id,
            role_id=role_id,
            knowledge_base_id=knowledge_base_id,
            force=force_reindex,
        )
        if str(indexed.document_id) != document_id:
            file_path.unlink(missing_ok=True)
        return UploadDocumentResponse(
            document_id=str(indexed.document_id),
            file_name=document.file_name,
            file_sha256=document.file_sha256,
            chunks_inserted=indexed.inserted,
            status="completed",
        )
    except HTTPException:
        if file_path is not None:
            file_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if file_path is not None:
            file_path.unlink(missing_ok=True)
        logger.exception("Document upload failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed",
        ) from exc


@router.post(
    "/documents/upload",
    response_model=UploadDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_id: int = Form(1),
    tenant_id: int | None = Form(None),
    role_id: int = Form(0),
    parser: str = Form("auto"),
    chunking_strategy: str = Form("paragraph"),
    chunk_size: int = Form(800),
    force_reindex: bool = Form(False),
) -> UploadDocumentResponse:
    tenant_id = access.tenant_id(tenant_id)
    access.owned_knowledge_base(knowledge_base_id)
    return await _upload_document(
        file,
        knowledge_base_id,
        tenant_id,
        role_id,
        parser,
        chunking_strategy,
        chunk_size,
        force_reindex,
    )


@router.post(
    "/knowledge-bases/{knowledge_base_id}/documents",
    response_model=UploadDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
@router.post(
    "/kb/{knowledge_base_id}/documents",
    response_model=UploadDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document_to_knowledge_base(
    knowledge_base_id: int,
    file: UploadFile = File(...),
    tenant_id: int | None = Form(None),
    role_id: int = Form(0),
    parser: str = Form("auto"),
    chunking_strategy: str = Form("paragraph"),
    chunk_size: int = Form(800),
    force_reindex: bool = Form(False),
) -> UploadDocumentResponse:
    tenant_id = access.tenant_id(tenant_id)
    access.owned_knowledge_base(knowledge_base_id)
    return await _upload_document(
        file,
        knowledge_base_id,
        tenant_id,
        role_id,
        parser,
        chunking_strategy,
        chunk_size,
        force_reindex,
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Compatibility adapter to the single ChatService pipeline."""
    try:
        from app.api.v1.chat_routes import get_chat_service

        scoped_request = access.scope_chat_request(
            ChatRequest(
                query=request.query,
                conversation_id=request.conversation_id,
                knowledge_base_id=request.knowledge_base_id,
                tenant_id=request.tenant_id,
                top_k=request.top_k,
                enable_rerank=request.rerank,
                enable_query_rewrite=False,
                enable_hybrid_search=True,
                temperature=request.temperature,
            )
        )
        chat = get_chat_service().answer(scoped_request)
        return QueryResponse(
            answer=chat.answer,
            query=chat.query,
            sources=[
                SourceChunk(
                    chunk_id=source.chunk_id,
                    text=source.text,
                    source=source.source,
                    score=source.score,
                    summary=source.summary,
                )
                for source in chat.sources
            ],
            model=chat.metadata.model,
            retrieval_count=chat.metadata.retrieval_count,
            prompt_tokens=chat.metadata.prompt_tokens,
            completion_tokens=chat.metadata.completion_tokens,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query processing failed",
        ) from exc


@router.get("/stats", response_model=IndexStatsResponse)
async def get_stats() -> IndexStatsResponse:
    """Get index statistics."""
    try:
        vector_store = _get_vector_store()
        vector_store.collection.load()
        return IndexStatsResponse(
            collection_name=os.getenv("MILVUS_COLLECTION", "kf_chunks"),
            total_vectors=vector_store.collection.num_entities,
            dimension=_get_embedding_client().dimension,
        )
    except Exception as exc:
        logger.exception("Stats retrieval failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stats retrieval failed",
        ) from exc
