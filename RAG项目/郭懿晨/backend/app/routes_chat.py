from pathlib import Path

from fastapi import APIRouter

from backend.app.config import AppSettings
from backend.app.embeddings import BgeM3Embedder, TextEmbedder
from backend.app.models import ChatRequest, ChatResponse
from backend.app.qa import QaService, build_citations, build_fallback_response
from backend.app.rerank import (
    RetrievalChain,
    build_coarse_scorer,
    CoarseReranker,
    FineReranker,
)
from backend.app.storage import JsonStateStore
from backend.app.vector_store import QdrantVectorStore, VectorStore


router = APIRouter(prefix="/api", tags=["chat"])


def build_retrieval_chain(settings: AppSettings, store: VectorStore, embedder: TextEmbedder) -> RetrievalChain:
    """构建检索重排链路。"""
    coarse = CoarseReranker(build_coarse_scorer(settings), settings.coarse_top_k)
    fine = FineReranker(
        settings.reranker_model_path,
        settings.rerank_batch_size,
        settings.final_top_k,
        settings.reranker_device,
    )
    return RetrievalChain(store, embedder, settings, coarse, fine)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """执行问答。"""
    question = request.question.strip()
    if not question:
        return build_fallback_response()

    settings = AppSettings()
    embedder = BgeM3Embedder(settings.bge_m3_model_path)
    store = QdrantVectorStore(settings.qdrant_path, settings.qdrant_collection)
    chain = build_retrieval_chain(settings, store, embedder)
    results = chain.retrieve(question)
    if not results:
        return build_fallback_response()

    document_store = JsonStateStore(Path("data/state.json"))
    document_names = {document.document_id: document.file_name for document in document_store.list_documents()}
    citations = build_citations(results, document_names)
    if not citations:
        return build_fallback_response()

    service = QaService(settings.ollama_model, settings.ollama_base_url)
    return service.answer(question, citations)
