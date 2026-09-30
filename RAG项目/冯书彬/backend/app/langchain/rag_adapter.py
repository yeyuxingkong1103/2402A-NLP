import inspect
import logging
from collections.abc import Callable
from typing import Any

from backend.app.rag.chat_adapter import safe_insufficient_decision
from backend.app.rag.context_builder import build_citations, build_context, documents_with_citations
from backend.app.rag.result_merger import RetrievedDocument, as_retrieved_document
from backend.app.rag.result_merger import RetrievalFilters
from backend.app.rag.pipeline import RetrievalDecision
from backend.app.services.legal_version_service import find_applicable_materials
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class LangChainUnavailableError(RuntimeError):
    """LangChain 依赖未安装或无法初始化。"""


def _load_langchain_types():
    try:
        from langchain_core.documents import Document
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        from langchain_core.retrievers import BaseRetriever
    except ImportError as exc:
        raise LangChainUnavailableError("langchain-core is not installed") from exc
    return BaseRetriever, Document, RunnableLambda, RunnablePassthrough


def _document_from_retrieved(document: RetrievedDocument, document_type: Any) -> Any:
    metadata = dict(document.metadata)
    metadata["_retrieved_document"] = document
    metadata.update(
        {
            "document_id": document.id,
            "material_id": document.material_id,
            "version_id": document.version_id,
            "score": document.score,
            "source": document.source,
        }
    )
    return document_type(page_content=document.text, metadata=metadata)


def _retrieved_from_document(document: Any) -> RetrievedDocument:
    original = document.metadata.get("_retrieved_document")
    if isinstance(original, RetrievedDocument):
        return original
    metadata = dict(document.metadata)
    metadata.pop("_retrieved_document", None)
    return as_retrieved_document(
        {
            "id": metadata.get("document_id", ""),
            "material_id": metadata.get("material_id", ""),
            "version_id": metadata.get("version_id", ""),
            "text": document.page_content,
            "score": metadata.get("score", 0.0),
            "source": metadata.get("source", "langchain"),
            "metadata": metadata,
        },
        default_source="langchain",
    )


def _filter_governed_documents(documents: list[RetrievedDocument], filters: RetrievalFilters | None) -> list[RetrievedDocument]:
    materials = [document.metadata.get("material") for document in documents if document.metadata.get("material") is not None]
    if not materials:
        return []
    query_date = filters.query_date if filters is not None else None
    relationship_type = filters.relationship_type if filters is not None else "general"
    candidate_materials = filters.materials if filters is not None and filters.materials else materials
    applicable = find_applicable_materials(query_date, relationship_type, candidate_materials)
    allowed = {(item.material_id, item.version_id) for item in applicable}
    return [
        document
        for document in documents
        if (document.material_id, document.version_id) in allowed
        and document.metadata["material"].status == "published"
        and document.metadata["material"].searchable is True
    ]


def _insufficient_decision(reason: str) -> RetrievalDecision:
    return RetrievalDecision(can_answer=False, reason=reason, top_documents=[], citations=[], context="", scores=[])


def create_langchain_rag_decider(
    candidate_provider: Callable[[str], Any],
    rerank_client: Any,
    filters: RetrievalFilters | None = None,
) -> Callable[[str], Any]:
    """使用 LangChain Retriever 和 LCEL 执行法律 RAG 决策。"""
    BaseRetriever, document_type, RunnableLambda, RunnablePassthrough = _load_langchain_types()

    class LegalHybridRetriever(BaseRetriever):
        candidate_provider: Callable[[str], Any]
        model_config = {"arbitrary_types_allowed": True}

        def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Any]:
            candidates = self.candidate_provider(query)
            if inspect.isawaitable(candidates):
                raise RuntimeError("async candidate provider requires ainvoke")
            return [_document_from_retrieved(item, document_type) for item in candidates]

        async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Any]:
            candidates = self.candidate_provider(query)
            if inspect.isawaitable(candidates):
                candidates = await candidates
            return [_document_from_retrieved(item, document_type) for item in candidates]

    retriever = LegalHybridRetriever(candidate_provider=candidate_provider)

    def rerank(payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload["query"])
        documents = [_retrieved_from_document(item) for item in payload["documents"]]
        if not documents:
            return {"query": query, "documents": [], "scores": []}
        scores = rerank_client.score(query, [document.text for document in documents])
        ranked = []
        for document, score in sorted(zip(documents, scores, strict=False), key=lambda pair: pair[1], reverse=True):
            document.metadata["rerank_score"] = score
            ranked.append(document)
        return {"query": query, "documents": ranked, "scores": scores}

    def govern(payload: dict[str, Any]) -> RetrievalDecision:
        documents = payload["documents"]
        scores = payload["scores"]
        if not scores or not documents:
            return _insufficient_decision("no_candidates")
        governed = documents_with_citations(_filter_governed_documents(documents, filters))
        if not governed:
            return _insufficient_decision("insufficient_legal_basis")
        highest_score = max(float(document.metadata.get("rerank_score", 0.0)) for document in governed)
        if highest_score < settings.RERANKER_THRESHOLD:
            return _insufficient_decision("insufficient_legal_basis")
        selected = governed[:5]
        return RetrievalDecision(
            can_answer=True,
            reason="ok",
            top_documents=selected,
            citations=build_citations(selected),
            context=build_context(selected),
            scores=scores,
        )

    chain = (
        {"query": RunnablePassthrough(), "documents": retriever}
        | RunnableLambda(rerank)
        | RunnableLambda(govern)
    )

    async def invoke(text: str) -> RetrievalDecision:
        result = await chain.ainvoke(text)
        if not isinstance(result, RetrievalDecision):
            logger.warning("LangChain RAG 返回类型异常", extra={"result_type": type(result).__name__})
            return safe_insufficient_decision("langchain_invalid_result")
        return result

    return invoke
