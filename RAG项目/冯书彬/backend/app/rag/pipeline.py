import logging
from dataclasses import dataclass, field
from typing import Any

from backend.app.core.config import settings
from backend.app.rag.context_builder import build_citations, build_context, documents_with_citations
from backend.app.rag.result_merger import RetrievedDocument, RetrievalFilters, as_retrieved_document
from backend.app.rerank.base import RerankClient
from backend.app.services.legal_version_service import find_applicable_materials
from backend.app.schemas.retrieval import Citation

logger = logging.getLogger(__name__)


@dataclass
class RetrievalDecision:
    # RAG 检索决策明确表达能否回答和所依据的最终文档。
    can_answer: bool
    reason: str
    top_documents: list[RetrievedDocument]
    citations: list[Citation]
    context: str = ""
    scores: list[float] = field(default_factory=list)


def rerank_and_select(
    query: str,
    candidates: list[RetrievedDocument],
    rerank_client: RerankClient,
    filters: RetrievalFilters | None = None,
    threshold: float = settings.RERANKER_THRESHOLD,
) -> RetrievalDecision:
    # Reranker 输入只传候选正文，日志不记录 query 或正文。
    scores = rerank_client.score(query, [candidate.text for candidate in candidates])
    logger.info("Reranker 打分完成", extra={"query_chars": len(query), "candidate_count": len(candidates), "score_count": len(scores)})
    return decide_after_rerank(scores=scores, documents=candidates, filters=filters, threshold=threshold)


def decide_after_rerank(
    scores: list[float],
    documents: list[RetrievedDocument | dict[str, Any]],
    filters: RetrievalFilters | None = None,
    threshold: float = settings.RERANKER_THRESHOLD,
) -> RetrievalDecision:
    # 无候选或无分数时，直接判定法律依据不足。
    if not scores or not documents:
        return _insufficient_decision("no_candidates")
    normalized = [as_retrieved_document(document) for document in documents]
    ranked = _rank_by_scores(normalized, scores)
    governed = documents_with_citations(_filter_governed_documents(ranked, filters))
    if not governed:
        logger.info("治理过滤后无可用材料", extra={"ranked_count": len(ranked)})
        return _insufficient_decision("insufficient_legal_basis")
    highest_governed_score = max(float(document.metadata.get("rerank_score", 0.0)) for document in governed)
    # 必须先过滤无效材料，再用最终可用材料的最高分判断阈值。
    if highest_governed_score < threshold:
        logger.info("Reranker 阈值阻断回答", extra={"highest_score": highest_governed_score, "threshold": threshold, "candidate_count": len(governed)})
        return _insufficient_decision("insufficient_legal_basis")
    selected = governed[:5]
    citations = build_citations(selected)
    context = build_context(selected)
    logger.info(
        "RAG 检索决策允许回答",
        extra={"selected_count": len(selected), "citation_count": len(citations), "highest_score": highest_governed_score, "threshold": threshold},
    )
    return RetrievalDecision(can_answer=True, reason="ok", top_documents=selected, citations=citations, context=context, scores=scores)


def _rank_by_scores(documents: list[RetrievedDocument], scores: list[float]) -> list[RetrievedDocument]:
    # 分数数量与文档数量不一致时只使用成对部分，避免错配依据。
    paired = list(zip(documents, scores, strict=False))
    paired.sort(key=lambda item: item[1], reverse=True)
    ranked: list[RetrievedDocument] = []
    for document, score in paired:
        # Reranker 分数写入元数据，便于审计最终排序。
        document.metadata["rerank_score"] = score
        ranked.append(document)
    return ranked


def _filter_governed_documents(documents: list[RetrievedDocument], filters: RetrievalFilters | None) -> list[RetrievedDocument]:
    # 不论调用方是否传 filters，都必须依赖材料元数据校验发布状态和时间适用性。
    materials = [document.metadata.get("material") for document in documents if document.metadata.get("material") is not None]
    if not materials:
        logger.info("最终上下文治理过滤完成", extra={"input_count": len(documents), "allowed_count": 0, "output_count": 0})
        return []
    query_date = filters.query_date if filters is not None else None
    relationship_type = filters.relationship_type if filters is not None else "general"
    candidate_materials = filters.materials if filters is not None and filters.materials else materials
    applicable = find_applicable_materials(query_date, relationship_type, candidate_materials)
    allowed = {(item.material_id, item.version_id) for item in applicable}
    filtered = [document for document in documents if (document.material_id, document.version_id) in allowed and _document_material_is_published(document)]
    logger.info("最终上下文治理过滤完成", extra={"input_count": len(documents), "allowed_count": len(allowed), "output_count": len(filtered)})
    return filtered


def _document_material_is_published(document: RetrievedDocument) -> bool:
    # 缺少可验证材料对象时不得进入最终上下文。
    material = document.metadata.get("material")
    if material is None:
        logger.warning("检索文档缺少可验证材料元数据", extra={"document_id": document.id, "material_id": document.material_id})
        return False
    return material.status == "published" and material.searchable is True


def _insufficient_decision(reason: str) -> RetrievalDecision:
    # 不足依据时清空文档、引用和上下文，保证不会继续分析。
    return RetrievalDecision(can_answer=False, reason=reason, top_documents=[], citations=[], context="", scores=[])
