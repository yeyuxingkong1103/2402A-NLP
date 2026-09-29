import logging
import math
from collections import Counter

from backend.app.models.knowledge_base import KnowledgeMaterial
from backend.app.rag.result_merger import RetrievedDocument, RetrievalFilters
from backend.app.services.legal_version_service import find_applicable_materials

logger = logging.getLogger(__name__)


def retrieve_bm25_candidates(query: str, filters: RetrievalFilters, top_k: int = 20) -> list[RetrievedDocument]:
    # BM25 只在已传入的治理材料集合内检索，避免绕过发布和适用性过滤。
    applicable = find_applicable_materials(filters.query_date, filters.relationship_type, filters.materials)
    allowed = {(item.material_id, item.version_id) for item in applicable}
    rows: list[tuple[float, KnowledgeMaterial]] = []
    terms = _tokenize(query)
    for material in filters.materials:
        version_id = str(getattr(material, "version_id", material.id))
        if (material.id, version_id) not in allowed:
            continue
        score = _keyword_score(terms, material.raw_text)
        if score > 0:
            rows.append((score, material))
    rows.sort(key=lambda item: item[0], reverse=True)
    documents = [_material_to_document(material, score) for score, material in rows[:top_k]]
    logger.info("BM25 检索完成", extra={"query_chars": len(query), "candidate_count": len(rows), "result_count": len(documents)})
    return documents


def _tokenize(text: str) -> list[str]:
    # 简化分词：中文按字符、英文数字按连续片段，满足 MVP 的关键词召回测试。
    tokens: list[str] = []
    current = ""
    for char in text.lower():
        if "一" <= char <= "鿿":
            if current:
                tokens.append(current)
                current = ""
            tokens.append(char)
        elif char.isalnum():
            current += char
        else:
            if current:
                tokens.append(current)
                current = ""
    if current:
        tokens.append(current)
    return tokens


def _keyword_score(query_terms: list[str], text: str) -> float:
    # 空查询不参与打分，避免把所有材料召回。
    if not query_terms:
        return 0.0
    document_terms = _tokenize(text)
    counts = Counter(document_terms)
    length_norm = max(len(document_terms), 1)
    score = 0.0
    for term in query_terms:
        # 使用简化 BM25 形态：词频经 log 压缩并按文档长度归一。
        score += math.log1p(counts.get(term, 0)) / math.sqrt(length_norm)
    return score


def _material_to_document(material: KnowledgeMaterial, score: float) -> RetrievedDocument:
    # 摘录长度受控，最终上下文再做二次截断。
    excerpt = material.raw_text[:500]
    version_id = str(getattr(material, "version_id", material.id))
    return RetrievedDocument(
        id=f"{material.id}:{version_id}",
        material_id=material.id,
        version_id=version_id,
        text=excerpt,
        score=score,
        source="bm25",
        metadata={
            "material": material,
            "article": getattr(material, "article", None),
            "paragraph": getattr(material, "paragraph", None),
        },
    )
