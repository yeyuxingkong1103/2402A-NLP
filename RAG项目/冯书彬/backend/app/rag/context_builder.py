import logging

from backend.app.rag.citation_builder import build_citation
from backend.app.rag.result_merger import RetrievedDocument
from backend.app.schemas.retrieval import Citation

logger = logging.getLogger(__name__)


def build_context(documents: list[RetrievedDocument], max_chars_per_doc: int = 500) -> str:
    # 上下文只由最终通过治理过滤且可构造引用的 Top 文档构建。
    parts: list[str] = []
    traceable_documents = documents_with_citations(documents)
    for index, document in enumerate(traceable_documents, start=1):
        # 每段正文截断，防止把法律全文塞入模型上下文。
        excerpt = _context_text(document)[:max_chars_per_doc]
        article = document.metadata.get("article") or ""
        paragraph = document.metadata.get("paragraph") or ""
        parts.append(f"[{index}] {document.material_id} {article} {paragraph}\n{excerpt}")
    logger.info("检索上下文构建完成", extra={"document_count": len(traceable_documents), "context_chars": sum(len(part) for part in parts)})
    return "\n\n".join(parts)


def _context_text(document: RetrievedDocument) -> str:
    # 子块负责召回，父块负责补齐同一条文或章节的上下文。
    return str(document.metadata.get("parent_text") or document.text)


def build_citations(documents: list[RetrievedDocument]) -> list[Citation]:
    # 引用从文档元数据中的治理材料生成，没有材料对象则跳过。
    citations: list[Citation] = []
    traceable_documents = documents_with_citations(documents)
    for document in traceable_documents:
        material = document.metadata.get("material")
        citations.append(
            build_citation(
                material=material,
                article=document.metadata.get("article"),
                paragraph=document.metadata.get("paragraph"),
                excerpt=_context_text(document)[:200],
            )
        )
    logger.info("引用构建完成", extra={"document_count": len(traceable_documents), "citation_count": len(citations)})
    return citations


def documents_with_citations(documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
    # 最终上下文必须可追溯；缺少材料对象或引用构造必需字段时剔除。
    traceable: list[RetrievedDocument] = []
    for document in documents:
        material = document.metadata.get("material")
        if material is None:
            logger.warning("检索文档缺少材料对象，跳过上下文", extra={"document_id": document.id, "material_id": document.material_id})
            continue
        try:
            build_citation(
                material=material,
                article=document.metadata.get("article"),
                paragraph=document.metadata.get("paragraph"),
                excerpt=_context_text(document)[:200],
            )
        except (AttributeError, ValueError, TypeError):
            logger.warning("检索文档无法构造引用，跳过上下文", extra={"document_id": document.id, "material_id": document.material_id})
            continue
        traceable.append(document)
    return traceable
