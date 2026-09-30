import logging
from typing import Any

from backend.app.embeddings.base import EmbeddingClient
from backend.app.rag.result_merger import RetrievedDocument, RetrievalFilters

logger = logging.getLogger(__name__)


def retrieve_dense_candidates(
    query: str,
    filters: RetrievalFilters,
    embedding_client: EmbeddingClient | None = None,
    vector_store: Any | None = None,
    top_k: int = 20,
) -> list[RetrievedDocument]:
    # 没有向量库时返回空结果，单元测试和最小 MVP 可只依赖 BM25。
    if vector_store is None or embedding_client is None:
        logger.info("向量检索跳过", extra={"has_vector_store": vector_store is not None, "has_embedding_client": embedding_client is not None})
        return []
    # 只记录问题长度，不输出用户问题全文。
    logger.info("开始向量检索", extra={"query_chars": len(query), "top_k": top_k})
    query_vector = embedding_client.embed_texts([query])[0]
    # 向量库适配器需提供 search(vector, top_k, filters)；具体 Milvus 接入由后续任务注入。
    rows = vector_store.search(query_vector, top_k=top_k, filters=filters)
    documents = [_row_to_document(row, "dense") for row in rows[:top_k]]
    logger.info("向量检索完成", extra={"result_count": len(documents)})
    return documents


def _row_to_document(row: dict[str, Any], source: str) -> RetrievedDocument:
    # 将外部存储返回结构收敛为统一 RetrievedDocument。
    metadata = dict(row.get("metadata", {}))
    return RetrievedDocument(
        id=str(row.get("id", row.get("material_id", ""))),
        material_id=str(row.get("material_id", "")),
        version_id=str(row.get("version_id", row.get("material_id", ""))),
        text=str(row.get("text", "")),
        score=float(row.get("score", 0.0)),
        source=source,
        metadata=metadata,
    )
