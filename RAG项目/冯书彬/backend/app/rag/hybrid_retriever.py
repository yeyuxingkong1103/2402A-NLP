import logging
from typing import Any

from backend.app.embeddings.base import EmbeddingClient
from backend.app.rag.dense_retriever import retrieve_dense_candidates
from backend.app.rag.result_merger import RetrievedDocument, RetrievalFilters, merge_retrieved_documents
from backend.app.rag.sparse_retriever import retrieve_bm25_candidates

logger = logging.getLogger(__name__)


def merge_hybrid_results(vector_results: list[Any], bm25_results: list[Any]) -> list[RetrievedDocument]:
    # 对外保留简洁函数名，内部统一使用 result_merger 的治理去重逻辑。
    return merge_retrieved_documents(vector_results, bm25_results)


def retrieve_candidates(
    query: str,
    filters: RetrievalFilters,
    embedding_client: EmbeddingClient | None = None,
    vector_store: Any | None = None,
) -> list[RetrievedDocument]:
    # 向量和 BM25 各取 Top 20，随后统一合并去重。
    vector_results = retrieve_dense_candidates(query, filters, embedding_client=embedding_client, vector_store=vector_store, top_k=20)
    bm25_results = retrieve_bm25_candidates(query, filters, top_k=20)
    merged = merge_hybrid_results(vector_results, bm25_results)
    logger.info(
        "混合检索完成",
        extra={"query_chars": len(query), "vector_count": len(vector_results), "bm25_count": len(bm25_results), "merged_count": len(merged)},
    )
    return merged
