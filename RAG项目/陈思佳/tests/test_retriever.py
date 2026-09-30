from unittest.mock import Mock

from src.edu_rag_ingest.retrieval.retriever import MilvusRetriever, RetrievedChunk
from src.edu_rag_ingest.retrieval.hybrid_search import KeywordResult


def test_fuse_results_combines_vector_and_keyword_rankings():
    retriever = object.__new__(MilvusRetriever)
    retriever.config = Mock()
    retriever.config.hybrid_search.rrf_k = 60
    retriever.config.hybrid_search.vector_weight = 1.0
    retriever.config.hybrid_search.keyword_weight = 1.0

    vector_results = [
        RetrievedChunk(0.9, "vector-only", 0, "向量结果", {}),
        RetrievedChunk(0.8, "shared", 1, "共同结果", {}),
    ]
    keyword_results = [
        KeywordResult("keyword-only", 2, 5.0, "关键词结果", {"source_file": "keywords.md"}),
        KeywordResult("shared", 1, 4.0, "共同结果", {}),
    ]

    results = retriever._fuse_results(vector_results, keyword_results, top_k=3)

    assert [result.chunk_id for result in results] == ["shared", "vector-only", "keyword-only"]
    assert results[2].metadata["source_file"] == "keywords.md"
