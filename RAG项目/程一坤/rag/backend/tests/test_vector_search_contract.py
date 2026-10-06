"""向量召回参数契约测试。"""

from app.models.reranker import RankedCandidate
from app.retrieval.context_builder import RetrievedArticle
from app.retrieval.vector_search import LegalRetriever


class FixedEmbeddingClient:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3]]


class RecordingMilvusClient:
    def __init__(self) -> None:
        self.kwargs = None

    def search(self, **kwargs):
        self.kwargs = kwargs
        return [[]]


def test_vector_rerank_preserves_article_metadata() -> None:
    class SingleReranker:
        def rerank(self, query, documents, top_n):
            return [RankedCandidate(index=0, score=0.95)]

    source = RetrievedArticle(
        chunk_key="child",
        parent_chunk_key="parent",
        content="正文",
        document_title="法规",
        source_url="https://example.com",
        recall_score=0.8,
        document_type="法律",
        jurisdiction="中国大陆",
        effective_date=1577836800,
        issuing_authority="机关",
        is_current=True,
        document_id="doc-1",
    )
    retriever = LegalRetriever(
        embedding_client=FixedEmbeddingClient(),
        milvus_client=RecordingMilvusClient(),
        collection_name="legal_documents",
        session_factory=None,
        reranker=SingleReranker(),
    )

    result = retriever._rerank("问题", [source], 1)[0]

    assert result.parent_chunk_key == "parent"
    assert result.document_type == "法律"
    assert result.document_id == "doc-1"
    assert result.rerank_score == 0.95


def test_milvus_filter_is_passed_as_top_level_argument() -> None:
    milvus = RecordingMilvusClient()
    retriever = LegalRetriever(
        embedding_client=FixedEmbeddingClient(),
        milvus_client=milvus,
        collection_name="legal_documents",
        session_factory=None,
    )

    retriever._recall(
        "经济补偿",
        recall_limit=20,
        filter_expr='jurisdiction == "中国大陆"',
    )

    assert milvus.kwargs["filter"] == 'jurisdiction == "中国大陆"'
    assert milvus.kwargs["search_params"] == {"metric_type": "COSINE"}
