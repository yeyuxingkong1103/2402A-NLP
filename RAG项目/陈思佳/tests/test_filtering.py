from src.edu_rag_ingest.retrieval.filtering import matches_metadata
from src.edu_rag_ingest.retrieval.hybrid_search import BM25Index, KeywordDocument
from src.edu_rag_ingest.retrieval.milvus_store import MilvusChunkStore


def test_matches_metadata_requires_all_requested_fields():
    metadata = {"subject": "语文", "grade": "九年级", "section": "阅读"}

    assert matches_metadata(metadata, {"subject": "语文", "grade": "九年级"})
    assert not matches_metadata(metadata, {"subject": "数学"})


def test_bm25_applies_metadata_filters():
    index = BM25Index(
        [
            KeywordDocument("cn", 0, "阅读教学重点", {"subject": "语文"}),
            KeywordDocument("math", 1, "阅读教学重点", {"subject": "数学"}),
        ]
    )

    results = index.search("阅读教学", top_k=5, filters={"subject": "语文"})

    assert [result.chunk_id for result in results] == ["cn"]


def test_milvus_filter_only_uses_scalar_metadata_fields():
    assert MilvusChunkStore.build_filter({"subject": "语文", "chapter": "阅读"}) == 'subject == "语文"'
