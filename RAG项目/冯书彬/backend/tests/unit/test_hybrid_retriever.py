from backend.app.rag.hybrid_retriever import merge_hybrid_results
from backend.app.rag.result_merger import RetrievedDocument


def test_merges_vector_top20_and_bm25_top20_with_dedup():
    vector = [{"id": f"v{i}", "text": "x"} for i in range(20)]
    bm25 = [{"id": "v0", "text": "x"}] + [{"id": f"b{i}", "text": "y"} for i in range(19)]

    merged = merge_hybrid_results(vector, bm25)

    assert len(merged) == 39
    assert [doc["id"] for doc in merged].count("v0") == 1


def test_merge_preserves_dense_and_sparse_scores_for_same_chunk():
    vector = [
        RetrievedDocument(
            id="doc-1",
            material_id="mat-1",
            version_id="ver-1",
            text="第一条 内容",
            score=0.91,
            source="dense",
            metadata={"article": "第一条", "paragraph": "一"},
        )
    ]
    bm25 = [
        RetrievedDocument(
            id="doc-1-alt",
            material_id="mat-1",
            version_id="ver-1",
            text="第一条 内容",
            score=7.2,
            source="bm25",
            metadata={"article": "第一条", "paragraph": "一"},
        )
    ]

    merged = merge_hybrid_results(vector, bm25)

    assert len(merged) == 1
    assert merged[0].metadata["dense_score"] == 0.91
    assert merged[0].metadata["bm25_score"] == 7.2
    assert merged[0].metadata["sources"] == ["dense", "bm25"]
