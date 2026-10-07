from common.models import DocumentChunk
from main import optimize_results


def test_hybrid_ranking_deduplicates_results():
    chunks = [DocumentChunk(source="a.pdf", page=1, content="收入增长"), DocumentChunk(source="a.pdf", page=1, content="收入增长"), DocumentChunk(source="a.pdf", page=2, content="收入下降")]
    results = optimize_results("收入", chunks, top_k=2)
    assert len(results) == 2
    assert results[0].chunk.content == "收入增长"
    assert 0 <= results[0].keyword_score <= 1
    assert 0 <= results[0].vector_score <= 1
