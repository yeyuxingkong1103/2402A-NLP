"""重排序单元测试"""
from src.rag.rerank import SimpleReranker, get_reranker


def test_simple_reranker_sorts_by_overlap():
    docs = [
        {"chunk_id": "1", "text": "苹果公司发布了新款手机"},
        {"chunk_id": "2", "text": "香蕉是常见的水果"},
        {"chunk_id": "3", "text": "苹果是一种水果，也可以指苹果公司"},
    ]
    reranked = SimpleReranker().rerank("苹果公司", docs, top_k=3)
    assert reranked[0]["chunk_id"] in ("1", "3")
    assert all("rerank_score" in d for d in reranked)


def test_simple_reranker_empty():
    assert SimpleReranker().rerank("x", [], top_k=3) == []


def test_get_reranker_auto_fallback():
    # 本机未装 FlagEmbedding，auto 应退回 SimpleReranker
    r = get_reranker("auto")
    assert isinstance(r, SimpleReranker)
