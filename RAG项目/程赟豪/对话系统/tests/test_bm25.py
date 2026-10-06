"""BM25 索引单元测试"""
from src.rag.bm25 import BM25Index


DOCS = [
    {"chunk_id": "1", "text": "同仁堂始创于1669年，是一家老字号药企", "source": "a", "page_number": 1},
    {"chunk_id": "2", "text": "变压器绝缘油需要定期检测溶解气体", "source": "b", "page_number": 1},
    {"chunk_id": "3", "text": "中金公司发布2025年中期报告", "source": "c", "page_number": 1},
]


def test_tokenize_chinese_bigram():
    toks = BM25Index.tokenize("同仁堂")
    assert "同仁" in toks and "仁堂" in toks


def test_tokenize_english_word():
    toks = BM25Index.tokenize("hello world 2025")
    assert "hello" in toks and "world" in toks and "2025" in toks


def test_build_and_search_rank():
    idx = BM25Index()
    idx.build(DOCS)
    results = idx.search("同仁堂老字号", top_k=3)
    assert results, "应能命中相关文档"
    assert results[0]["chunk_id"] == "1"


def test_search_empty_query():
    idx = BM25Index()
    idx.build(DOCS)
    assert idx.search("", top_k=3) == []


def test_search_unbuilt_index():
    assert BM25Index().search("任意", top_k=3) == []
