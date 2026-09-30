from pathlib import Path

from src.edu_rag_ingest.retrieval.hybrid_search import BM25Index, KeywordDocument, tokenize


def test_tokenize_supports_chinese_bigrams_and_ascii_terms():
    tokens = tokenize("阅读教学要求 BGE-M3")

    assert "阅读" in tokens
    assert "教学" in tokens
    assert "bge" in tokens
    assert "m3" in tokens


def test_bm25_prefers_exact_keyword_match():
    index = BM25Index(
        [
            KeywordDocument("one", 0, "阅读教学要求和课堂活动设计", {"title": "课标"}),
            KeywordDocument("two", 1, "数学作业设计和试题分析", {"title": "题库"}),
        ]
    )

    results = index.search("阅读教学", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "one"
    assert results[0].metadata["title"] == "课标"


def test_bm25_from_jsonl_missing_file_returns_empty(tmp_path: Path):
    index = BM25Index.from_jsonl(tmp_path / "missing.jsonl")

    assert index.search("不存在", top_k=5) == []
