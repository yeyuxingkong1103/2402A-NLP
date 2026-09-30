"""BM25 关键词路测试（不依赖外部服务）。"""

from __future__ import annotations

from role_rag.retrieval.bm25 import BM25Index, tokenize


ROWS = [
    {"pk": 1, "text": "试用期最长不得超过六个月，试用期工资不得低于本单位相同岗位最低档工资的百分之八十。",
     "doc_id": "d1", "chunk_index": 0, "doc_title": "劳动争议要点", "section": "一、试用期"},
    {"pk": 2, "text": "普通诉讼时效期间为三年，自权利人知道或者应当知道权利受到损害之日起计算。",
     "doc_id": "d2", "chunk_index": 0, "doc_title": "证据规则", "section": "三、诉讼时效"},
    {"pk": 3, "text": "民间借贷利率超过合同成立时一年期贷款市场报价利率四倍的部分不受法律保护。",
     "doc_id": "d3", "chunk_index": 0, "doc_title": "民间借贷", "section": "二、利率规则"},
]


def test_tokenize_filters_stopwords_and_punctuation():
    tokens = tokenize("试用期的，工资是 80% 吗？")
    assert "试用期" in tokens
    assert "的" not in tokens
    assert "，" not in tokens
    assert all(token == token.lower() for token in tokens)


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("，。！？") == []


def test_index_returns_relevant_first():
    index = BM25Index("lawyer", ROWS, version=1)
    assert len(index) == 3
    hits = index.search("诉讼时效是几年", limit=3)
    assert hits and hits[0].payload["doc_id"] == "d2"
    assert hits[0].chunk_id == "d2#0"


def test_trial_period_query_matches_expected_doc():
    index = BM25Index("lawyer", ROWS, version=1)
    hits = index.search("试用期最长多久", limit=3)
    assert hits[0].payload["doc_title"] == "劳动争议要点"


def test_empty_index_returns_nothing():
    index = BM25Index("empty", [], version=1)
    assert len(index) == 0
    assert index.search("任意查询", limit=5) == []


def test_index_skips_rows_without_tokens():
    index = BM25Index("weird", [{"text": "，。！", "doc_id": "d", "chunk_index": 0}], version=1)
    assert len(index) == 0


def test_stats_shape():
    stats = BM25Index("lawyer", ROWS, version=7).stats()
    assert stats["scope"] == "lawyer" and stats["docs"] == 3 and stats["version"] == 7
