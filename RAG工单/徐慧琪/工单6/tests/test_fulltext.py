# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务

测试：自研倒排索引全文检索（src/fulltext.py）

覆盖工单技术要求：倒排索引、布尔查询、短语匹配、模糊匹配、多字段检索。
全部用手工构造的小语料，**不加载任何模型、不联网**。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import fulltext  # noqa: E402
from src.fulltext import InvertedIndex, Term, parse_query  # noqa: E402


def _docs() -> list[dict]:
    """构造小语料：3 篇文档，字段情况各不相同。"""
    return [
        {"source": "a.pdf", "chunk_id": "c1", "page_idx": 0, "block_type": "text",
         "heading_path": ["第一章", "发行概况"], "text": "本次发行股数为1,670万股，占发行后总股本的比例为25.04%。",
         "parent_text": "本次发行股数为1,670万股，占发行后总股本的比例为25.04%。"},
        {"source": "a.pdf", "chunk_id": "c2", "page_idx": 1, "block_type": "table_row",
         "heading_path": ["第一章"], "text": "公司注册资本为5,520.00万元。",
         "parent_text": "注册资本", "table_header": "项目|注册资本"},
        {"source": "b.pdf", "chunk_id": "c3", "page_idx": 2, "block_type": "text",
         "heading_path": ["第二章", "关联方"], "text": "存在控制关系的关联方为赵马克，持股比例42.35%。",
         "parent_text": "存在控制关系的关联方为赵马克，持股比例42.35%。"},
    ]


def _index() -> InvertedIndex:
    return InvertedIndex().build(_docs())


# ---------------------------------------------------------------------------
# 基础：倒排表与 BM25 打分
# ---------------------------------------------------------------------------
def test_build_creates_postings_and_stats():
    idx = _index()
    st = idx.stats()
    assert st["docs"] == 3
    assert st["vocab"]["body"] > 10          # 词表非空
    # 位置倒排：'注册资本' 出现在 c2 的 body 中，且记录了位置
    post = idx.postings["body"].get("注册资本")
    assert post and 1 in post
    assert len(post[1]) >= 1


def test_single_term_search_ranks_relevant_first():
    idx = _index()
    hits = idx.search("注册资本", top_k=3)
    assert hits, "应有命中"
    assert hits[0]["chunk_id"] == "c2"
    assert hits[0]["fulltext_score"] > 0


# ---------------------------------------------------------------------------
# 布尔查询
# ---------------------------------------------------------------------------
def test_boolean_and_requires_both_terms():
    idx = _index()
    hits = idx.search("注册资本 AND 法定代表人", top_k=5)
    assert hits == []                        # 没有任何文档同时含两者

    hits = idx.search("注册资本 AND 5,520", top_k=5)
    assert [h["chunk_id"] for h in hits] == ["c2"]


def test_boolean_or_unions_results():
    idx = _index()
    hits = idx.search("注册资本 OR 赵马克", top_k=5)
    ids = {h["chunk_id"] for h in hits}
    assert ids == {"c2", "c3"}


def test_boolean_not_excludes():
    idx = _index()
    hits = idx.search("发行 NOT 注册资本", top_k=5)
    ids = {h["chunk_id"] for h in hits}
    assert "c1" in ids and "c2" not in ids


def test_parentheses_grouping():
    idx = _index()
    hits = idx.search("(注册资本 OR 赵马克) AND 关联方", top_k=5)
    assert [h["chunk_id"] for h in hits] == ["c3"]


def test_chinese_operators_alias():
    idx = _index()
    assert [h["chunk_id"] for h in idx.search("注册资本 与 5,520", top_k=5)] == ["c2"]
    assert {h["chunk_id"] for h in idx.search("注册资本 或 赵马克", top_k=5)} == {"c2", "c3"}


# ---------------------------------------------------------------------------
# 短语匹配
# ---------------------------------------------------------------------------
def test_phrase_match_hits_only_adjacent():
    idx = _index()
    hits = idx.search('"存在控制关系的关联方"', top_k=5)
    assert [h["chunk_id"] for h in hits] == ["c3"]

    # 词都在、但不成短语（顺序/间隔不对）→ 不应命中短语，但普通 OR 查询能命中
    hits2 = idx.search('"关联方存在控制关系"', top_k=5)
    assert hits2 == []


def test_phrase_score_boosted_over_same_terms_unquoted():
    """同一组词：带引号（短语，位置相邻才命中 + ×1.6 加成）应高于不带引号（普通 OR）。"""
    idx = _index()
    phrase = idx.search('"本次发行股数"', top_k=5)
    unquoted = idx.search("本次发行股数", top_k=5)
    assert phrase and unquoted
    assert phrase[0]["chunk_id"] == unquoted[0]["chunk_id"] == "c1"
    assert phrase[0]["fulltext_score"] > unquoted[0]["fulltext_score"]


# ---------------------------------------------------------------------------
# 模糊匹配
# ---------------------------------------------------------------------------
def test_fuzzy_matches_edit_distance_one():
    idx = _index()
    # "法定代人" 少一个字 → 与 "法定代表人" 编辑距离 1；但语料里没有该词，
    # 故换一个语料里真实存在的词做模糊：'注册资本' → 少一字 '注册资本' 变体
    hits = idx.search("注册资木~", top_k=5)      # 木/本 一字之差
    assert hits, "模糊匹配应能命中 注册资本"
    assert hits[0]["chunk_id"] == "c2"


def test_fuzzy_disabled_by_default_for_exact_terms():
    idx = _index()
    assert idx.search("注册资木", top_k=5) == []   # 不加 ~ 不模糊


def test_levenshtein_helper():
    assert fulltext._levenshtein_le1("注册资本", "注册资木")   # 替换
    assert fulltext._levenshtein_le1("abc", "ab")             # 删除
    assert fulltext._levenshtein_le1("abc", "abcd")            # 插入
    assert fulltext._levenshtein_le1("abc", "abd")             # 替换，距离 1 → True
    assert not fulltext._levenshtein_le1("abcd", "ab")         # 距离 2 → False
    assert not fulltext._levenshtein_le1("abc", "xyz")         # 全不同 → False


# ---------------------------------------------------------------------------
# 多字段检索
# ---------------------------------------------------------------------------
def test_field_scoped_search_title_vs_body():
    idx = _index()
    # "关联方" 出现在 c3 的**标题**里（heading_path）与正文里
    by_title = idx.search("title:关联方", top_k=5)
    assert [h["chunk_id"] for h in by_title] == ["c3"]
    # "赵马克" 只在正文，标题里没有
    assert idx.search("title:赵马克", top_k=5) == []
    assert [h["chunk_id"] for h in idx.search("body:赵马克", top_k=5)] == ["c3"]


def test_field_scoped_summary():
    idx = _index()
    hits = idx.search("summary:注册资本", top_k=5)
    assert "c2" in {h["chunk_id"] for h in hits}


# ---------------------------------------------------------------------------
# 查询解析
# ---------------------------------------------------------------------------
def test_parse_query_shapes():
    node = parse_query("A AND B")
    assert node.__class__.__name__ == "And"
    node = parse_query("A OR B")
    assert node.__class__.__name__ == "Or"
    node = parse_query("NOT A")
    assert node.__class__.__name__ == "Not"
    node = parse_query('"短语"')
    assert isinstance(node, Term) and node.phrase
    node = parse_query("法定代理人~")
    assert isinstance(node, Term) and node.fuzzy == 1
    node = parse_query("title:注册资本")
    assert isinstance(node, Term) and node.field == "title"


def test_parse_query_empty_and_operators_only():
    assert parse_query("") is None
    assert parse_query("   ") is None
    assert parse_query("AND OR") is None


def test_implicit_or_between_adjacent_terms():
    idx = _index()
    hits = idx.search("注册资本 赵马克", top_k=5)
    assert {h["chunk_id"] for h in hits} == {"c2", "c3"}


def test_search_empty_query_returns_empty():
    idx = _index()
    assert idx.search("", top_k=5) == []
