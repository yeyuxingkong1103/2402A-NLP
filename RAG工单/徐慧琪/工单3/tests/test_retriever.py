# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化 —— 混合检索测试"""
import pytest

from src import retriever


def test_tokenize_protects_domain_terms():
    toks = retriever.tokenize("武汉兴图新科电子股份有限公司注册资本是多少")
    assert "注册资本" in toks
    # 数字与英文整体保留
    toks2 = retriever.tokenize("7,360.00万元 IPO 融资")
    assert "7,360.00" in toks2
    assert "ipo" in toks2


def test_rrf_fuse_orders_by_reciprocal_rank():
    dense = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    sparse = [{"chunk_id": "b"}, {"chunk_id": "a"}]
    fused = retriever.rrf_fuse([("dense", dense), ("sparse", sparse)], k=60)
    # a: 1/61 + 1/62 ; b: 1/62 + 1/61 → 两者并列最高，且都在 c 前
    top2 = {fused[0]["chunk_id"], fused[1]["chunk_id"]}
    assert top2 == {"a", "b"}
    assert fused[2]["chunk_id"] == "c"
    assert fused[0]["rrf_score"] > fused[2]["rrf_score"]


def test_rrf_merges_metadata_from_both_sources():
    dense = [{"chunk_id": "a", "score": 0.9, "text": "t"}]
    sparse = [{"chunk_id": "a", "bm25_score": 12.5}]
    fused = retriever.rrf_fuse([("dense", dense), ("sparse", sparse)])
    assert len(fused) == 1
    assert fused[0]["vector_score"] == pytest.approx(0.9)
    assert fused[0]["bm25_score"] == pytest.approx(12.5)
    assert fused[0]["ranks"] == {"dense": 1, "sparse": 1}
    assert fused[0]["text"] == "t"          # payload 字段保留


def test_search_hybrid_calls_both_paths(monkeypatch):
    calls = {"dense": 0, "sparse": 0}

    def fake_dense(query, top_k, name, sources=None, exclude_block_types=None):
        calls["dense"] += 1
        return [{"chunk_id": "a", "score": 1.0, "text": "x"}]

    def fake_sparse(query, top_k, sources=None, exclude_block_types=None):
        calls["sparse"] += 1
        return [{"chunk_id": "a", "bm25_score": 5.0}]

    monkeypatch.setattr(retriever, "_dense_search", fake_dense)
    monkeypatch.setattr(retriever, "_sparse_search", fake_sparse)
    out = retriever.search("注册资本", top_k=3, hybrid=True)
    assert calls == {"dense": 1, "sparse": 1}
    assert out and out[0]["chunk_id"] == "a"


def test_search_multi_fuses_subqueries(monkeypatch):
    def fake_dense(query, top_k, name, sources=None, exclude_block_types=None):
        return [{"chunk_id": f"d-{query}", "score": 1.0}]

    def fake_sparse(query, top_k, sources=None, exclude_block_types=None):
        return [{"chunk_id": f"s-{query}", "bm25_score": 1.0}]

    monkeypatch.setattr(retriever, "_dense_search", fake_dense)
    monkeypatch.setattr(retriever, "_sparse_search", fake_sparse)
    out = retriever.search_multi(["军用领域收入", "军用领域收入占比"], top_k=5)
    ids = {h["chunk_id"] for h in out}
    assert "d-军用领域收入" in ids and "s-军用领域收入占比" in ids
