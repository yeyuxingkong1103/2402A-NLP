# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
tests/test_hybrid_retriever.py —— 工单二混合检索器单元测试

策略：注入伪 embedder / 预置向量，不依赖真实 Milvus 与模型，秒级完成；
真实链路（bge-m3 + reranker + 全量 1565 子块）由 python -m src.hybrid_retriever 演示验证。
"""
import zlib

import numpy as np
import pytest

from src.hybrid_retriever import (HybridRetriever, LocalVectorIndex, RRF_K,
                                  detect_intent, expand_query, _tokenize)
from src.reranker import Reranker


def _stable_seed(text: str) -> int:
    """内容确定性种子（工单二 Step9 修复：原先用内置 hash()，受 PYTHONHASHSEED
    随机化影响导致融合排名边界翻转、测试偶发失败；人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    return zlib.crc32(text.encode("utf-8"))

DOC_ID = "testdoc"


@pytest.fixture
def chunks():
    """6 个测试子块：c0/c1 命中军用收入语义，c2 普通业务，c3 风险，c4/c5 无关"""
    return [
        {"chunk_id": "c0", "text": "报告期内公司军品收入为12,000万元，军用领域收入占营业收入的比例为45%。",
         "page": 33, "chunk_type": "text", "heading": "第二节 概览", "parent_id": "p0", "char_count": 48},
        {"chunk_id": "c1", "text": "公司军用电子信息系统主要应用于国防通信领域，军用领域收入持续增长。",
         "page": 34, "chunk_type": "text", "heading": "第二节 概览", "parent_id": "p0", "char_count": 33},
        {"chunk_id": "c2", "text": "公司主要客户为国家大型企业，客户集中度较高。",
         "page": 40, "chunk_type": "text", "heading": "第三节 客户", "parent_id": "p1", "char_count": 22},
        {"chunk_id": "c3", "text": "公司面临市场竞争加剧的风险，可能导致毛利率下降。",
         "page": 55, "chunk_type": "text", "heading": "第四节 风险因素", "parent_id": "p2", "char_count": 24},
        {"chunk_id": "c4", "text": "本次发行的募集资金总额为60,000万元。",
         "page": 12, "chunk_type": "text", "heading": "第一节 释义", "parent_id": "p3", "char_count": 19},
        {"chunk_id": "c5", "text": "| 项目 | 金额 |\\n| --- | --- |\\n| 营业收入 | 100万 |",
         "page": 90, "chunk_type": "table", "heading": "附表", "parent_id": "p4", "char_count": 60},
    ]


@pytest.fixture
def fake_vectors(chunks):
    """构造向量：c0（军用收入）方向最强，c1（军用）次之，其余近随机；
    查询向量 = c0 方向，保证"军用领域收入"向量路 c0 必排第一"""
    rng = np.random.RandomState(42)
    mat = rng.randn(len(chunks), 8).astype("float32") * 0.5
    mat[0] += [3, 3, 3, 3, 0, 0, 0, 0]   # c0 军用收入
    mat[1] += [2, 2, 0, 0, 0, 0, 0, 0]   # c1 军用
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat, mat[0].copy()


class FakeEmbedder:
    """伪 embedder：按查询关键词直接映射到 fixture 中同一查询向量方向"""

    def __init__(self, q):
        self._q = q

    def encode(self, texts, batch_size=64, show_progress_bar=False):
        out = []
        for t in texts:
            v = self._q.copy()
            if "风险" in t:
                v = np.roll(v, 2)
            if "募集" in t or "假设" in t or "HyDE" in t:
                v = np.roll(v, 4)
            v = v + np.random.RandomState(_stable_seed(t)).randn(8) * 0.001
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype="float32")


def make_retriever(chunks, mat, q, use_rerank=True):
    """构造被测检索器：本地向量索引 + 预置向量 + 伪 embedder + 空转 reranker
    （vec_cache_path=None 防止测试污染生产向量缓存，人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    lvi = LocalVectorIndex(chunks, cache_path=None, embedder=FakeEmbedder(q), vectors=mat)
    rr = Reranker()
    rr._failed = True  # 测试环境不加载真实模型 → 降级保序（score 透传）
    return HybridRetriever(chunks=chunks, vector_store=None, embedder=FakeEmbedder(q),
                           reranker=rr, use_rerank=use_rerank, use_hyde=False,
                           vec_cache_path=None), lvi


# ---------- 1. 查询改写（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_expand_query_synonyms():
    r = expand_query("军用领域收入是多少")
    assert set(r["expanded_terms"]) >= {"军品", "军事", "国防", "营业收入"}
    assert "军品" in r["expanded_query"]


def test_detect_intent():
    assert "statistics" in detect_intent("军用领域收入是多少")
    assert "summary" in detect_intent("公司有哪些主要产品")
    assert detect_intent("你好") == ["qa"]


# ---------- 2. 多路召回与融合（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_multi_recall_dedup(chunks, fake_vectors):
    """向量 top10 + BM25 top10，chunk_id 去重：结果 chunk_id 无重复"""
    ret, _ = make_retriever(chunks, *fake_vectors)
    out = ret.retrieve("军用领域收入是多少", top_k=5)
    ids = [r["chunk_id"] for r in out["results"]]
    assert len(ids) == len(set(ids)), "结果必须按 chunk_id 去重"


def test_result_fields(chunks, fake_vectors):
    ret, _ = make_retriever(chunks, *fake_vectors)
    out = ret.retrieve("军用领域收入", top_k=5)
    for r in out["results"]:
        for key in ("chunk_id", "content", "score", "source", "page"):
            assert key in r, f"缺少字段 {key}"
        assert "#page=" in r["source"]


def test_vec_ranking_relevance(chunks, fake_vectors):
    """向量最相关的 c0 应进入 top2（融合后）"""
    ret, _ = make_retriever(chunks, *fake_vectors)
    out = ret.retrieve("军用领域收入", top_k=5)
    top_ids = [r["chunk_id"] for r in out["results"][:2]]
    assert "c0" in top_ids


def test_bm25_only_query(chunks, fake_vectors):
    """纯 BM25 可命中的词：结果非空且含对应块"""
    ret, _ = make_retriever(chunks, *fake_vectors)
    out = ret.retrieve("募集资金", top_k=5)
    ids = [r["chunk_id"] for r in out["results"]]
    assert "c4" in ids  # BM25 词命中


def test_no_rerank_fallback(chunks, fake_vectors):
    ret, _ = make_retriever(chunks, *fake_vectors, use_rerank=False)
    out = ret.retrieve("军用领域收入", top_k=3)
    assert len(out["results"]) == 3


# ---------- 3. 本地向量索引（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_local_vector_index_shapes(chunks, fake_vectors):
    mat, q = fake_vectors
    lvi = LocalVectorIndex(chunks, cache_path=None, embedder=None, vectors=mat)
    hits = lvi.search(q, top_k=2)
    assert len(hits) == 2
    assert hits[0]["chunk"]["chunk_id"] in ("c0", "c1")


def test_local_index_dim_mismatch_rebuild(chunks, fake_vectors):
    """向量缓存与分块数不一致时自动标记重建（matrix=None）"""
    mat, _ = fake_vectors
    lvi = LocalVectorIndex(chunks, cache_path=None, embedder=None, vectors=mat[:3])
    assert lvi._matrix is None


# ---------- 4. Reranker 降级逻辑（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_reranker_degradation_passthrough(chunks, fake_vectors):
    rr = Reranker()
    rr._failed = True  # 模拟模型不可用
    cands = [{"content": "a", "score": 1.0, "chunk_id": "x"},
             {"content": "b", "score": 2.0, "chunk_id": "y"}]
    out = rr.rerank("q", cands, top_k=1)
    assert len(out) == 1 and out[0]["chunk_id"] == "x"  # 保序截断
    assert "rerank_score" in out[0]


def test_tokenize_basic():
    toks = _tokenize("军用领域收入是多少")
    assert "军用" in "".join(toks) or any("军用" in t for t in toks)
