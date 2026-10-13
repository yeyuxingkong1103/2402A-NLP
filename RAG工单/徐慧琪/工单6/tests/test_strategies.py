# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务

测试：三种检索策略与配置（src/strategies.py）+ 运行时设置（src/runtime.py）

全部通过 monkeypatch 伪造召回，不加载任何模型、不联网。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, runtime, strategies  # noqa: E402


def _hit(cid: str, page: int = 0, score: float = 1.0) -> dict:
    return {"source": "a.pdf", "chunk_id": cid, "page_idx": page,
            "text": f"文本 {cid}", "score": score,
            "fulltext_score": score * 10}


@pytest.fixture(autouse=True)
def _clean_runtime():
    runtime.reset()
    yield
    runtime.reset()


# ---------------------------------------------------------------------------
# RetrievalConfig
# ---------------------------------------------------------------------------
def test_defaults_match_ticket_config_items():
    """工单配置项：Top K、召回候选数、精排后返回数、Score 阈值、Reranker 模型开关。"""
    cfg = strategies.RetrievalConfig()
    assert cfg.strategy in config.RETRIEVAL_STRATEGIES
    assert cfg.top_k_recall == config.TOP_K_RECALL == 20
    assert cfg.top_k_rerank == config.TOP_K_RERANK == 5
    assert cfg.score_threshold == config.SCORE_THRESHOLD == 0.0
    assert cfg.reranker_model in config.RERANKER_MODELS
    assert cfg.rerank_algorithm in config.RERANK_ALGORITHMS
    assert cfg.embedding_model in config.EMBEDDING_MODELS


def test_weight_normalization_for_rrf_uses_max():
    """rrf/vote 用「max 归一」：默认等权 → 无权重 RRF；非等权 → 强臂保持满权重。"""
    cfg = strategies.RetrievalConfig(vector_weight=0.5, fulltext_weight=0.5, fusion="rrf")
    assert cfg.normalized_weights() == {"dense": 1.0, "fulltext": 1.0}
    cfg = strategies.RetrievalConfig(vector_weight=0.7, fulltext_weight=0.35, fusion="rrf")
    assert cfg.normalized_weights() == {"dense": 1.0, "fulltext": 0.5}


def test_weight_normalization_for_weighted_uses_sum():
    cfg = strategies.RetrievalConfig(vector_weight=2.0, fulltext_weight=1.0,
                                     fusion="weighted")
    w = cfg.normalized_weights()
    assert abs(w["dense"] - 2 / 3) < 1e-9 and abs(w["fulltext"] - 1 / 3) < 1e-9


def test_zero_weights_fall_back_to_equal():
    cfg = strategies.RetrievalConfig(vector_weight=0, fulltext_weight=0)
    assert cfg.normalized_weights() == {"dense": 1.0, "fulltext": 1.0}


def test_with_overrides_only_applies_non_none():
    cfg = strategies.RetrievalConfig(strategy="hybrid")
    out = strategies.with_overrides(cfg, strategy="vector", top_k_recall=None)
    assert out.strategy == "vector"
    assert out.top_k_recall == cfg.top_k_recall


# ---------------------------------------------------------------------------
# 策略调度（伪造召回）
# ---------------------------------------------------------------------------
def test_vector_strategy_uses_only_dense(monkeypatch):
    calls = {"dense": 0, "fulltext": 0}

    def fake_dense(q, top_k, cfg):
        calls["dense"] += 1
        return [_hit("c1"), _hit("c2")]

    def fake_full(q, top_k, cfg):
        calls["fulltext"] += 1
        return [_hit("c3")]

    monkeypatch.setattr(strategies, "_dense_recall", fake_dense)
    monkeypatch.setattr(strategies, "_fulltext_recall", fake_full)

    res = strategies.retrieve("q", strategies.RetrievalConfig(
        strategy="vector", rerank_enabled=False, top_k_recall=5))
    assert calls == {"dense": 1, "fulltext": 0}
    assert res["info"]["arms"] == ["dense"]
    assert {h["chunk_id"] for h in res["hits"]} == {"c1", "c2"}


def test_fulltext_strategy_uses_only_fulltext(monkeypatch):
    calls = {"dense": 0, "fulltext": 0}
    monkeypatch.setattr(strategies, "_dense_recall",
                        lambda q, k, c: calls.__setitem__("dense", calls["dense"] + 1) or [])
    monkeypatch.setattr(strategies, "_fulltext_recall",
                        lambda q, k, c: calls.__setitem__("fulltext", calls["fulltext"] + 1)
                        or [_hit("f1"), _hit("f2")])
    res = strategies.retrieve("q", strategies.RetrievalConfig(
        strategy="fulltext", rerank_enabled=False, top_k_recall=5))
    assert calls["dense"] == 0 and calls["fulltext"] == 1
    assert res["info"]["arms"] == ["fulltext"]
    assert {h["chunk_id"] for h in res["hits"]} == {"f1", "f2"}


def test_hybrid_strategy_uses_both_arms_and_fuses(monkeypatch):
    monkeypatch.setattr(strategies, "_dense_recall",
                        lambda q, k, c: [_hit("shared"), _hit("d1")])
    monkeypatch.setattr(strategies, "_fulltext_recall",
                        lambda q, k, c: [_hit("shared"), _hit("f1")])
    res = strategies.retrieve("q", strategies.RetrievalConfig(
        strategy="hybrid", rerank_enabled=False, top_k_recall=10,
        fusion="rrf"))
    assert res["info"]["arms"] == ["dense", "fulltext"]
    assert res["info"]["recall_counts"] == {"dense": 2, "fulltext": 2}
    assert res["hits"][0]["chunk_id"] == "shared"       # 两路都命中 → 融合分最高
    assert res["info"]["fusion"] == "rrf"


def test_hybrid_fusion_methods_switch(monkeypatch):
    monkeypatch.setattr(strategies, "_dense_recall",
                        lambda q, k, c: [_hit("d1", score=0.9), _hit("d2", score=0.1)])
    monkeypatch.setattr(strategies, "_fulltext_recall",
                        lambda q, k, c: [_hit("f1", score=9.0)])
    for method in ("rrf", "weighted", "vote"):
        res = strategies.retrieve("q", strategies.RetrievalConfig(
            strategy="hybrid", rerank_enabled=False, fusion=method))
        assert res["hits"], f"{method} 应返回结果"
        assert res["info"]["fusion"] == method


def test_unknown_strategy_raises(monkeypatch):
    monkeypatch.setattr(strategies, "_dense_recall", lambda q, k, c: [])
    with pytest.raises(ValueError):
        strategies.retrieve("q", strategies.RetrievalConfig(strategy="nope"))


def test_empty_query_returns_empty():
    res = strategies.retrieve("   ", strategies.RetrievalConfig())
    assert res["hits"] == [] and "error" in res["info"]


# ---------------------------------------------------------------------------
# 精排阶段：deferred / 关闭 / 阈值
# ---------------------------------------------------------------------------
def test_do_rerank_false_defers_to_caller(monkeypatch):
    monkeypatch.setattr(strategies, "_dense_recall", lambda q, k, c: [_hit("c1")])
    res = strategies.retrieve("q", strategies.RetrievalConfig(strategy="vector"),
                              do_rerank=False)
    assert res["info"]["rerank"]["algorithm"] == "deferred"
    assert res["info"]["rerank_ms"] == 0.0


def test_rerank_disabled_truncates_to_top_k(monkeypatch):
    monkeypatch.setattr(strategies, "_dense_recall",
                        lambda q, k, c: [_hit(f"c{i}") for i in range(10)])
    res = strategies.retrieve("q", strategies.RetrievalConfig(
        strategy="vector", rerank_enabled=False, top_k_recall=10, top_k_rerank=3))
    assert len(res["hits"]) == 3
    assert res["info"]["rerank"]["degraded"] is True


def test_fulltext_rerank_switch(monkeypatch):
    monkeypatch.setattr(strategies, "_fulltext_recall",
                        lambda q, k, c: [_hit(f"f{i}") for i in range(6)])
    res = strategies.retrieve("q", strategies.RetrievalConfig(
        strategy="fulltext", fulltext_rerank=False, top_k_rerank=2))
    assert len(res["hits"]) == 2
    assert res["info"]["rerank"]["algorithm"] == "off"


# ---------------------------------------------------------------------------
# 运行时设置
# ---------------------------------------------------------------------------
def test_runtime_update_and_describe():
    runtime.update(strategy="vector", top_k_recall=30, fusion="vote")
    d = runtime.describe()
    assert d["strategy"] == "vector" and d["top_k_recall"] == 30 and d["fusion"] == "vote"
    assert d["revision"] >= 1


def test_runtime_rejects_unknown_values():
    with pytest.raises(ValueError):
        runtime.update(strategy="nope")
    with pytest.raises(ValueError):
        runtime.update(fusion="nope")
    with pytest.raises(ValueError):
        runtime.update(rerank_algorithm="nope")
    with pytest.raises(ValueError):
        runtime.update(reranker_model="nope")
    with pytest.raises(ValueError):
        runtime.update(what_is_this=1)


def test_runtime_get_returns_snapshot():
    runtime.update(strategy="fulltext")
    snap = runtime.get()
    runtime.update(strategy="hybrid")
    assert snap.strategy == "fulltext"      # 快照不被后续修改影响


def test_describe_lists_all_options():
    d = strategies.describe()
    assert set(d["strategies"]) == {"vector", "fulltext", "hybrid"}
    assert set(d["fusion_methods"]) == set(config.FUSION_METHODS)
    assert len(d["reranker_models"]) >= 2
    assert len(d["embedding_models"]) >= 2
