# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务

测试：Reranker 精排注册表（src/rerankers.py）

重点验证不依赖大模型的部分（可在无 GPU 的机器上跑）：
  · 注册表与模型可切换（≥2 种）
  · 精排前后 Top-K 名次变化记录（工单硬性要求2）
  · 分数阈值 / top_k / 降级路径
  · TF-IDF 重排器、用户反馈自适应重排器（对比基线）
  · 反馈落盘与统计
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, rerankers  # noqa: E402


def _cands() -> list[dict]:
    return [
        {"source": "a.pdf", "chunk_id": "c1", "text": "公司注册资本为5,520.00万元。",
         "rrf_score": 0.01},
        {"source": "a.pdf", "chunk_id": "c2", "text": "本次发行股数为1,670万股，占发行后总股本25.04%。",
         "rrf_score": 0.02},
        {"source": "a.pdf", "chunk_id": "c3", "text": "公司主营电子元器件的分销与技术服务。",
         "rrf_score": 0.005},
    ]


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
def test_model_registry_has_at_least_two_rerankers():
    """工单硬性要求：Reranker 模型 ≥2 种可切换。"""
    assert len(config.RERANKER_MODELS) >= 2
    assert "bge-reranker-v2-m3" in config.RERANKER_MODELS
    assert "bge-reranker-base" in config.RERANKER_MODELS


def test_rerank_algorithms_cover_ticket_requirement():
    """PDF 工单要求 ≥3 种重排算法：LLM / TF-IDF / 用户反馈自适应（+生产默认模型）。"""
    for algo in ("model", "llm", "tfidf", "adaptive"):
        assert algo in config.RERANK_ALGORITHMS


def test_info_lists_models_and_paths():
    info = rerankers.info()
    assert info["model"] in config.RERANKER_MODELS
    for name, m in info["models"].items():
        assert "path" in m and "exists" in m


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        rerankers._model_spec("no-such-model")


# ---------------------------------------------------------------------------
# 关闭精排 / 降级路径
# ---------------------------------------------------------------------------
def test_disabled_rerank_truncates_and_marks_degraded(monkeypatch):
    monkeypatch.setattr(config, "RERANKER_ENABLED", False)
    res = rerankers.rerank_detailed("注册资本", _cands(), top_k=2)
    assert len(res["hits"]) == 2
    assert res["info"]["degraded"] is True
    assert "关闭" in (res["info"]["degrade_reason"] or "")


def test_unavailable_model_degrades_without_raising(monkeypatch):
    """模型不可用（路径缺失/冷却）时必须降级为按召回序截断，绝不抛异常。"""
    monkeypatch.setattr(rerankers, "is_available", lambda name=None: False)
    res = rerankers.rerank_detailed("注册资本", _cands(), algorithm="model", top_k=2)
    assert len(res["hits"]) == 2
    assert res["info"]["degraded"] is True
    assert "不可用" in res["info"]["degrade_reason"]


# ---------------------------------------------------------------------------
# TF-IDF 重排器（对比基线）
# ---------------------------------------------------------------------------
def test_tfidf_reranker_orders_by_lexical_overlap():
    res = rerankers.rerank_detailed("公司注册资本是多少", _cands(),
                                    algorithm="tfidf", top_k=3)
    assert res["hits"][0]["chunk_id"] == "c1"
    assert res["info"]["algorithm"] == "tfidf"
    assert res["info"]["n_in"] == 3


def test_tfidf_reranker_records_rank_changes():
    res = rerankers.rerank_detailed("本次发行股数", _cands(), algorithm="tfidf", top_k=3)
    info = res["info"]
    assert info["n_in"] == 3 and info["n_out"] == 3
    assert isinstance(info["rank_changes"], list)
    # c2 从召回第 2 升到第 1 → 必然记录一条变化
    assert info["n_changed"] >= 1


# ---------------------------------------------------------------------------
# 用户反馈自适应重排器（对比基线）
# ---------------------------------------------------------------------------
def test_feedback_record_and_stats(tmp_path, monkeypatch):
    store = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(config, "FEEDBACK_STORE", str(store))
    rerankers.load_feedback(force=True)

    rerankers.record_feedback("注册资本是多少", _cands()[0], True, strategy="hybrid")
    rerankers.record_feedback("注册资本是多少", _cands()[2], False, strategy="hybrid")
    s = rerankers.feedback_stats()
    assert s["n"] == 2 and s["helpful"] == 1 and s["unhelpful"] == 1
    rows = [json.loads(l) for l in store.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows[0]["chunk_id"] == "c1" and rows[0]["helpful"] is True


def test_adaptive_reranker_boosts_liked_chunk(tmp_path, monkeypatch):
    store = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(config, "FEEDBACK_STORE", str(store))
    rerankers.load_feedback(force=True)

    # 给召回第 3 名（c3）点赞，相似问法下它应被提升
    rerankers.record_feedback("公司主营是什么", _cands()[2], True, strategy="hybrid")
    res = rerankers.rerank_detailed("公司主营是什么", _cands(),
                                    algorithm="adaptive", top_k=3)
    assert res["hits"][0]["chunk_id"] == "c3"
    assert res["hits"][0].get("adaptive_boost", 0) > 0


def test_adaptive_reranker_without_feedback_keeps_order(tmp_path, monkeypatch):
    store = tmp_path / "feedback.jsonl"          # 不创建文件
    monkeypatch.setattr(config, "FEEDBACK_STORE", str(store))
    rerankers.load_feedback(force=True)
    res = rerankers.rerank_detailed("注册资本", _cands(), algorithm="adaptive", top_k=3)
    assert [h["chunk_id"] for h in res["hits"]] == ["c1", "c2", "c3"]


# ---------------------------------------------------------------------------
# 阈值与边界
# ---------------------------------------------------------------------------
def test_score_threshold_filters_after_rerank():
    res = rerankers.rerank_detailed("本次发行股数", _cands(), algorithm="tfidf",
                                    top_k=3, score_threshold=0.99)
    assert all((h.get("rerank_score") or 0) >= 0.99 for h in res["hits"])


def test_empty_candidates_returns_empty():
    res = rerankers.rerank_detailed("q", [], algorithm="tfidf")
    assert res["hits"] == [] and res["info"]["n_in"] == 0


def test_unknown_algorithm_degrades_gracefully():
    res = rerankers.rerank_detailed("q", _cands(), algorithm="nope", top_k=2)
    assert len(res["hits"]) == 2
    assert res["info"]["degraded"] is True
