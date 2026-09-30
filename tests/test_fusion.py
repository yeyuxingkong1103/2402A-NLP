"""加权 RRF 融合测试。"""

from __future__ import annotations

import pytest

from role_rag.retrieval.fusion import FusedItem, dedupe, normalize_scores, weighted_rrf


def test_rrf_basic_order():
    fused = weighted_rrf({"dense": ["a", "b"], "bm25": ["b", "c"]}, {"dense": 1.0, "bm25": 1.0}, k=60)
    assert [item.key for item in fused][:2] == ["b", "a"]
    best = fused[0]
    assert best.routes["dense"]["rank"] == 2.0
    assert best.routes["bm25"]["rank"] == 1.0
    assert best.score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_weights_change_result():
    rankings = {"dense": ["a"], "bm25": ["b"]}
    fused = weighted_rrf(rankings, {"dense": 0.9, "bm25": 0.1}, k=60)
    assert fused[0].key == "a"
    fused2 = weighted_rrf(rankings, {"dense": 0.1, "bm25": 0.9}, k=60)
    assert fused2[0].key == "b"


def test_zero_weight_route_ignored():
    fused = weighted_rrf({"dense": ["a"], "bm25": ["b"]}, {"dense": 1.0, "bm25": 0.0}, k=60)
    assert [item.key for item in fused] == ["a"]


def test_raw_scores_attached():
    fused = weighted_rrf({"dense": ["a"]}, {"dense": 1.0}, k=60, scores={"dense": {"a": 0.87}})
    assert fused[0].routes["dense"]["raw"] == pytest.approx(0.87)


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        weighted_rrf({"dense": ["a"]}, k=0)


def test_missing_weights_default_to_one():
    fused = weighted_rrf({"dense": ["a"]}, k=60)
    assert fused[0].score == pytest.approx(1 / 61)


def test_dedupe_keeps_order():
    assert dedupe(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_normalize_scores():
    assert normalize_scores([1, 2, 3]) == [0.0, 0.5, 1.0]
    assert normalize_scores([5, 5]) == [1.0, 1.0]
    assert normalize_scores([]) == []


def test_fused_item_to_dict():
    item = FusedItem(key="a", score=0.5, routes={"dense": {"rank": 1.0}})
    payload = item.to_dict()
    assert payload["key"] == "a" and payload["score"] == 0.5
