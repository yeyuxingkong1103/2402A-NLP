# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务

测试：混合检索融合算法（src/fusion.py）—— RRF / 加权平均 / 投票
不加载模型、不联网。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import fusion  # noqa: E402


def _hit(source: str, cid: str, score: float = 1.0) -> dict:
    return {"source": source, "chunk_id": cid, "text": f"text-{cid}", "score": score}


def _two_lists():
    dense = [_hit("a.pdf", "c1", 0.9), _hit("a.pdf", "c2", 0.8), _hit("a.pdf", "c3", 0.7)]
    fulltext = [_hit("a.pdf", "c3", 30.0), _hit("b.pdf", "c1", 20.0), _hit("a.pdf", "c1", 10.0)]
    return [("dense", dense), ("fulltext", fulltext)]


# ---------------------------------------------------------------------------
# 融合键
# ---------------------------------------------------------------------------
def test_doc_key_includes_source():
    """chunk_id 在多文档下会重复，融合键必须带 source（上游工单3 踩坑）。"""
    a = {"source": "a.pdf", "chunk_id": "c1"}
    b = {"source": "b.pdf", "chunk_id": "c1"}
    assert fusion.doc_key(a) != fusion.doc_key(b)


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------
def test_rrf_formula_and_order():
    out = fusion.rrf_fuse(_two_lists(), k=60)
    k = 60
    scores = {fusion.doc_key(h): h["rrf_score"] for h in out}
    assert abs(scores["a.pdf#c1"] - (1 / (k + 1) + 1 / (k + 3))) < 1e-6
    assert abs(scores["a.pdf#c3"] - (1 / (k + 3) + 1 / (k + 1))) < 1e-6
    # 两路都命中的排在只在单路命中的前面；最后一名是只在单路出现的块。
    # 注意 chunk_id 会跨文档重复（a.pdf#c1 与 b.pdf#c1 都叫 c1），必须用融合键判定。
    assert fusion.doc_key(out[0]) in ("a.pdf#c1", "a.pdf#c3")
    assert fusion.doc_key(out[-1]) in ("a.pdf#c2", "b.pdf#c1")
    assert out[-1]["rrf_score"] <= out[0]["rrf_score"]


def test_rrf_records_ranks_and_arms():
    out = fusion.rrf_fuse(_two_lists(), k=60)
    top = out[0]
    assert set(top["ranks"]) == {"dense", "fulltext"}
    assert set(top["arms"]) == {"dense", "fulltext"}


def test_rrf_weights_scale_contribution():
    plain = fusion.rrf_fuse(_two_lists(), k=60)
    weighted = fusion.rrf_fuse(_two_lists(), k=60,
                               weights={"dense": 1.0, "fulltext": 0.0})
    # 全文权重置 0 后，纯全文命中项（b.pdf#c1）应消失
    assert "b.pdf#c1" not in {fusion.doc_key(h) for h in weighted}
    assert len(plain) > len(weighted)


# ---------------------------------------------------------------------------
# 加权平均
# ---------------------------------------------------------------------------
def test_weighted_fuse_normalizes_scores():
    out = fusion.weighted_fuse(_two_lists(), weights={"dense": 0.5, "fulltext": 0.5})
    assert out, "应有结果"
    for h in out:
        assert 0.0 <= h["fused_score"] <= 1.0
    # 分数最高的应来自两路都靠前的文档
    assert out[0]["chunk_id"] in ("c1", "c3")


def test_weighted_fuse_respects_weight_zero():
    out = fusion.weighted_fuse(_two_lists(), weights={"dense": 1.0, "fulltext": 0.0})
    keys = {fusion.doc_key(h) for h in out}
    assert "b.pdf#c1" not in keys          # 只在全文臂出现的文档被排除


# ---------------------------------------------------------------------------
# 投票
# ---------------------------------------------------------------------------
def test_vote_fuse_borda_counts():
    out = fusion.vote_fuse(_two_lists(), weights={"dense": 1.0, "fulltext": 1.0})
    votes = {fusion.doc_key(h): h["votes"] for h in out}
    # dense 第 1 名（N=3）得 3 票；fulltext 第 1 名也得 3 票
    assert votes["a.pdf#c1"] == 3 + 1      # dense#1(3) + fulltext#3(1)
    assert votes["a.pdf#c3"] == 1 + 3      # dense#3(1) + fulltext#1(3)
    assert out[0]["votes"] >= out[-1]["votes"]


def test_vote_tiebreak_prefers_more_arms():
    """票数相同时，被更多通路召回的文档排前（tie-break = 命中路数）。"""
    dense = [_hit("a.pdf", "x"), _hit("a.pdf", "y")]
    ft = [_hit("a.pdf", "y")]
    out = fusion.vote_fuse([("dense", dense), ("fulltext", ft)])
    votes = {h["chunk_id"]: h["votes"] for h in out}
    assert votes["x"] == votes["y"] == 2      # x: dense#1(2票); y: dense#2(1)+ft#1(1)
    assert out[0]["chunk_id"] == "y"          # 同票 → 命中路数多者优先
    assert len(out[0]["arms"]) == 2


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def test_fuse_dispatch():
    for method, key in (("rrf", "rrf_score"), ("weighted", "fused_score"),
                        ("vote", "votes")):
        out = fusion.fuse(method, _two_lists())
        assert out and key in out[0], f"{method} 应产出 {key}"
    # 未知方法回退 RRF
    out = fusion.fuse("unknown", _two_lists())
    assert "rrf_score" in out[0]


def test_fuse_empty_inputs():
    assert fusion.fuse("rrf", []) == []
    assert fusion.fuse("rrf", [("dense", [])]) == []
