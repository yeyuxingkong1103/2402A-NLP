# -*- coding: utf-8 -*-
"""RRF 融合单元测试"""

import pytest

from backend.config import settings
from backend.rag_pipeline.rrf import fuse


def test_单通道时退化为原顺序():
    result = fuse([["a", "b", "c"]], k=60, top_k=3)
    assert [cid for cid, _ in result] == ["a", "b", "c"]


def test_分数按公式计算():
    # rank=1 时 1/(60+1)；rank=2 时 1/(60+2)
    result = dict(fuse([["a", "b"]], k=60, top_k=2))
    assert result["a"] == pytest.approx(1 / 61)
    assert result["b"] == pytest.approx(1 / 62)


def test_跨通道命中获得更高分():
    # x 在两个通道都出现，应高于只在一个通道中排第一的 a
    result = dict(fuse([["a", "x"], ["b", "x"]], k=60, top_k=3))
    assert result["x"] > result["a"]
    assert result["x"] == pytest.approx(1 / 62 + 1 / 62)


def test_未出现在某通道不计该项分():
    result = dict(fuse([["a"], ["b"]], k=60, top_k=5))
    assert result["a"] == pytest.approx(1 / 61)
    assert result["b"] == pytest.approx(1 / 61)


def test_同通道内重复块只记首次名次():
    result = dict(fuse([["a", "a", "a"]], k=60, top_k=3))
    assert result["a"] == pytest.approx(1 / 61)


def test_分数相同时按_chunk_id_升序保证可复现():
    result = fuse([["b"], ["a"]], k=60, top_k=5)
    assert [cid for cid, _ in result] == ["a", "b"]


def test_空输入返回空列表():
    assert fuse([], k=60, top_k=5) == []
    assert fuse([[], []], k=60, top_k=5) == []


def test_top_k_截断():
    result = fuse([["a", "b", "c", "d"]], k=60, top_k=2)
    assert len(result) == 2


def test_k_缺省时取自配置():
    """k 必须来自 settings.rrf_k，不得硬编码"""
    expected = dict(fuse([["a"]], k=settings.rrf_k, top_k=1))["a"]
    assert dict(fuse([["a"]], top_k=1))["a"] == pytest.approx(expected)


def test_k_非法值报错():
    with pytest.raises(ValueError):
        fuse([["a"]], k=0, top_k=1)
