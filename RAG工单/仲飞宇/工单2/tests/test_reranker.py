"""BGEReranker 的分数合并与降级行为。

不发真请求：把 `_get_client` 换成回放固定响应的替身，覆盖服务端的几种"不标准"回法。
"""
from __future__ import annotations

import logging

import pytest

from app.core.config import Settings
from app.core.reranker import BGEReranker, ScoreFusionReranker


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _Client:
    """记录请求体，回放固定响应。"""

    def __init__(self, payload):
        self.payload = payload
        self.request = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, path, json=None):  # noqa: A002 - 跟 httpx 的签名一致
        self.request = json
        return _Resp(self.payload)


def _reranker(payload):
    r = BGEReranker(Settings(reranker="bge"))
    client = _Client(payload)
    r._get_client = lambda: client
    return r, client


def _cands():
    # score 是检索阶段的 RRF 融合分（0.01~0.13 量级），与 BGE 的 0~1 不同量纲
    return [
        {"id": "a", "text": "A", "score": 0.13},
        {"id": "b", "text": "B", "score": 0.05},
        {"id": "c", "text": "C", "score": 0.12},
    ]


def test_partial_scores_never_outrank_scored_candidates(caplog):
    """服务端只回部分 index 时，未打分的不能盖过「精排明确打了 0 分」的。

    把 BGE 的 relevance_score 与 RRF 融合分混进同一次 sort 就会出这种事：未打分项
    带着 0.05/0.12 的融合分，排到精排器打 0 分的候选前面，精排结论被整个盖掉。
    """
    r, _ = _reranker({"results": [{"index": 0, "relevance_score": 0.0}]})

    with caplog.at_level(logging.WARNING, logger="reranker"):
        out = r.rerank("q", _cands())

    assert [c["id"] for c in out] == ["a", "b", "c"], "未打分的必须整体排在已打分之后"
    assert out[0]["score"] == 0.0
    assert "只回了 1/3 条分数" in caplog.text, "部分打分要留痕，否则没人知道精排只覆盖了一部分"


def test_request_asks_for_scores_of_every_candidate():
    """显式传 top_n：有的服务商默认只回前 N 条，不传就注定有一批候选没有精排分。"""
    r, client = _reranker({"results": []})
    r.rerank("q", _cands())
    assert client.request["top_n"] == 3
    assert client.request["documents"] == ["A", "B", "C"]


def test_empty_results_falls_back_with_a_warning(caplog):
    """200 + 错误信封（无 results 键）以前静默退回 RRF——链路可用，但一条日志都没有。"""
    r, _ = _reranker({"error": {"message": "rate limited"}})

    with caplog.at_level(logging.WARNING, logger="reranker"):
        out = r.rerank("q", _cands())

    assert [c["id"] for c in out] == ["a", "c", "b"], "退回 RRF 时要保持融合分排序"
    assert "拿不到任何分数" in caplog.text
    assert "error" in caplog.text, "要把响应的键打出来，否则不知道上游回的是什么"


def test_malformed_entries_are_treated_as_unscored(caplog):
    """越界下标与缺字段的条目按「未打分」处理，不能拿别人的分数覆盖候选。"""
    r, _ = _reranker(
        {
            "results": [
                {"index": 99, "relevance_score": 0.9},  # 越界：不是任何一个候选
                {"index": 1, "relevance_score": 0.8},
                {"nope": 1},
                {"index": "x", "relevance_score": 1.0},
            ]
        }
    )

    with caplog.at_level(logging.WARNING, logger="reranker"):
        out = r.rerank("q", _cands())

    assert [c["id"] for c in out] == ["b", "a", "c"]
    assert out[0]["score"] == 0.8


def test_full_scores_replace_fusion_scores():
    """正常路径：全员都有精排分时，分数与顺序都换成 BGE 的。"""
    r, _ = _reranker(
        {
            "results": [
                {"index": 0, "relevance_score": 0.2},
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.5},
            ]
        }
    )

    out = r.rerank("q", _cands())

    assert [c["id"] for c in out] == ["b", "c", "a"]
    assert [c["score"] for c in out] == [0.9, 0.5, 0.2]


def test_candidates_not_mutated_in_place():
    """候选列表是调用方的：就地改 score 会让上游拿到被改写的融合分。"""
    r, _ = _reranker({"results": [{"index": 0, "relevance_score": 0.9}]})
    cands = _cands()
    r.rerank("q", cands)
    assert [c["score"] for c in cands] == [0.13, 0.05, 0.12]


def test_service_down_degrades_to_score_fusion(caplog):
    """服务不可用时按融合分排序，行为与 ScoreFusionReranker 一致。"""

    class _Boom:
        def __enter__(self):
            raise RuntimeError("connection refused")

        def __exit__(self, *exc):
            return False

    r = BGEReranker(Settings(reranker="bge"))
    r._get_client = lambda: _Boom()

    with caplog.at_level(logging.WARNING, logger="reranker"):
        out = r.rerank("q", _cands())

    assert [c["id"] for c in out] == [c["id"] for c in ScoreFusionReranker().rerank("q", _cands())]
    assert "不可用" in caplog.text


def test_empty_candidates_short_circuit():
    """没有候选时不该发请求（精排服务是逐对推理，白跑一趟还占超时预算）。"""
    r, client = _reranker({"results": []})
    assert r.rerank("q", []) == []
    assert client.request is None


@pytest.mark.parametrize("payload", [None, [], "boom", {"results": "not-a-list"}])
def test_weird_payload_shapes_degrade(payload):
    """响应形状完全不对时也只降级，不抛异常（精排挂掉不能拖垮整条问答链路）。"""
    r, _ = _reranker(payload)
    assert len(r.rerank("q", _cands())) == 3
