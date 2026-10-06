# -*- coding: utf-8 -*-
"""V3 链路测试

重点验证**降级行为**：重排/改写故障时不得中断问答，必须退回 V2 顺序。
"""

from typing import Any, Dict, List

import pytest

from backend.rag_pipeline import v3_pipeline as v3


def _cand(cid: str, page: int = 1) -> Dict[str, Any]:
    return {
        "chunk_id": cid, "score": 0.5, "page_no": page, "page_nums": [page],
        "file_name": "x.pdf", "content": "内容" + cid,
        "content_type": "text", "section_title": "",
    }


class _FakeV2:
    """假的 V2 检索，按查询返回预设候选"""

    def __init__(self, table: Dict[str, List[Dict[str, Any]]]) -> None:
        self._table = table
        self.calls: List[str] = []

    def retrieve(self, question: str, *, top_k=None):
        self.calls.append(question)
        return list(self._table.get(question, []))


class _FakeReranker:
    def __init__(self, order: List[str], boom: bool = False) -> None:
        self._order = order
        self._boom = boom
        # 调用记录：用于验证「重排关闭时确实没调用重排」。
        # 只断言返回顺序无法证明这一点 —— 一个会抛 RerankerNotAvailableError
        # 的重排器被调用后同样会降级出相同顺序，断言会恒真。
        self.calls: List[str] = []

    def rerank(self, query, candidates, *, top_k=None):
        self.calls.append(query)
        if self._boom:
            raise v3.RerankerNotAvailableError("模拟重排故障")
        index = {cid: i for i, cid in enumerate(self._order)}
        out = sorted(candidates, key=lambda c: index.get(c["chunk_id"], 999))
        return out[:top_k] if top_k else out


class _BoomValueReranker:
    """抛非 RerankerNotAvailableError 的重排器（覆盖宽 except 分支）"""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def rerank(self, query, candidates, *, top_k=None):
        self.calls.append(query)
        raise ValueError("模拟非预期的重排异常")


class _FakeRewriter:
    def __init__(self, queries: List[str]) -> None:
        self._queries = queries

    def rewrite(self, question: str) -> List[str]:
        return list(self._queries)


class _BoomRewriter:
    """总会抛异常的改写器（覆盖 rewrite 抛错时的降级分支）"""

    def rewrite(self, question: str) -> List[str]:
        raise RuntimeError("模拟改写故障")


class _FlakyV2:
    """对指定查询抛异常的 V2 替身（模拟单路检索的瞬时故障）"""

    def __init__(self, table: Dict[str, List[Dict[str, Any]]],
                 boom_queries: List[str]) -> None:
        self._table = table
        self._boom = set(boom_queries)
        self.calls: List[str] = []

    def retrieve(self, question: str, *, top_k=None):
        self.calls.append(question)
        if question in self._boom:
            raise RuntimeError("模拟检索故障")
        return list(self._table.get(question, []))


def _patch(monkeypatch, *, v2, reranker=None, rewriter=None,
           rerank_enabled=True, rewrite_enabled=False):
    monkeypatch.setattr(v3.settings, "rerank_enabled", rerank_enabled, raising=False)
    monkeypatch.setattr(v3.settings, "query_rewrite_enabled", rewrite_enabled, raising=False)
    monkeypatch.setattr(v3.settings, "rerank_candidate_k", 10, raising=False)
    monkeypatch.setattr(v3.settings, "retrieve_top_k", 5, raising=False)
    monkeypatch.setattr(v3, "get_reranker", lambda: reranker or _FakeReranker([]))
    monkeypatch.setattr(v3, "get_rewriter", lambda: rewriter or _FakeRewriter([]))
    return v2


def test_仅原问题时等同于单路检索(monkeypatch):
    fake = _patch(monkeypatch, v2=_FakeV2({"q": [_cand("a"), _cand("b")]}))
    p = v3.V3Pipeline.__new__(v3.V3Pipeline)
    p._v2 = fake
    out = p.retrieve("q", top_k=5)
    assert [c["chunk_id"] for c in out] == ["a", "b"]
    assert fake.calls == ["q"]


def test_重排改变最终顺序(monkeypatch):
    fake = _patch(
        monkeypatch,
        v2=_FakeV2({"q": [_cand("a"), _cand("b"), _cand("c")]}),
        reranker=_FakeReranker(["c", "a", "b"]),
    )
    p = v3.V3Pipeline.__new__(v3.V3Pipeline)
    p._v2 = fake
    out = p.retrieve("q", top_k=3)
    assert [c["chunk_id"] for c in out] == ["c", "a", "b"]


def test_重排故障时降级为原顺序且不抛异常(monkeypatch):
    fake = _patch(
        monkeypatch,
        v2=_FakeV2({"q": [_cand("a"), _cand("b")]}),
        reranker=_FakeReranker([], boom=True),
    )
    p = v3.V3Pipeline.__new__(v3.V3Pipeline)
    p._v2 = fake
    out = p.retrieve("q", top_k=5)
    assert [c["chunk_id"] for c in out] == ["a", "b"]


def test_重排关闭时不调用重排(monkeypatch):
    boom = _FakeReranker([], boom=True)
    fake = _patch(
        monkeypatch,
        v2=_FakeV2({"q": [_cand("a")]}),
        reranker=boom,
        rerank_enabled=False,
    )
    p = v3.V3Pipeline.__new__(v3.V3Pipeline)
    p._v2 = fake
    out = p.retrieve("q", top_k=5)
    assert [c["chunk_id"] for c in out] == ["a"]
    # 真验证：必须断言「一次都没调用」。仅断言返回顺序是恒真的 ——
    # 这个会抛异常的重排器即使被调用，降级后同样返回 ["a"]。
    assert boom.calls == []


def test_重排抛非预期异常时也退回原顺序(monkeypatch):
    """RerankerNotAvailableError 之外的异常同样不得中断问答（宽 except 分支）"""
    boom = _BoomValueReranker()
    fake = _patch(
        monkeypatch,
        v2=_FakeV2({"q": [_cand("a"), _cand("b")]}),
        reranker=boom,
    )
    p = v3.V3Pipeline.__new__(v3.V3Pipeline)
    p._v2 = fake
    out = p.retrieve("q", top_k=5)
    assert [c["chunk_id"] for c in out] == ["a", "b"]
    assert boom.calls == ["q"]          # 确实调用了，失败后才降级


def test_改写器抛异常时降级为只用原问题(monkeypatch):
    fake = _patch(
        monkeypatch,
        v2=_FakeV2({"q": [_cand("a")]}),
        rewriter=_BoomRewriter(),
        rewrite_enabled=True,
        rerank_enabled=False,
    )
    p = v3.V3Pipeline.__new__(v3.V3Pipeline)
    p._v2 = fake
    out = p.retrieve("q", top_k=5)
    assert [c["chunk_id"] for c in out] == ["a"]
    assert fake.calls == ["q"]


def test_改写查询检索失败时不中断问答(monkeypatch):
    """改写那一路检索抛错时，原问题的结果仍须正常返回（ADR-024）"""
    fake = _patch(
        monkeypatch,
        v2=_FlakyV2({"q": [_cand("a")]}, boom_queries=["改写"]),
        rewriter=_FakeRewriter(["改写"]),
        rewrite_enabled=True,
        rerank_enabled=False,
    )
    p = v3.V3Pipeline.__new__(v3.V3Pipeline)
    p._v2 = fake
    out = p.retrieve("q", top_k=5)
    assert set(fake.calls) == {"q", "改写"}    # 两条都试过
    assert [c["chunk_id"] for c in out] == ["a"]


def test_全部查询检索失败时返回空列表(monkeypatch):
    """所有查询都失败 -> 返回 []，交由上层走「检索无结果」路径"""
    fake = _patch(
        monkeypatch,
        v2=_FlakyV2({}, boom_queries=["q", "改写"]),
        rewriter=_FakeRewriter(["改写"]),
        rewrite_enabled=True,
        rerank_enabled=False,
    )
    p = v3.V3Pipeline.__new__(v3.V3Pipeline)
    p._v2 = fake
    assert p.retrieve("q", top_k=5) == []


def test_改写开启时会对多个查询分别检索(monkeypatch):
    fake = _patch(
        monkeypatch,
        v2=_FakeV2({"q": [_cand("a")], "改写": [_cand("b")]}),
        rewriter=_FakeRewriter(["改写"]),
        rewrite_enabled=True,
        rerank_enabled=False,
    )
    p = v3.V3Pipeline.__new__(v3.V3Pipeline)
    p._v2 = fake
    out = p.retrieve("q", top_k=5)
    assert set(fake.calls) == {"q", "改写"}
    assert {c["chunk_id"] for c in out} == {"a", "b"}


def test_改写关闭时不调用改写器(monkeypatch):
    fake = _patch(
        monkeypatch,
        v2=_FakeV2({"q": [_cand("a")]}),
        rewriter=_FakeRewriter(["改写"]),
        rewrite_enabled=False,
        rerank_enabled=False,
    )
    p = v3.V3Pipeline.__new__(v3.V3Pipeline)
    p._v2 = fake
    p.retrieve("q", top_k=5)
    assert fake.calls == ["q"]


def test_检索无结果时返回空列表(monkeypatch):
    fake = _patch(monkeypatch, v2=_FakeV2({"q": []}), rerank_enabled=False)
    p = v3.V3Pipeline.__new__(v3.V3Pipeline)
    p._v2 = fake
    assert p.retrieve("q", top_k=5) == []


# ---------------------------------------------------------------------------
# 溯源顺序（V3 覆盖 _build_sources）
# ---------------------------------------------------------------------------

def _scored(cid: str, page: int, score: float, rerank_score=None) -> Dict[str, Any]:
    """构造候选：score 为 RRF 分，rerank_score 为可选的（顺序相反的）重排分"""
    cand = _cand(cid, page)
    cand["score"] = score
    if rerank_score is not None:
        cand["rerank_score"] = rerank_score
    return cand


def test_有重排分时溯源按重排分排序():
    """sources 必须跟随重排顺序，而不是重排前的 RRF 分顺序"""
    a = _scored("a", 1, 0.9, 0.1)   # RRF 高、重排低
    b = _scored("b", 2, 0.1, 0.9)   # RRF 低、重排高
    out = v3.V3Pipeline._build_sources([a, b])
    assert [s["chunk_id"] for s in out] == ["b", "a"]
    # 输出字段未变：score 仍是 RRF 分，重排分不进响应体
    assert [s["score"] for s in out] == [0.1, 0.9]
    assert "rerank_score" not in out[0]


def test_无重排分时溯源按原始分数排序():
    """降级/关闭重排时与 V1/V2 行为一致"""
    a = _scored("a", 1, 0.9)
    b = _scored("b", 2, 0.1)
    out = v3.V3Pipeline._build_sources([a, b])
    assert [s["chunk_id"] for s in out] == ["a", "b"]


def test_溯源去重按重排分取同页最优():
    """同一 (文件名, 页码) 取「最优」时判据与排序一致，用 rerank_score。

    构造的正是真实数据上出现过的失配：RRF 分更高的那个 rerank 分反而更低。
    此时必须保留 **rerank 分更高**的那个 —— 否则用户看到的来源首位
    就不是答案实际依据的首条上下文，V3 内部自相矛盾。
    """
    high_rrf = _scored("high_rrf", 1, 0.9, 0.1)     # RRF 高、重排低
    high_rerank = _scored("high_rerank", 1, 0.2, 0.9)  # RRF 低、重排高
    out = v3.V3Pipeline._build_sources([high_rrf, high_rerank])
    assert len(out) == 1
    assert out[0]["chunk_id"] == "high_rerank"


def test_无重排分时溯源与_V1_逐字段一致():
    """无 rerank_score 时（重排关闭/降级），V3 的溯源与 V1 原实现完全一致"""
    from backend.rag_pipeline.v1_pipeline import V1Pipeline

    # 含同页多块（触发去重）与跨页块，充分覆盖去重 + 排序两条路径
    ordered = [
        _scored("a", 3, 0.2),
        _scored("b", 1, 0.9),
        _scored("c", 1, 0.4),   # 与 b 同页，分更低 -> 应被去重淘汰
        _scored("d", 2, 0.5),
    ]
    assert v3.V3Pipeline._build_sources(ordered) == V1Pipeline._build_sources(ordered)
