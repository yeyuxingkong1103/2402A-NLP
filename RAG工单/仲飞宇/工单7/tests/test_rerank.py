# 工单编号：人工智能NLP-RAG-混合检索任务
# 工单06 - 混合检索（向量检索 / 全文检索 / 混合检索）
"""
重排器测试。**不碰 Milvus、不碰 Ollama**（LLM 用假 client 验调用契约）。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import rerank as R                                   # noqa: E402
from app.core.profiles import RetrievalProfile, get_profile        # noqa: E402
from app.core.vectorstore import SearchHit                         # noqa: E402


def _hit(cid: int, text: str, *, page: str = "1-1-1", score: float = 0.8,
         kind: str = "cosine", section: str = "") -> SearchHit:
    return SearchHit(chunk_id=cid, score=score, content=text, page_no=cid,
                     page_label=page, chunk_type="text", section_path=section,
                     doc_name="招股说明书1.pdf", chunk_index=cid,
                     doc_id="d1", score_kind=kind)


def _run(coro):
    return asyncio.run(coro)


# ======================================================================
# lexical（工单02 的原式）
# ======================================================================
def test_lexical_score_matches_legacy_formula():
    """词法融合分必须与工单02 的公式一致 —— 它是 16/16 的地基。"""
    h = _hit(1, "公司注册资本为5,520万元", score=0.6)
    qt = {"注册资本"}
    expect = 0.5 * ((0.6 + 1) / 2) + 0.5 * (1 / 1)
    assert R.lexical_score(h, qt, 0.5) == pytest.approx(expect)


def test_lexical_normalizes_non_cosine_by_span():
    """非 COSINE 分数**绝不能**套 `(score+1)/2`，要按候选集 min-max 归一化。

    RRF 的分只有 0.016 量级，`(0.016+1)/2 ≈ 0.508` 对候选集里所有块几乎相同 ——
    `w_dense * dense` 退化成常数项，排序悄悄只剩词覆盖率在起作用，**且不报错**。
    实测这一处就是混合检索一度全面劣于纯向量的原因（页召回 60.4% → 38.5%）。
    """
    h = _hit(1, "随便", score=12.5, kind="bm25")
    # span=10, lo=2.5 → (12.5-2.5)/10 = 1.0；qt 只有一个词且命中 → lex=1.0
    assert R.lexical_score(h, {"随便"}, 0.5, span=10.0, lo=2.5) == pytest.approx(1.0)
    # 不命中查询词时只剩归一化分量
    assert R.lexical_score(h, {"别的词"}, 0.5, span=10.0, lo=2.5) == pytest.approx(0.5)


def test_lexical_weights_none_kind_at_zero():
    """`none` 尺度（关键词路独有、向量没召回）在稠密分量上记 0。"""
    h = _hit(1, "军用领域的收入", score=0.02, kind="fused")
    h2 = _hit(1, "军用领域的收入", score=0.02, kind="fused")
    object.__setattr__(h2, "base_score", 0.0)
    object.__setattr__(h2, "base_kind", "none")
    assert R.lexical_score(h2, {"军用"}, 0.5, span=1.0, lo=0.0) == pytest.approx(0.5)


def test_lexical_uses_base_score_when_present():
    """融合过的命中要用 `base_score`（融合前的原始 COSINE），不是融合分。"""
    from dataclasses import replace
    h = _hit(1, "随便", score=0.02, kind="fused")      # 融合分很小
    hb = replace(h, base_score=0.6, base_kind="cosine")
    # 用 base=(0.6+1)/2=0.8；若误用融合分则是 (0.02+1)/2≈0.51
    assert R.lexical_score(hb, {"别的"}, 0.5) == pytest.approx(0.5 * 0.8)


def test_lexical_reorders_by_overlap():
    a = _hit(1, "完全无关的内容", score=0.9)
    b = _hit(2, "公司注册资本为5,520万元", score=0.5)
    out, _ = _run(R.rerank([a, b], "注册资本", profile=get_profile("delivered")))
    assert [h.chunk_id for h in out][0] == 2


# ======================================================================
# 只改顺序，不增删
# ======================================================================
def test_rerank_only_reorders():
    hits = [_hit(i, f"内容{i}", score=0.5 + i / 100) for i in range(1, 6)]
    for name in ("lexical", "tfidf", "feedback", "none"):
        out, _ = _run(R.rerank(hits, "内容", profile=RetrievalProfile(name="t", reranker=name)))
        assert sorted(h.chunk_id for h in out) == sorted(h.chunk_id for h in hits), name


# ======================================================================
# feedback
# ======================================================================
def test_feedback_noop_when_file_missing(monkeypatch, tmp_path):
    """**冷启动必须等价于不重排** —— 这是 16/16 的保护闸门。"""
    monkeypatch.setattr(R, "_feedback_file", lambda: tmp_path / "不存在.jsonl")
    hits = [_hit(1, "a", page="1-1-21"), _hit(2, "b", page="1-1-22")]
    out, info = _run(R.rerank(hits, "x",
                              profile=RetrievalProfile(name="t", reranker="feedback")))
    assert [h.chunk_id for h in out] == [1, 2]
    assert info.feedback_n == 0


def test_feedback_requires_min_support(monkeypatch, tmp_path):
    """样本不足时 boost 表为空（一条反馈不足以重排）。"""
    p = tmp_path / "fb.jsonl"
    p.write_text(json.dumps({"rating": "up",
                             "citations": [{"page_label": "1-1-21",
                                            "section_path": "", "is_neighbor": False}]},
                            ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(R, "_feedback_file", lambda: p)
    boosts, n = R.feedback_boosts(min_support=3, max_boost=0.15)
    assert n == 1 and boosts == {}


def test_feedback_boost_is_capped_and_signed(monkeypatch, tmp_path):
    p = tmp_path / "fb.jsonl"
    recs = []
    for _ in range(4):
        recs.append({"rating": "up", "citations": [
            {"page_label": "1-1-21", "section_path": "第五节", "is_neighbor": False}]})
    for _ in range(2):
        recs.append({"rating": "down", "citations": [
            {"page_label": "1-1-22", "section_path": "第六节", "is_neighbor": False}]})
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs),
                 encoding="utf-8")
    monkeypatch.setattr(R, "_feedback_file", lambda: p)
    boosts, n = R.feedback_boosts(min_support=2, max_boost=0.15)
    assert n == 6
    key_good = R._feedback_key("1-1-21", "第五节")
    key_bad = R._feedback_key("1-1-22", "第六节")
    assert boosts[key_good] == pytest.approx(0.15)      # 4👍 0👎 → 上限
    assert boosts[key_bad] == pytest.approx(-0.15)      # 0👍 1👎 → 下限
    assert all(abs(v) <= 0.15 + 1e-9 for v in boosts.values())


def test_feedback_ignores_neighbor_citations(monkeypatch, tmp_path):
    """邻块引用不参与反馈统计（它本来就不是答案依据）。"""
    p = tmp_path / "fb.jsonl"
    p.write_text(json.dumps({"rating": "up", "citations": [
        {"page_label": "1-1-99", "section_path": "", "is_neighbor": True}]},
        ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(R, "_feedback_file", lambda: p)
    boosts, n = R.feedback_boosts(min_support=1, max_boost=0.15)
    assert n == 1 and boosts == {}


def test_feedback_promotes_upvoted_page(monkeypatch, tmp_path):
    p = tmp_path / "fb.jsonl"
    p.write_text("\n".join(json.dumps(
        {"rating": "up", "citations": [{"page_label": "1-1-22",
                                        "section_path": "", "is_neighbor": False}]},
        ensure_ascii=False) for _ in range(3)), encoding="utf-8")
    monkeypatch.setattr(R, "_feedback_file", lambda: p)
    hits = [_hit(1, "a", page="1-1-21", score=1.0),
            _hit(2, "b", page="1-1-22", score=0.95)]
    out, info = _run(R.rerank(hits, "x",
                              profile=RetrievalProfile(name="t", reranker="feedback")))
    assert info.feedback_n == 3
    assert out[0].chunk_id == 2


# ======================================================================
# llm（验调用契约，不验模型质量）
# ======================================================================
class _FakeClient:
    def __init__(self, reply='{"order": [2, 1]}', boom=None, sleep=0.0):
        self.reply, self.boom, self.sleep, self.calls = reply, boom, sleep, []

    async def chat(self, messages, **kw):
        self.calls.append(kw)
        if self.sleep:
            await asyncio.sleep(self.sleep)
        if self.boom:
            raise self.boom
        return self.reply


def test_llm_rerank_reorders_by_model_order():
    hits = [_hit(1, "a", score=0.9), _hit(2, "b", score=0.5)]
    prof = RetrievalProfile(name="t", reranker="llm", rerank_top_n=5)
    out, info = _run(R.rerank(hits, "q", profile=prof, client=_FakeClient()))
    assert [h.chunk_id for h in out] == [2, 1]
    assert not info.fallback


def test_llm_rerank_call_contract():
    c = _FakeClient()
    prof = RetrievalProfile(name="t", reranker="llm")
    _run(R.rerank([_hit(1, "a"), _hit(2, "b")], "q", profile=prof, client=c))
    kw = c.calls[0]
    assert kw.get("fmt") == "json" and kw.get("temperature") == 0.0
    assert "model" not in kw or kw["model"] is None      # 复用常驻模型，不另拉


def test_llm_timeout_falls_back(monkeypatch):
    prof = RetrievalProfile(name="t", reranker="llm", rerank_timeout_ms=100)
    hits = [_hit(1, "a", score=0.9), _hit(2, "b", score=0.5)]
    out, info = _run(R.rerank(hits, "q", profile=prof,
                              client=_FakeClient(sleep=0.5)))
    assert info.fallback and "超时" in info.detail
    assert sorted(h.chunk_id for h in out) == [1, 2]      # 结果集不变


def test_llm_bad_json_falls_back():
    prof = RetrievalProfile(name="t", reranker="llm")
    out, info = _run(R.rerank([_hit(1, "a"), _hit(2, "b")], "q",
                              profile=prof, client=_FakeClient(reply="我不确定")))
    assert info.fallback and sorted(h.chunk_id for h in out) == [1, 2]


def test_llm_exception_falls_back():
    prof = RetrievalProfile(name="t", reranker="llm")
    out, info = _run(R.rerank([_hit(1, "a"), _hit(2, "b")], "q",
                              profile=prof, client=_FakeClient(boom=RuntimeError("模型不可用"))))
    assert info.fallback and sorted(h.chunk_id for h in out) == [1, 2]


def test_llm_order_completed_for_missing_ids():
    """模型少给编号时，漏掉的按原顺序补在后面 —— 绝不因此丢候选。"""
    hits = [_hit(i, f"c{i}") for i in (1, 2, 3)]
    prof = RetrievalProfile(name="t", reranker="llm")
    out, info = _run(R.rerank(hits, "q", profile=prof,
                              client=_FakeClient(reply='{"order": [3]}')))
    assert [h.chunk_id for h in out] == [3, 1, 2]
    assert not info.fallback


def test_llm_only_touches_top_n():
    hits = [_hit(i, f"c{i}") for i in range(1, 7)]
    prof = RetrievalProfile(name="t", reranker="llm", rerank_top_n=2)
    out, _ = _run(R.rerank(hits, "q", profile=prof,
                           client=_FakeClient(reply='{"order": [2, 1]}')))
    assert [h.chunk_id for h in out] == [2, 1, 3, 4, 5, 6]
