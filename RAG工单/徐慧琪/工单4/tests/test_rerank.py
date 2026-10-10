# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
from unittest.mock import patch

import pytest

from rag04.config import Settings, get_settings
from rag04.schema import Hit
from rag04.retrieve.rerank import (
    heuristic_rerank, rerank, CrossEncoderReranker, get_reranker,
)


def _h(cid, text, score=1.0, bt="text", page=1):
    return Hit(chunk_id=cid, doc_id="d", page=page, block_type=bt,
               source_id=cid, text=text, score=score)


def test_heuristic_rerank_prefers_keyword_overlap():
    q = "大客户销售部下设几个销售处"
    hits = [
        _h("a", "募集资金投资项目情况说明", 0.9),
        _h("b", "大客户销售部下设北京销售处、深圳销售处等6个销售处", 0.5),
    ]
    out = heuristic_rerank(hits, q, top_k=2)
    assert out[0].chunk_id == "b", "关键词重合度高的应胜出"


def test_heuristic_rerank_respects_top_k():
    hits = [_h(f"c{i}", f"内容{i}") for i in range(10)]
    assert len(heuristic_rerank(hits, "内容", top_k=3)) == 3


def test_heuristic_rerank_empty():
    assert heuristic_rerank([], "q", top_k=5) == []


def test_heuristic_rerank_deterministic():
    hits = [_h("a", "相同文本", 0.5), _h("b", "相同文本", 0.5)]
    r1 = [h.chunk_id for h in heuristic_rerank(hits, "相同文本", 3)]
    r2 = [h.chunk_id for h in heuristic_rerank(hits, "相同文本", 3)]
    assert r1 == r2, "同分时排序必须稳定可复现"


def test_rerank_falls_back_when_model_unavailable():
    s = get_settings()
    hits = [_h("a", "销售部相关文本", 0.9), _h("b", "无关", 0.5)]
    with patch.object(CrossEncoderReranker, "_load", side_effect=FileNotFoundError("no model")):
        out = rerank(hits, "销售部", s, top_k=2)
    assert len(out) == 2
    assert out[0].chunk_id == "a"


def test_rerank_empty_returns_empty():
    assert rerank([], "q", get_settings()) == []


def test_cross_encoder_available_false_when_missing(tmp_path):
    s = get_settings()
    object.__setattr__(s, "models_dir", tmp_path)   # 空目录
    r = CrossEncoderReranker(s)
    assert r.available is False


def _fake_model_setup(tmp_path, monkeypatch):
    """建假模型目录 + 隔离缓存，返回 (settings, 加载计数列表)。

    `_load` 用「复刻真实惰性语义」的桩替代：只有 `_model is None` 时才计入
    一次加载，从而能区分「实例被复用」与「每次新建实例」。不加载真实模型。
    """
    from rag04.retrieve import rerank as rr_mod

    monkeypatch.setattr(rr_mod, "_RERANKER_CACHE", {})     # 不污染/不受全局缓存影响
    model_dir = tmp_path / "bge-reranker-base"
    model_dir.mkdir()
    (model_dir / "pytorch_model.bin").touch()              # 让 available 为真
    loads = []

    def counting_load(self):
        if self._model is None:
            loads.append(self)
            self._model = (object(), object(), "cpu")
        return self._model

    monkeypatch.setattr(CrossEncoderReranker, "_load", counting_load)

    def stub_rerank(self, hits, question, top_k):
        self._load()
        return list(hits[:top_k])

    monkeypatch.setattr(CrossEncoderReranker, "rerank", stub_rerank)
    return Settings(models_dir=tmp_path), loads


def test_rerank_reuses_cached_reranker(tmp_path, monkeypatch):
    """同一 settings 连续两次 rerank：底层模型只加载一次（缓存复用）。"""
    s, loads = _fake_model_setup(tmp_path, monkeypatch)
    hits = [_h("a", "销售部相关文本", 0.9), _h("b", "无关", 0.5)]

    out1 = rerank(hits, "销售部", s, top_k=2)
    out2 = rerank(hits, "销售部", s, top_k=2)

    assert len(out1) == len(out2) == 2
    assert len(loads) == 1, f"同一 settings 的模型应只加载一次，实际加载 {len(loads)} 次"


def test_reranker_cache_does_not_leak_across_models_dir(tmp_path, monkeypatch):
    """models_dir 不同的 Settings 必须各持实例，不得串用缓存。"""
    from rag04.retrieve import rerank as rr_mod

    monkeypatch.setattr(rr_mod, "_RERANKER_CACHE", {})
    s1 = Settings(models_dir=tmp_path / "m1")
    s2 = Settings(models_dir=tmp_path / "m2")

    r1 = get_reranker(s1)
    r2 = get_reranker(s2)

    assert r1 is not r2, "不同 models_dir 不得共享 reranker 实例"
    assert r1.model_dir != r2.model_dir
    assert get_reranker(s1) is r1, "同一 settings 必须复用实例"


def test_load_failure_marks_unavailable_and_keeps_fallback(tmp_path, monkeypatch):
    """加载失败后 available 转 False，且后续调用仍走启发式而非空结果。"""
    from rag04.retrieve import rerank as rr_mod

    monkeypatch.setattr(rr_mod, "_RERANKER_CACHE", {})
    model_dir = tmp_path / "bge-reranker-base"
    model_dir.mkdir()
    (model_dir / "pytorch_model.bin").touch()
    s = Settings(models_dir=tmp_path)

    def boom():
        raise RuntimeError("模拟加载失败")

    # 在真实 _load 的首个可失败步骤上注入异常，走真实 _load 的失败分支，
    # 从而验证「失败即置 _failed」是本类自己的行为而非测试桩的行为。
    monkeypatch.setattr(rr_mod, "_ensure_safe_import_order", boom)

    r = get_reranker(s)
    assert r.available is True
    with pytest.raises(RuntimeError):
        r._load()
    assert r._failed is True
    assert r.available is False, "加载失败后 available 必须转为 False（不再重试）"

    hits = [_h("a", "销售部相关文本", 0.9), _h("b", "无关", 0.5)]
    out = rerank(hits, "销售部", s, top_k=2)
    assert len(out) == 2, "模型不可用时应降级启发式，绝不能返回空结果"
    assert out[0].chunk_id == "a"


@pytest.mark.integration
def test_real_reranker_prefers_relevant():
    s = get_settings()
    hits = [
        _h("a", "本次发行股数占发行后总股本比例", 0.9),
        _h("b", "大客户销售部下设六个销售处包括北京、深圳、广州、成都、珠海、武汉", 0.3),
    ]
    out = rerank(hits, "大客户销售部有几个销售处", s, top_k=2)
    assert out[0].chunk_id == "b"
