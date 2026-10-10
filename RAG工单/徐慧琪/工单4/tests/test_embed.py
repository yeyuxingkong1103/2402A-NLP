# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
from unittest.mock import patch

import pytest

from rag04.config import get_settings
from rag04.retrieve.embed import embed_texts, embed_one, DIM, EmbedUnavailableError


def test_dim_constant_matches_bge_m3():
    assert DIM == 1024


def _fake_post(vecs):
    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"embeddings": vecs}

    return lambda *a, **k: Resp()


def test_embed_texts_returns_vectors():
    s = get_settings()
    with patch("rag04.retrieve.embed.requests.post",
               _fake_post([[0.1] * DIM, [0.2] * DIM])):
        out = embed_texts(["武汉力源", "组织结构图"], s)
    assert len(out) == 2
    assert len(out[0]) == DIM


def test_embed_texts_empty_input_returns_empty():
    s = get_settings()
    assert embed_texts([], s) == []


def test_embed_texts_batches_by_16():
    s = get_settings()
    calls = {"n": 0}

    def counting_post(url, **kw):
        calls["n"] += 1
        import json as _j
        payload = _j.loads(kw["data"])
        n = len(payload["input"])
        return _fake_post([[0.0] * DIM] * n)(url, **kw)

    with patch("rag04.retrieve.embed.requests.post", counting_post):
        out = embed_texts([f"文本{i}" for i in range(35)], s)
    assert len(out) == 35
    assert calls["n"] == 3, f"35 条应按 16 批量发 3 次，实际 {calls['n']}"


def test_embed_texts_falls_back_to_smaller_batch():
    """整批失败时降批次重试，而不是整体放弃。"""
    s = get_settings()
    seen = []

    def flaky_post(url, **kw):
        import json as _j
        n = len(_j.loads(kw["data"])["input"])
        seen.append(n)
        if n > 4:
            raise ConnectionError("batch too large")
        return _fake_post([[0.0] * DIM] * n)(url, **kw)

    with patch("rag04.retrieve.embed.requests.post", flaky_post):
        out = embed_texts([f"t{i}" for i in range(8)], s)
    assert len(out) == 8
    assert 1 in seen or 2 in seen or 4 in seen


def test_embed_one_returns_single_vector():
    s = get_settings()
    with patch("rag04.retrieve.embed.requests.post", _fake_post([[0.3] * DIM])):
        v = embed_one("武汉力源", s)
    assert len(v) == DIM


@pytest.mark.integration
def test_real_ollama_embedding():
    s = get_settings()
    v = embed_one("武汉力源信息技术股份有限公司", s)
    assert len(v) == DIM
    assert any(abs(x) > 1e-6 for x in v)
