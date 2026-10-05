# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化 —— 向量库测试"""
import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """隔离的本地 Qdrant（临时目录）。"""
    from src import config, vector_store as vs
    monkeypatch.setattr(config, "QDRANT_LOCAL_PATH", str(tmp_path))
    monkeypatch.setattr(config, "QDRANT_MODE", "local")
    vs.reset_client_for_tests()
    yield vs
    vs.reset_client_for_tests()


def _chunk(cid="c000001", **kw):
    base = {"chunk_id": cid, "text": "注册资本为7,360.00万元", "parent_id": "s000001",
            "parent_text": "第三节 公司基本情况……注册资本为7,360.00万元……",
            "parent_page_idx": 10, "page_idx": 10, "heading_path": ["第三节", "基本情况"],
            "source": "招股说明书1.pdf", "block_type": "text", "seq": 0}
    base.update(kw)
    return base


def test_upsert_search_roundtrip_metadata(store):
    store.ensure_collection(dim=4, recreate=True)
    assert store.upsert_chunks([_chunk()], [[1.0, 0.0, 0.0, 0.0]]) == 1
    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    h = hits[0]
    assert h["chunk_id"] == "c000001"
    assert h["page_idx"] == 10
    assert h["heading_path"] == ["第三节", "基本情况"]
    assert h["source"] == "招股说明书1.pdf"
    assert h["block_type"] == "text"
    assert h["parent_id"] == "s000001"
    assert "注册资本" in h["parent_text"]
    assert h["score"] == pytest.approx(1.0, abs=1e-5)


def test_recreate_purges_old_points(store):
    """重建后不得残留旧点（本地模式已知坑）。"""
    store.ensure_collection(dim=4, recreate=True)
    store.upsert_chunks([_chunk("c000001"), _chunk("c000002")],
                        [[1, 0, 0, 0], [0, 1, 0, 0]])
    assert store.count() == 2
    store.ensure_collection(dim=4, recreate=True)
    assert store.count() == 0
    store.upsert_chunks([_chunk("c000003")], [[1, 0, 0, 0]])
    assert store.count() == 1


def test_scroll_all_returns_payloads(store):
    store.ensure_collection(dim=4, recreate=True)
    store.upsert_chunks([_chunk("c000001"), _chunk("c000002", block_type="table")],
                        [[1, 0, 0, 0], [0, 1, 0, 0]])
    rows = store.scroll_all()
    assert len(rows) == 2
    assert {r["chunk_id"] for r in rows} == {"c000001", "c000002"}


def test_count_and_info(store):
    store.ensure_collection(dim=4, recreate=True)
    store.upsert_chunks([_chunk()], [[1, 0, 0, 0]])
    assert store.count() == 1
    assert store.info()["points"] == 1


# ---------------------------------------------------------------------------
# 工单3：Qdrant 点 ID 的多文档唯一性
# ---------------------------------------------------------------------------
def test_stable_id_unique_across_documents():
    """同一 chunk_id 在两个文档下必须得到不同的点 ID（否则会互相覆盖）。"""
    from src.vector_store import _stable_id
    a = _stable_id("招股说明书1.pdf#c000001")
    b = _stable_id("招股说明书2.pdf#c000001")
    assert a != b
    assert a > 0 and b > 0


def test_stable_id_deterministic():
    """跨进程稳定：同一键必须每次得到同一个 ID（不能用内置 hash）。"""
    from src.vector_store import _stable_id
    assert _stable_id("招股说明书1.pdf#c000123") == _stable_id("招股说明书1.pdf#c000123")
