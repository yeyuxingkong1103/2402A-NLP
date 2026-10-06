# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
tests/test_table_store.py —— 工单三表格存储单测

覆盖：
  1. TableStore 建表 / 字段完整
  2. insert_tables + search_tables 往返
  3. delete_by_doc_id 幂等
  4. 按 doc_id 过滤检索
  5. 集成：真实 data/tables/*.json 入库 + "发行股数" 检索
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.table_parser.table_store import TableStore, TABLE_COLLECTION_NAME
from src.embedding import get_embedder

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ================= 单元测试（用 Milvus Lite 临时库） =================
@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """工单三：用临时 Milvus Lite 库避免污染主库"""
    lite = tmp_path_factory.mktemp("milvus") / "test_tables.db"
    # 工单四：force_lite 真正隔离临时库，避免误连远程并 drop 生产 rag_tables
    s = TableStore(lite_path=str(lite), force_lite=True)
    s.drop_collection()  # 确保干净
    s.ensure_collection()
    yield s
    s.close()


def _fake_chunks(doc_id="test_doc", n=3):
    """工单三：构造假表格 chunk（含 table_text）"""
    return [
        {
            "table_chunk_id": f"tc_tbl_{i+1:03d}",
            "table_id": f"tbl_{i+1:03d}",
            "doc_id": doc_id,
            "page_range": [i + 1, i + 1],
            "caption": f"测试表{i+1}",
            "headers": ["项目", "金额"],
            "row_count": 2,
            "table_text": f"表标题: 测试表{i+1}\n列: 项目、金额\n行1: 发行股数 为 {1000*(i+1)}",
            "rows": [["发行股数", f"{1000*(i+1)}"]],
        }
        for i in range(n)
    ]


def _fake_vectors(n, dim=1024):
    """工单三：构造假向量（dim=1024）"""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dim)).astype(np.float32)


# ---------- 1. 建表 + 字段 ----------
def test_collection_created(store):
    stats = store.get_stats()
    assert stats["exists"] is True
    assert stats["collection"] == TABLE_COLLECTION_NAME


# ---------- 2. insert + search 往返 ----------
def test_insert_and_search(store):
    chunks = _fake_chunks(doc_id="test_insert", n=3)
    vecs = _fake_vectors(3)
    n = store.insert_tables(chunks, vecs)
    assert n == 3

    # 用第一条向量检索，应能召回自己
    hits = store.search_tables(vecs[0], top_k=3, doc_id="test_insert")
    assert len(hits) >= 1
    assert hits[0]["doc_id"] == "test_insert"
    assert "发行股数" in hits[0]["table_text"]


# ---------- 3. delete_by_doc_id 幂等 ----------
def test_delete_by_doc_id(store):
    chunks = _fake_chunks(doc_id="test_del", n=2)
    vecs = _fake_vectors(2)
    store.insert_tables(chunks, vecs)
    # 第一次删除
    store.delete_by_doc_id("test_del")
    hits = store.search_tables(vecs[0], top_k=5, doc_id="test_del")
    assert len(hits) == 0, "删除后应搜不到"
    # 第二次删除（幂等，不报错）
    store.delete_by_doc_id("test_del")


# ---------- 4. 按 doc_id 过滤 ----------
def test_search_filter_by_doc_id(store):
    chunks_a = _fake_chunks(doc_id="docA", n=2)
    chunks_b = _fake_chunks(doc_id="docB", n=2)
    vecs_a = _fake_vectors(2)
    vecs_b = _fake_vectors(2) + 1.0  # 错开向量
    store.insert_tables(chunks_a, vecs_a)
    store.insert_tables(chunks_b, vecs_b)

    # 只查 docA
    hits = store.search_tables(vecs_a[0], top_k=10, doc_id="docA")
    assert len(hits) >= 1
    assert all(h["doc_id"] == "docA" for h in hits), "doc_id 过滤失效"


# ---------- 5. 真实集成：data/tables/*.json 入库 + 检索 ----------
REAL_TABLES = [
    PROJECT_ROOT / "data" / "tables" / "招股说明书1_tables.json",
    PROJECT_ROOT / "data" / "tables" / "招股说明书2_tables.json",
]


def _real_data_available():
    return all(p.exists() for p in REAL_TABLES)


@pytest.mark.skipif(not _real_data_available(),
                    reason="data/tables/*.json 不存在（先跑 ingest_all_pdfs.py）")
def test_real_ingest_and_search(tmp_path):
    """工单三：真实表格入库 + '发行股数' 检索"""
    # 工单四：force_lite 隔离（人工智能NLP-RAG-图像内容解析及检索优化）
    store = TableStore(lite_path=str(tmp_path / "real_tables.db"), force_lite=True)
    store.drop_collection()
    store.ensure_collection()

    from src.table_parser.table_embedding import load_table_chunks_with_vectors
    # 入库两份
    total = 0
    for jf in REAL_TABLES:
        data = json.loads(jf.read_text(encoding="utf-8"))
        doc_id = data.get("doc_name") or jf.stem
        chunks, vecs = load_table_chunks_with_vectors(str(jf))
        for c in chunks:
            c.setdefault("doc_id", doc_id)
        store.delete_by_doc_id(doc_id)
        total += store.insert_tables(chunks, vecs)
    assert total >= 400, f"真实入库应 >=400 条，实际 {total}"

    # 检索"发行股数"
    embedder = get_embedder()
    qvec = embedder.encode(["发行股数"], show_progress_bar=False)[0]
    hits = store.search_tables(qvec, top_k=5)
    assert len(hits) == 5
    # top5 中至少一条含"发行股数"或"股数"
    found = any("发行股数" in h["table_text"] or "股数" in h["table_text"]
                for h in hits)
    assert found, f"top5 应含'发行股数'相关表，实际：{[h['table_text'][:60] for h in hits]}"
    store.close()
