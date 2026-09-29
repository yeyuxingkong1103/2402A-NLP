"""milvus_store.py 单元测试：mock MilvusClient，验证 store 层各函数（不连真实服务）。"""
from unittest.mock import MagicMock

import pytest

import milvus_store


@pytest.fixture
def fake_client(monkeypatch):
    """把 get_milvus_client 换成 MagicMock，返回该 mock 供断言。"""
    client = MagicMock()
    monkeypatch.setattr(milvus_store, "get_milvus_client", lambda: client)
    return client


def test_create_collection_builds_schema_and_index(fake_client):
    """新建：10 个字段、HNSW/COSINE 索引，建表后载入。"""
    fake_client.has_collection.return_value = False
    milvus_store.create_collection("c", dim=1024)

    schema = fake_client.create_schema.return_value
    names = [c.args[0] for c in schema.add_field.call_args_list]
    assert names == [
        "id", "embedding", "content", "page", "source", "domain",
        "created_at", "updated_at", "summary", "parent_content",
    ]

    idx = fake_client.prepare_index_params.return_value
    idx.add_index.assert_called_once()
    assert idx.add_index.call_args.kwargs["field_name"] == "embedding"
    assert idx.add_index.call_args.kwargs["index_type"] == "HNSW"
    assert idx.add_index.call_args.kwargs["metric_type"] == "COSINE"

    fake_client.create_collection.assert_called_once()
    fake_client.load_collection.assert_called_once_with("c")


def test_create_collection_existing_just_loads(fake_client):
    """已存在且字段齐全：不重建 schema，仅载入内存。"""
    fake_client.has_collection.return_value = True
    fake_client.describe_collection.return_value = {
        "fields": [{"name": n} for n in (
            "id", "embedding", "content", "page", "source", "domain",
            "created_at", "updated_at", "summary", "parent_content",
        )]
    }
    milvus_store.create_collection("c")
    fake_client.load_collection.assert_called_once_with("c")
    fake_client.create_collection.assert_not_called()


def test_create_collection_missing_fields_raises(fake_client):
    """已存在但缺新字段：抛错提示手动 drop 重建，不自动 drop。"""
    fake_client.has_collection.return_value = True
    fake_client.describe_collection.return_value = {
        "fields": [{"name": n} for n in ("id", "embedding", "content", "page", "source", "domain")]
    }
    with pytest.raises(RuntimeError):
        milvus_store.create_collection("c")
    fake_client.drop_collection.assert_not_called()


def test_insert_passes_records_and_flushes(fake_client):
    records = [{"id": "p1_c0", "embedding": [0.5] * 1024, "content": "x", "page": 1}]
    milvus_store.insert("c", records)
    fake_client.insert.assert_called_once_with("c", data=records)
    fake_client.flush.assert_called_once_with("c")


def test_search_normalizes_distance_as_similarity(fake_client):
    """Milvus COSINE 的 distance 即相似度，直接作为 similarity（不做 1-distance）。"""
    fake_client.search.return_value = [[
        {"id": "p3_c0", "distance": 0.9, "entity": {"content": "指南片段", "page": 3}},
    ]]
    hits = milvus_store.search("c", [0.5] * 1024, 5)
    assert hits == [{
        "id": "p3_c0", "content": "指南片段", "page": 3, "similarity": 0.9,
        "created_at": 0, "updated_at": 0, "summary": "", "parent_content": "",
    }]


def test_search_returns_new_scalar_fields(fake_client):
    """search 返回带上 created_at/updated_at/summary。"""
    fake_client.search.return_value = [[
        {
            "id": "p1_c0",
            "distance": 0.8,
            "entity": {
                "content": "x", "page": 1,
                "created_at": 1700000000, "updated_at": 1700000001, "summary": "摘要",
            },
        },
    ]]
    hits = milvus_store.search("c", [0.5] * 1024, 5)
    assert hits[0]["created_at"] == 1700000000
    assert hits[0]["updated_at"] == 1700000001
    assert hits[0]["summary"] == "摘要"


def test_search_returns_parent_content(fake_client):
    """search 返回带上 parent_content（父子分块的冗余父块字段）。"""
    fake_client.search.return_value = [[
        {
            "id": "p1_c0",
            "distance": 0.8,
            "entity": {"content": "子块", "page": 1, "parent_content": "父块全文"},
        },
    ]]
    hits = milvus_store.search("c", [0.5] * 1024, 5)
    assert hits[0]["parent_content"] == "父块全文"


def test_search_empty_result(fake_client):
    fake_client.search.return_value = [[]]
    assert milvus_store.search("c", [0.5] * 1024, 5) == []


def test_search_collection_not_found_returns_empty(fake_client):
    """collection 不存在时降级返回 []，而非抛 MilvusException。"""
    from pymilvus.exceptions import MilvusException

    fake_client.search.side_effect = MilvusException(
        message="collection not found[collection=psychology_guide]"
    )
    assert milvus_store.search("psychology_guide", [0.5] * 1024, 5) == []


def test_delete_by_source_filters_and_flushes(fake_client):
    milvus_store.delete_by_source("c", "guide.pdf")
    fake_client.delete.assert_called_once_with("c", filter='source == "guide.pdf"')
    fake_client.flush.assert_called_once_with("c")


def test_query_all_iterates_batches(fake_client):
    """QueryIterator 用 .next() 逐批拉取，直到返回空为止。"""
    it = MagicMock()
    it.next.side_effect = [
        [{"id": "a", "content": "x", "page": 1, "source": "s", "domain": ""}],
        [{"id": "b", "content": "y", "page": 2, "source": "s", "domain": ""}],
        None,
    ]
    fake_client.query_iterator.return_value = it
    rows = milvus_store.query_all("c")
    assert [r["id"] for r in rows] == ["a", "b"]


def test_list_documents_aggregates_by_source(fake_client):
    """按 source 聚合 chunk 数与最小 created_at。"""
    it = MagicMock()
    it.next.side_effect = [
        [
            {"source": "a.pdf", "created_at": 100},
            {"source": "a.pdf", "created_at": 200},
            {"source": "b.pdf", "created_at": 300},
        ],
        None,
    ]
    fake_client.query_iterator.return_value = it
    docs = milvus_store.list_documents("c")
    assert docs == [
        {"source": "a.pdf", "chunk_count": 2, "created_at": 100},
        {"source": "b.pdf", "chunk_count": 1, "created_at": 300},
    ]


def test_drop_and_list_pass_through(fake_client):
    fake_client.list_collections.return_value = ["a", "b"]
    assert milvus_store.list_collections() == ["a", "b"]
    milvus_store.drop_collection("a")
    fake_client.drop_collection.assert_called_once_with("a")
