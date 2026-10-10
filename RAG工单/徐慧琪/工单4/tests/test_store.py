# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
import pytest

from rag04.config import get_settings
from rag04.schema import Chunk
from rag04.ingest.store import (
    VectorStore, COLL_TEXT, COLL_TABLE, COLL_IMAGE, collection_for,
)
from rag04.retrieve.embed import DIM

# image_chunks 是 CLIP 512 维；文本/表格才是 bge-m3 的 1024 维（约束3）。
CLIP_DIM = 512


def _c(i, text, bt="text", page=1, extra=None):
    return Chunk(chunk_id=f"c{i}", doc_id="d", page=page, block_type=bt,
                 source_id=f"s{i}", text=text, extra=extra or {})


@pytest.fixture
def store(tmp_path):
    s = get_settings()
    st = VectorStore(s, path=tmp_path / "qdrant")
    st.ensure_collections()
    yield st
    st.close()


def test_collection_for_maps_block_types():
    assert collection_for("text") == COLL_TEXT
    assert collection_for("table") == COLL_TABLE
    assert collection_for("image") == COLL_IMAGE


def test_collection_names_are_three_separate():
    assert len({COLL_TEXT, COLL_TABLE, COLL_IMAGE}) == 3


def test_ensure_collections_is_idempotent(tmp_path):
    s = get_settings()
    st = VectorStore(s, path=tmp_path / "q2")
    st.ensure_collections()
    st.ensure_collections()   # 二次调用不应抛错
    st.close()


def test_clear_collections_removes_stale_points(store):
    """RC-2 重建安全：分块改动会改变 chunk_id，重建前必须清库，否则新旧共存。

    回归：嵌入式 delete_collection 在 Windows 上静默残留旧 sqlite（重开客户端
    依旧读到旧点），故必须用空过滤器删点。
    """
    store.upsert_chunks([_c(0, "旧块"), _c(1, "旧块2")],
                        [[1.0] + [0.0] * (DIM - 1)] * 2)
    assert store.counts()[COLL_TEXT] == 2

    before = store.clear_collections()
    assert before[COLL_TEXT] == 2, "应返回清空前的点数快照"
    assert store.counts() == {COLL_TEXT: 0, COLL_TABLE: 0, COLL_IMAGE: 0}
    # 清空后集合仍在且可继续写入（维度不变），旧点不得复活
    store.upsert_chunks([_c(0, "新块")], [[1.0] + [0.0] * (DIM - 1)])
    assert store.counts()[COLL_TEXT] == 1
    hits = store.search(COLL_TEXT, [1.0] + [0.0] * (DIM - 1), k=5)
    assert [h.chunk_id for h in hits] == ["c0"] and hits[0].text == "新块"


def test_search_returns_dedup_key_from_payload(store):
    """RC2：图描述文本块的 dedup_key（=image 块 id）必须随 payload 取回。"""
    store.upsert_chunks(
        [_c(0, "组织结构图描述", extra={"dedup_key": "img-1"})],
        [[1.0] + [0.0] * (DIM - 1)])
    hits = store.search(COLL_TEXT, [1.0] + [0.0] * (DIM - 1), k=1)
    assert hits[0].dedup_key == "img-1"


def test_upsert_and_search_roundtrip(store):
    chunks = [_c(0, "销售部下设4个部门"), _c(1, "募集资金投资项目")]
    vecs = [[1.0] + [0.0] * (DIM - 1), [0.0, 1.0] + [0.0] * (DIM - 2)]
    n = store.upsert_chunks(chunks, vecs)
    assert n == 2

    hits = store.search(COLL_TEXT, [1.0] + [0.0] * (DIM - 1), k=2)
    assert hits
    assert hits[0].chunk_id == "c0"
    assert hits[0].block_type == "text"


def test_upsert_routes_to_correct_collection(store):
    store.upsert_chunks(
        [_c(0, "图描述", bt="image", extra={"image_path": "a.png"})],
        [[1.0] + [0.0] * (CLIP_DIM - 1)],
    )
    assert store.counts()[COLL_IMAGE] == 1
    assert store.counts()[COLL_TEXT] == 0


def test_image_hit_carries_image_path(store):
    store.upsert_chunks(
        [_c(0, "组织结构图描述", bt="image", extra={"image_path": "figs/a.png"})],
        [[1.0] + [0.0] * (CLIP_DIM - 1)],
    )
    h = store.search(COLL_IMAGE, [1.0] + [0.0] * (CLIP_DIM - 1), k=1)[0]
    assert h.image_path == "figs/a.png"


def test_upsert_length_mismatch_raises(store):
    with pytest.raises(ValueError, match="数量"):
        store.upsert_chunks([_c(0, "a"), _c(1, "b")], [[0.0] * DIM])


def test_counts_empty_store(store):
    assert store.counts() == {COLL_TEXT: 0, COLL_TABLE: 0, COLL_IMAGE: 0}


def test_upsert_is_idempotent_on_same_chunk_id(store):
    v = [[1.0] + [0.0] * (DIM - 1)]
    store.upsert_chunks([_c(0, "内容")], v)
    store.upsert_chunks([_c(0, "内容")], v)
    assert store.counts()[COLL_TEXT] == 1, "相同 chunk_id 应覆盖而非重复"
