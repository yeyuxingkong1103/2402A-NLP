# -*- coding: utf-8 -*-
"""Milvus 稀疏字段与稀疏检索的集成测试

需要 Milvus 在 settings.milvus_host:settings.milvus_port 运行。
使用独立的临时集合，测试结束后清理。

⚠️ 安全约定：本文件绝不触碰真实集合 cs_kb_chunks。
   该 Milvus 实例上还共存着其它项目的 6 个集合，务必不要误删。
"""

import pytest

from backend.config import settings

TEMP_COLLECTION = "cs_rag_test_sparse"


def _milvus_available() -> bool:
    try:
        from backend.db import milvus_client

        return milvus_client.health_check()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _milvus_available(), reason="Milvus 不可用，跳过集成测试"
)


@pytest.fixture
def temp_collection():
    """建一个启用稀疏字段的临时集合，用完即删"""
    from backend.db import milvus_client

    client = milvus_client.get_client()
    milvus_client.create_collection(TEMP_COLLECTION, enable_sparse=True)
    yield TEMP_COLLECTION
    if client.has_collection(TEMP_COLLECTION):
        client.drop_collection(TEMP_COLLECTION)


def test_启用稀疏字段后集合含_sparse_embedding(temp_collection):
    from backend.db import milvus_client

    desc = milvus_client.get_client().describe_collection(temp_collection)
    names = {f["name"] for f in desc["fields"]}
    assert "embedding" in names
    assert "sparse_embedding" in names


def test_默认不建稀疏字段():
    """回归保护：enable_sparse 默认 False，不影响 V1 行为"""
    from backend.db import milvus_client

    name = TEMP_COLLECTION + "_dense_only"
    client = milvus_client.get_client()
    try:
        milvus_client.create_collection(name)
        desc = client.describe_collection(name)
        names = {f["name"] for f in desc["fields"]}
        assert "sparse_embedding" not in names
    finally:
        if client.has_collection(name):
            client.drop_collection(name)


def test_稀疏写入后可被检索到(temp_collection):
    from backend.db import milvus_client

    sparse_a = {101: 0.9, 102: 0.5}
    sparse_b = {201: 0.8, 202: 0.3}
    # insert_vectors 签名（已核实）：
    #   insert_vectors(chunk_ids, doc_ids, page_nos, contents, embeddings, *,
    #                  sparse_embeddings=None, name=None, batch_size=256)
    # 集合名关键字是 name=，不是 collection_name=
    milvus_client.insert_vectors(
        ["t_a", "t_b"],
        ["d1", "d1"],
        [1, 2],
        ["块A", "块B"],
        [[0.1] * settings.milvus_dim, [0.2] * settings.milvus_dim],
        sparse_embeddings=[sparse_a, sparse_b],
        name=temp_collection,
    )
    hits = milvus_client.search_sparse(sparse_a, top_k=2, name=temp_collection)
    # 只返回与查询**有重叠维度**的块：t_b 的词元 {201,202} 与查询 {101,102}
    # 完全不重叠，内积为 0，会被 Milvus 直接丢弃（这是稀疏检索的正确语义）
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "t_a"
    assert hits[0]["page_no"] == 1
    assert "score" in hits[0]
    assert hits[0]["score"] > 0


def test_无重叠维度的块不被返回(temp_collection):
    """
    稀疏检索的语义：词元完全不重叠的块内积为 0，不会进入结果集。

    这条语义对混合检索很关键 —— 稀疏通道只提供「字面命中」的信号，
    它召回不了同义改写的内容，这正是需要稠密通道互补的原因。
    """
    from backend.db import milvus_client

    milvus_client.insert_vectors(
        ["s_a", "s_b"],
        ["d1", "d1"],
        [1, 2],
        ["有重叠", "无重叠"],
        [[0.1] * settings.milvus_dim, [0.2] * settings.milvus_dim],
        sparse_embeddings=[{501: 0.7}, {999: 0.9}],   # 两者无任何共同词元
        name=temp_collection,
    )
    hits = milvus_client.search_sparse({501: 0.5}, top_k=10, name=temp_collection)
    ids = [h["chunk_id"] for h in hits]
    assert ids == ["s_a"], f"只应返回有词元重叠的块，实际={ids}"


def test_稀疏检索返回结构含页码(temp_collection):
    from backend.db import milvus_client

    milvus_client.insert_vectors(
        ["t_c"],
        ["d2"],
        [7],
        ["块C"],
        [[0.3] * settings.milvus_dim],
        sparse_embeddings=[{301: 1.0}],
        name=temp_collection,
    )
    hits = milvus_client.search_sparse({301: 1.0}, top_k=1, name=temp_collection)
    assert hits[0]["page_no"] == 7
    assert hits[0]["doc_id"] == "d2"
    assert hits[0]["chunk_id"] == "t_c"


def test_超长中文_content_写入时被截断到字节上限(temp_collection):
    """
    回归测试：中文 content 超长时必须按 UTF-8 **字节**截断。

    背景（这是一个真实修复过的 bug）：
        Milvus 的 VARCHAR(max_length=8192) 以 UTF-8 字节计数，而旧实现用的是
        Python 字符切片 content[:8192]。一个中文字符占 3 字节，因此
        「长」*20000 经旧实现切片后仍有 24576 字节，Milvus 直接拒绝整条插入：

            MilvusException(code=1100, length: 24576, max length: 8192)

        当前管线因 CHUNK_MAX_CHARS=2000（=6000 字节）碰不到该阈值，
        阈值实为 2730 个中文字符 —— 一旦调大配置就会整体写不进去。
    """
    from backend.db import milvus_client

    long_text = "长" * 20000          # 远超 _MAX_LEN_CONTENT(8192)
    milvus_client.insert_vectors(
        ["t_long"],
        ["d3"],
        [9],
        [long_text],
        [[0.4] * settings.milvus_dim],
        sparse_embeddings=[{401: 1.0}],
        name=temp_collection,
    )
    hits = milvus_client.search_sparse({401: 1.0}, top_k=1, name=temp_collection)
    assert len(hits) == 1
    stored = hits[0]["content"]
    assert len(stored.encode("utf-8")) <= 8192, (
        f"content 的 UTF-8 字节数应不超过字段上限，实际={len(stored.encode('utf-8'))}"
    )
