# 回填是 FR-3.5/AC-8 的落点：命中子块要按 parent_id 取回整条父块，
# 否则喂给模型的是缺头少尾的片段。逻辑部分可离线断言，
# 真回填必须查 Milvus（父块就在同一集合里，见②期设计 4.4）。
import pytest

from app.db.milvus import COLLECTION, get_client
from app.retrieval import backfill
from app.retrieval.backfill import BACKFILL_FIELDS, backfill_parents, parent_ids_of


def _milvus_available() -> bool:
    try:
        get_client().list_collections()
        return True
    except Exception:
        return False


def test_parent_ids_dedupes_and_keeps_order():
    hits = [{"parent_id": "b"}, {"parent_id": "a"}, {"parent_id": "b"}]
    assert parent_ids_of(hits) == ["b", "a"]


def test_parent_ids_skips_missing_and_father_hits():
    # 父块自身命中的话 parent_id 为 None；索引不齐时也可能缺键——
    # 两者都不该当成 id 去查，否则会发出 filter chunk_id in [None] 这种垃圾
    hits = [{"parent_id": None}, {"chunk_type": "father"}, {"parent_id": "a"}]
    assert parent_ids_of(hits) == ["a"]


def test_backfill_fields_include_text_and_path():
    # 提示词要原文、引用展示要路径，少一个下游就得回表
    assert "text" in BACKFILL_FIELDS
    assert "path" in BACKFILL_FIELDS


requires_milvus = pytest.mark.skipif(not _milvus_available(), reason="Milvus 不在线")


def _five_distinct_parent_hits(client) -> list[dict]:
    """取 5 条真实子块，且保证分属 5 个不同父块。

    排序与分批两处断言都需要「ids 恰好 5 个」这个结构：分批上限压到 2 时才真的
    跑满 3 批，反序传参时才分得清「跟随传入顺序」与「跟随 Milvus 返回行序」。
    取不到 5 个不同父块就断言失败而不是 skip——skip 会让判别力静默消失。
    """
    rows = client.query(COLLECTION, filter='chunk_type == "paragraph"',
                        output_fields=["chunk_id", "parent_id"], limit=10)
    hits: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        # 同一父块只留最早出现的子块：后来的同父块条目对上面两处断言无增量
        if row["parent_id"] not in seen:
            seen.add(row["parent_id"])
            hits.append(row)
        if len(hits) == 5:
            break
    assert len(hits) == 5, "库里 paragraph 子块应能凑出 5 个不同父块；取不到说明库被清过"
    return hits


@requires_milvus
def test_backfill_returns_parent_blocks_for_real_chunks():
    client = get_client()
    # 取两条真实子块当夹具：不许造数据，取不到就让测试失败而不是跳过
    hits = client.query(COLLECTION, filter='chunk_type == "paragraph"',
                        output_fields=["chunk_id", "parent_id", "article_no"],
                        limit=2)
    assert hits, "集合里应有 paragraph 子块；为空说明库被清过"
    blocks = backfill_parents(client, hits)
    assert len(blocks) == len({h["parent_id"] for h in hits})
    for block in blocks:
        assert block["chunk_type"] == "father"
        assert block["source"] == "vector"
        assert block["text"].strip(), "父块没有原文，回填等于没做"


@requires_milvus
def test_backfill_returns_empty_for_missing_parent():
    # 不存在的 parent_id 走真实查询（库里没这个 id 本身就是事实，不用造假数据）：
    # 实现必须安静跳过，而不是抛错或塞个 chunk_id 对得上的空壳进下游
    assert backfill_parents(get_client(), [{"parent_id": "0000000000000000"}]) == []


@requires_milvus
def test_backfill_skips_missing_parent_and_keeps_rank_gap():
    client = get_client()
    real = client.query(COLLECTION, filter='chunk_type == "paragraph"',
                        output_fields=["chunk_id", "parent_id"], limit=1)
    assert real, "集合里应有 paragraph 子块；为空说明库被清过"
    # 幽灵父块放前、真实父块放后：断言只回填真实块，且 rank_index 保留空档
    # （取 ids 里的位置 1，不压缩成 0）。语义如此的理由：rank_index 是「该父块的
    # 子块在候选里第几个不同父块位置出现」的编号，跳过缺块不重编号，后面父块的
    # rank 才不会被无关的缺块整体前移——一旦压缩，同一父块的 rank 会随库里缺
    # 不缺块而漂移，下游就没法拿它对齐粗排位置了
    hits = [{"parent_id": "0000000000000000"}, {"parent_id": real[0]["parent_id"]}]
    blocks = backfill_parents(client, hits)
    assert len(blocks) == 1
    assert blocks[0]["chunk_id"] == real[0]["parent_id"]
    assert blocks[0]["rank_index"] == 1
    assert blocks[0]["text"].strip(), "父块没有原文，回填等于没做"


@requires_milvus
def test_backfill_orders_by_first_child_occurrence():
    # 反序传参，断言输出严格跟随传入顺序：偷懒按 Milvus 返回行序（实测为主键序）
    # 产出的实现在这里必然对不上。原版靠「恰好取到不同父块」否则 skip，判别力
    # 会随数据静默消失；夹具函数保证 5 个不同父块，skip 分支就此消掉
    client = get_client()
    hits = list(reversed(_five_distinct_parent_hits(client)))
    expected: list[str] = []
    for hit in hits:
        if hit["parent_id"] not in expected:
            expected.append(hit["parent_id"])
    blocks = backfill_parents(client, hits)
    assert [b["chunk_id"] for b in blocks] == expected


@requires_milvus
def test_backfill_queries_in_batches(monkeypatch):
    # QUERY_BATCH=200 在 RECALL_TOPK=50 下永远只跑一批，分批循环是「改坏不会变红」
    # 的死代码；把上限压到 2，5 个真实父块刚好 3 批。只查第一批的实现会缺块，
    # 而只累加最后一批的实现同样对不上，一条断言同时守住两处
    client = get_client()
    hits = _five_distinct_parent_hits(client)
    monkeypatch.setattr(backfill, "QUERY_BATCH", 2)
    blocks = backfill_parents(client, hits)
    assert [b["chunk_id"] for b in blocks] == [h["parent_id"] for h in hits]


def test_backfill_on_empty_hits_returns_empty():
    # 空输入是纯逻辑短路，不碰 Milvus，所以这条不加 requires_milvus：离线时它
    # 是唯一还能跑的回填用例。哨兵 client 一旦被 query 就自爆，把「短路发生在
    # 查询之前」钉死；不用 get_client() 是因为离线时它先连不上，报的错与
    # 被测行为无关，断言会退化成「测网线」
    class _ExplodingClient:
        def query(self, *args, **kwargs):
            raise AssertionError("空 hits 不该发起 Milvus 查询")

    assert backfill_parents(_ExplodingClient(), []) == []
